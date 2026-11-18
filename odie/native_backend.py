"""Persistent real TCNN optimizer with graph recapture on support/LR changes."""
import ctypes
import json
import torch


class PlaneState:
    def __init__(self, backend, planes):
        self.backend = backend
        self.ranges = {}
        offset = 0
        for plane in planes:
            self.ranges[plane] = (offset, offset + plane.numel())
            offset += plane.numel()

    def get(self, plane, default=None):
        if plane not in self.ranges:
            return default
        value = self.backend.state_summary(*self.ranges[plane])
        return {} if value['pristine'] else {'native_adam_state': True}


class NativeOptimizer:
    def __init__(self, model, target, shape, config, planes, half=True):
        from .build import native_library
        library = native_library(config)
        self.build_manifest = json.loads((library.parent/'build_manifest.json').read_text())
        self.model, self.target, self.config = model, target, config
        self.half = half
        self.graph = self.key = None
        self.recaptures = 0
        self.lib = ctypes.CDLL(str(library))
        lib = self.lib
        lib.odie_create.argtypes = [ctypes.c_char_p]; lib.odie_create.restype = ctypes.c_void_p
        lib.odie_step.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        lib.odie_configure.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.odie_pointers.argtypes = [ctypes.c_void_p]; lib.odie_pointers.restype = ctypes.c_char_p
        lib.odie_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; lib.odie_state.restype = ctypes.c_char_p
        lib.odie_pin_contexts.argtypes = [ctypes.c_void_p]
        lib.odie_release_contexts.argtypes = [ctypes.c_void_p]
        lib.odie_destroy.argtypes = [ctypes.c_void_p]
        lib.odie_sync_weights.argtypes = [ctypes.c_void_p]
        lib.odie_error.restype = ctypes.c_char_p
        lib.odie_copy.argtypes = [ctypes.c_uint64] * 4
        self.stream = torch.cuda.current_stream()
        self.param_groups = [dict(name='grid', params=planes, lr=config['grid_lr'])]
        if model.gate is not None:
            self.param_groups.append(dict(name='gate', params=list(model.gate.parameters()), lr=config['gate_lr']))
        self.param_groups.append(dict(name='decoder', params=list(model.decoder.parameters()), lr=config['decoder_lr']))
        settings = dict(stream=self.stream.cuda_stream, total=model.grid.numel(),
                        width=shape[1], height=shape[0], mode=20 if half else 16, warp_limit=128,
                        rgb=target.data_ptr(), grid=model.grid.data_ptr(),
                        decoder=model.decoder.params.data_ptr(),
                        **self.current_settings())
        if model.gate is not None:
            settings['gate'] = model.gate.params.data_ptr()
        if config['representation'] == 'hash_grid':
            settings['hash_encoding'] = config['hash_encoding']
        self.handle = lib.odie_create(json.dumps(settings).encode())
        if not self.handle:
            raise RuntimeError(lib.odie_error().decode())
        self.pointers = json.loads(lib.odie_pointers(self.handle))
        self.fixed = torch.zeros(65536, device='cuda', dtype=torch.int64)
        self.squares = torch.empty(65536, device='cuda', dtype=torch.float32)
        self.state = PlaneState(self, planes)
        self.warmed = False

    def current_settings(self):
        if self.config['representation'] == 'hash_grid':
            return dict(active=self.model.active_prefix,
                        **{g['name']+'_lr': float(g['lr']) for g in self.param_groups})
        return dict(active=self.model.active_prefix,
                    **{k: getattr(self.model, k).data_ptr() for k in ('descriptors', 'node_bitmap', 'node_prefix', 'bitmap')},
                    **{g['name']+'_lr': float(g['lr']) for g in self.param_groups})

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.odie_error().decode())

    def step_raw(self, ids, update=True):
        self.check(self.lib.odie_step(self.handle, ids.data_ptr(), ids.numel(), int(update)))

    def copied(self, name, shape, dtype):
        result = torch.empty(shape, device='cuda', dtype=dtype)
        self.check(self.lib.odie_copy(result.data_ptr(), self.pointers[name], result.numel()*result.element_size(), self.stream.cuda_stream))
        return result

    def release_graph(self):
        torch.cuda.synchronize()
        self.graph = self.key = None
        self.lib.odie_release_contexts(self.handle)

    def prepare_epoch(self):
        settings = self.current_settings()
        key = tuple(settings.items())
        if key == self.key:
            return
        self.release_graph()
        self.check(self.lib.odie_configure(self.handle, json.dumps(settings).encode()))
        # Allocate TCNN forward/backward contexts without changing parameters or Adam state.
        # A new TCNN context is allocated before its preceding context is released.
        # Repeated dry runs reserve both live arena allocations before graph capture.
        for _ in range(3):
            self.step_raw(self.fixed, update=False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.step_raw(self.fixed)
            self.check(self.lib.odie_copy(self.squares.data_ptr(), self.pointers['squares'], 65536*4, self.stream.cuda_stream))
            self.graph_loss = self.squares.double().mean()
        self.lib.odie_pin_contexts(self.handle)
        self.graph, self.key = graph, key
        self.recaptures += 1

    def train_batch(self, ids):
        if ids.numel() == 65536:
            self.fixed.copy_(ids)
            self.graph.replay()
            return self.graph_loss
        self.step_raw(ids)
        self.check(self.lib.odie_copy(self.squares.data_ptr(), self.pointers['squares'], ids.numel()*4, self.stream.cuda_stream))
        return self.squares[:ids.numel()].double().mean()

    def state_summary(self, begin=-1, end=-1):
        data = self.lib.odie_state(self.handle, begin, end)
        if data is None:
            raise RuntimeError(self.lib.odie_error().decode())
        return json.loads(data)

    def native_state_digest(self):
        return json.dumps(self.state_summary(), sort_keys=True)

    def close(self):
        self.release_graph()
        self.lib.odie_destroy(self.handle)
        self.handle = None


def make_native_optimizer(model, target, shape, config):
    planes = [torch.nn.Parameter(model.grid[level['plane_offset']:level['plane_offset'] +
              level['plane_node_capacity']*level['plane_dim']].detach()) for level in model.layout['levels']]
    return NativeOptimizer(model, target, shape, config, planes), planes
