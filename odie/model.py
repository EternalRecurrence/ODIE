"""Shared configurable packed-grid model; reference autograd and native evaluation."""
from copy import deepcopy
from hashlib import sha256
import math
import os
from pathlib import Path

import torch
from torch import nn

EXTENSION = None
COORDINATE_VERSION = 'unit-endpoints-native-fmaf-half-cell-align8-modulo-v1'
LAYOUT_VERSION = 'packed-xy-only-native-closure-v1'
ADDRESSING_VERSION = 'node-bitmap64-exclusive-prefix32-v1'


def router_weights(logits, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(logits)
    if activation == 'softplus':
        return torch.nn.functional.softplus(logits, beta=1., threshold=20.)
    raise ValueError('Unknown router activation: ' + activation)


def gate_coarse_levels(config):
    """K changes constant-one gates only; full-cover training prefix is separate."""
    count = int(config.get('gate_coarse_levels_K', config['coarse_levels']))
    if not 0 <= count <= int(config['n_levels']) or int(config['coarse_levels']) != count:
        raise ValueError('Gate K must agree with coarse_levels and remain within declared level count')
    return count


def align8(value):
    return (int(value) + 7) // 8 * 8


def plane_closure(support, resolution):
    cells = support.nonzero(as_tuple=True)[0]
    return torch.unique(torch.cat([cells, cells + 1, cells + resolution,
                                   cells + resolution + 1]).remainder(align8(resolution**2)), sorted=True)


def make_node_index(nodes, points):
    """CPU node presence and per-word exclusive rank, in sorted native order."""
    if not isinstance(points, int) or points <= 0 or points > torch.iinfo(torch.int32).max:
        raise ValueError('Native node universe must fit positive int32 addressing')
    if nodes.device.type != 'cpu' or nodes.dtype != torch.int64 or nodes.ndim != 1:
        raise ValueError('Nodes must be a flat CPU int64 tensor')
    if nodes.numel() and (int(nodes[0]) < 0 or int(nodes[-1]) >= points or
                          bool((nodes[1:] <= nodes[:-1]).any())):
        raise ValueError('Nodes must be sorted, unique and inside the native universe')
    words = (points + 63) // 64
    present = torch.zeros(words * 64, dtype=torch.bool)
    present[nodes] = True
    rows = present.reshape(words, 64)
    # int64 bit 63 is negative; view preserves the raw uint64 bit pattern.
    bits = torch.ones(64, dtype=torch.int64) << torch.arange(64)
    bitmap = (rows.to(torch.int64) * bits).sum(1).view(torch.uint64)
    counts = rows.sum(1)
    prefix = (counts.cumsum(0) - counts).to(torch.int32)
    return bitmap, prefix


def make_layout(config, supports, plane_dims, plane_node_caps):
    resolutions = list(config['resolutions'])
    if not (len(resolutions) == len(supports) == len(plane_dims) == len(plane_node_caps) == int(config['n_levels'])) or not all(f in (1, 2, 4) for f in plane_dims):
        raise ValueError('Require aligned level descriptors and F1/F2/F4 features')
    if config['representation'] != 'xy_planes':
        raise ValueError('XY-only representation is required')
    gate_coarse_levels(config)
    if config['gate_topology'] != 'shared' or config['gate_position'] != 'feature':
        raise ValueError('Preserve the shared feature gate')
    node_bitmaps, prefixes, bitmaps, descriptors, levels, clean = [], [], [], [], [], []
    params = feature_offset = node_word_offset = bitmap_offset = 0
    for level, (resolution, support, dim, cap) in enumerate(zip(resolutions, supports, plane_dims, plane_node_caps)):
        if support.device.type != 'cpu' or support.dtype != torch.bool or support.shape != (resolution**2,):
            raise ValueError('Support must be a flat CPU bool R^2 tensor')
        if level < int(config['initial_active_prefix']) and not bool(support.all()):
            raise ValueError('Declared initial full-cover supports must remain full')
        support = support.clone()
        points = align8(resolution**2)
        nodes = plane_closure(support, resolution)
        if cap % 8 or not nodes.numel() <= cap <= points:
            raise ValueError('Aligned node capacity must contain the complete interpolation closure')
        if points > torch.iinfo(torch.int32).max or params + cap * dim > torch.iinfo(torch.int32).max:
            raise ValueError('Native points and packed offsets must fit int32 addressing')
        # Sorted unique closure with exactly `points` entries is identity order.
        direct = nodes.numel() == points
        if direct:
            index_offset = -1
        else:
            node_bitmap, prefix = make_node_index(nodes, points)
            index_offset = node_word_offset
            node_bitmaps.append(node_bitmap)
            prefixes.append(prefix)
            node_word_offset += node_bitmap.numel()
        padded = torch.zeros(points, dtype=torch.int64)
        padded[:resolution**2] = support.to(torch.int64)
        bitmap = (padded.reshape(-1, 8) * (1 << torch.arange(8))).sum(1).to(torch.uint8)
        # R, points, plane offset, node word offset, cell bitmap offset, Fxy, output offset, direct.
        descriptors.append([resolution, points, params, index_offset, bitmap_offset, dim, feature_offset, int(direct)])
        levels.append({'level': level, 'resolution': resolution, 'plane_dim': dim,
                       'allowed_cells': int(support.sum()), 'plane_retained_nodes': nodes.numel(),
                       'plane_node_capacity': cap, 'plane_unused_slots': cap - nodes.numel(),
                       'native_plane_points': points, 'plane_offset': params,
                       'feature_offset': feature_offset, 'nodes': nodes,
                       'node_index_words': 0 if direct else node_bitmap.numel(),
                       'direct_node_access': direct})
        bitmaps.append(bitmap)
        clean.append(support)
        params += cap * dim
        feature_offset += dim
        bitmap_offset += bitmap.numel()
    return {'descriptors': torch.tensor(descriptors, dtype=torch.int32),
            'node_bitmap': torch.cat(node_bitmaps) if node_bitmaps else torch.empty(0, dtype=torch.uint64),
            'node_prefix': torch.cat(prefixes) if prefixes else torch.empty(0, dtype=torch.int32),
            'bitmap': torch.cat(bitmaps), 'levels': levels, 'supports': clean,
            'params': params, 'input_width': feature_offset}


def cuda_extension():
    global EXTENSION
    if EXTENSION is None:
        from torch.utils.cpp_extension import load
        source = Path(__file__).resolve().parent
        files = [source / 'packed_bindings.cpp', source / 'packed_kernels.cu']
        digest = sha256(b''.join(path.read_bytes() for path in files)).hexdigest()[:12]
        build = Path(os.environ.get('ODIE_BUILD_CACHE', str(Path.home()/'.cache/odie'))) / ('query-' + digest)
        build.mkdir(parents=True, exist_ok=True)
        EXTENSION = load(name='odie_bitmap_rank_' + digest, sources=[str(p) for p in files],
                         build_directory=str(build), extra_cflags=['-O3'],
                         extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-lineinfo'],
                         verbose=os.environ.get('ODIE_BUILD_VERBOSE') == '1')
    return EXTENSION


class PackedQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coordinates, grid, descriptors, node_bitmap, node_prefix, bitmap, width):
        features, visible = cuda_extension().forward(coordinates, grid, descriptors, node_bitmap, node_prefix, bitmap, width)
        ctx.save_for_backward(coordinates, descriptors, node_bitmap, node_prefix, bitmap)
        ctx.param_count = grid.numel()
        ctx.mark_non_differentiable(visible)
        return features, visible

    @staticmethod
    def backward(ctx, feature_gradient, visibility_gradient):
        coordinates, descriptors, node_bitmap, node_prefix, bitmap = ctx.saved_tensors
        gradient = cuda_extension().backward(coordinates, feature_gradient.contiguous(),
                                              descriptors, node_bitmap, node_prefix, bitmap, ctx.param_count)
        return None, gradient, None, None, None, None, None


class PackedXYPlanes(nn.Module):
    def __init__(self, config, supports, plane_dims, plane_node_caps, axis_mode='none', kernel_a=0., kernel_omega=1.5 * math.pi):
        super().__init__()
        if axis_mode != 'none' or kernel_a != 0.:
            raise ValueError('No axis branches or non-linear interpolation kernel in this ablation')
        self.config = deepcopy(config)
        self.router_enabled = config.get('router_enabled', True)
        self.gate_coarse_levels_K = gate_coarse_levels(config)
        self.layout = make_layout(config, supports, plane_dims, plane_node_caps)
        self.supports = self.layout['supports']
        self.plane_dims, self.plane_node_caps = list(plane_dims), list(plane_node_caps)
        self.input_width = self.layout['input_width']
        self.n_levels = int(config['n_levels'])
        device = torch.device('cuda')
        import tinycudann as tcnn
        seed = config['seed']
        self.grid = nn.Parameter(torch.zeros(self.layout['params'], dtype=torch.float32, device=device))
        self.initialized_planes = [False] * self.n_levels
        with torch.no_grad():
            for level, item in enumerate(self.layout['levels']):
                if item['nodes'].numel():
                    self._initialize_plane(level)
        for name in ('descriptors', 'node_bitmap', 'node_prefix', 'bitmap'):
            self.register_buffer(name, self.layout[name].to(device), persistent=False)
        self.layout.pop('node_bitmap')
        self.layout.pop('node_prefix')
        self.layout.pop('bitmap')
        feature_levels = [level for level, dim in enumerate(plane_dims) for _ in range(dim)]
        self.register_buffer('feature_levels', torch.tensor(feature_levels, dtype=torch.int64, device=device), persistent=False)
        self.register_buffer('level_active', torch.ones(self.n_levels, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer('feature_active', torch.ones(self.input_width, dtype=torch.float16, device=device), persistent=False)
        self.active_prefix = self.n_levels
        decoder = {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None',
                   'n_neurons': 64, 'n_hidden_layers': 2}
        gate = dict(decoder, n_neurons=32, n_hidden_layers=1)
        # TCNN's native initialization uses the default CUDA stream.
        initialization_stream = torch.cuda.default_stream(device)
        with torch.cuda.stream(initialization_stream):
            self.gate = (tcnn.Network(self.input_width, self.n_levels, gate, seed=(seed + 2000003) % (2**31))
                         if self.router_enabled else None)
            self.decoder = tcnn.Network(self.input_width, 3, decoder, seed=(seed + 4000037) % (2**31))
        torch.cuda.current_stream(device).wait_stream(initialization_stream)
        if ((self.gate.params.numel() if self.gate is not None else 0) != config['gate_parameters'] or
                self.decoder.params.numel() != config['decoder_parameters']):
            raise RuntimeError('Unexpected TCNN head allocation for the configured input width')
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise RuntimeError('Trainable parameter masters must be FP32')

    def query_features(self, coordinates):
        if coordinates.dtype != torch.float32 or coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.requires_grad:
            raise ValueError('Coordinates must be FP32 Nx2, without coordinate gradients')
        return PackedQuery.apply(coordinates.contiguous(), self.grid, self.descriptors, self.node_bitmap,
                                 self.node_prefix, self.bitmap, self.input_width)

    @torch.no_grad()
    def _initialize_plane(self, level):
        if self.initialized_planes[level]:
            raise RuntimeError('Each plane must be initialized exactly once')
        item = self.layout['levels'][level]
        # Keep baseline F2 draws exactly; F4 uses four draws per native node. Unopened
        # planes make no feature RNG draws until their support is selected.
        level_seed = (self.config['seed'] + 1000003 * (level + 1)) % (2**31)
        branch_seed = (level_seed + 104729 * 3) % (2**31)
        generator = torch.Generator(device=self.grid.device).manual_seed(branch_seed)
        original = torch.empty((item['native_plane_points'], max(2, item['plane_dim'])), dtype=torch.float32,
                               device=self.grid.device)
        original.normal_(0., self.config['feature_init_std'], generator=generator)
        values = original.index_select(0, item['nodes'].to(self.grid.device))[:, :item['plane_dim']].contiguous().flatten()
        offset = item['plane_offset']
        self.grid[offset:offset + values.numel()].copy_(values)
        self.initialized_planes[level] = True

    @torch.no_grad()
    def activate_support(self, level, support):
        """Add only the next unopened plane, retaining packed storage/offsets."""
        if level != self.active_prefix or not 2 <= level < self.n_levels:
            raise ValueError('Select the next unopened plane before advancing the prefix')
        if self.initialized_planes[level] or bool(self.supports[level].any()):
            raise RuntimeError('An opened support cannot be replaced or initialized again')
        resolution = self.config['resolutions'][level]
        if support.device.type != 'cpu' or support.dtype != torch.bool or support.shape != (resolution**2,):
            raise ValueError('New support must be a flat CPU bool R^2 tensor')
        nodes = plane_closure(support, resolution)
        cap = self.plane_node_caps[level]
        if not 0 < nodes.numel() <= cap:
            raise ValueError('New support requires a nonempty complete closure within its cap')
        storage_pointer = self.grid.data_ptr()
        updated = list(self.supports)
        updated[level] = support.clone()
        # Rebuild nonparameter addressing so a newly full identity closure needs
        # no mapping allocation. Feature capacities/offsets remain unchanged.
        layout = make_layout(self.config, updated, self.plane_dims, self.plane_node_caps)
        for name in ('descriptors', 'node_bitmap', 'node_prefix', 'bitmap'):
            setattr(self, name, layout[name].to(self.grid.device))
        for name in ('node_bitmap', 'node_prefix', 'bitmap'):
            layout.pop(name)
        self.layout, self.supports = layout, layout['supports']
        points = layout['levels'][level]['native_plane_points']
        self._initialize_plane(level)
        if self.grid.data_ptr() != storage_pointer:
            raise AssertionError('Packed storage changed during plane activation')
        return {'level': level, 'initialized_once': True, 'grid_storage_unchanged': True,
                'original_dense_f2_seed': (self.config['seed'] + 1000003 * (level + 1) + 104729 * 3) % (2**31),
                'native_dense_draw_shape': [points, max(2, self.plane_dims[level])], 'selected_nodes': nodes.numel()}

    def set_active_prefix(self, prefix):
        if not isinstance(prefix, int) or not 2 <= prefix <= self.n_levels:
            raise ValueError('Active prefix must be an integer between 2 and declared level count')
        self.active_prefix = prefix
        self.level_active.copy_(torch.arange(self.n_levels, device=self.grid.device) < prefix)
        self.feature_active.copy_((self.feature_levels < prefix).to(torch.float16))

    def forward(self, coordinates):
        raw, visible = self.query_features(coordinates)
        raw = raw * self.feature_active
        if self.gate is None:
            return self.decoder(raw).float()[:, :3]
        if self.active_prefix > self.gate_coarse_levels_K:
            own = router_weights(self.gate(raw).float(), self.config.get('router_output_activation', 'sigmoid'))
            count = self.gate_coarse_levels_K
            gates = torch.cat([torch.ones_like(own[:, :count]), own[:, count:]], dim=1)
        else:
            gates = torch.ones_like(visible, dtype=torch.float32)
        effective = gates * (visible & self.level_active).float()
        weights = effective.to(raw.dtype).index_select(1, self.feature_levels)
        return self.decoder(raw * weights).float()[:, :3]

    def storage_stats(self):
        grid = self.grid.numel()
        gate_params = self.gate.params.numel() if self.gate is not None else 0
        heads = gate_params + self.decoder.params.numel()
        node_bitmaps = self.node_bitmap.numel() * self.node_bitmap.element_size()
        prefixes = self.node_prefix.numel() * self.node_prefix.element_size()
        indices = node_bitmaps + prefixes
        bitmaps = self.bitmap.numel() * self.bitmap.element_size()
        descriptors = self.descriptors.numel() * self.descriptors.element_size()
        gate_index = self.feature_levels.numel() * self.feature_levels.element_size()
        active_flags = (self.level_active.numel() * self.level_active.element_size() +
                        self.feature_active.numel() * self.feature_active.element_size())
        return {'actual_params': grid + heads, 'grid_params': grid, 'head_params': heads,
                'gate_params': gate_params, 'decoder_params': self.decoder.params.numel(),
                'router_enabled': self.router_enabled,
                'axis_params': 0, 'plane_params': grid, 'plane_caps': self.plane_node_caps,
                'fp16_parameter_payload_bytes': 2 * (grid + heads),
                'fp32_master_bytes': 4 * (grid + heads), 'adam_moment_bytes': 8 * (grid + heads),
                'full_gradient_bytes': 4 * (grid + heads), 'index_bytes': indices,
                'node_bitmap_bytes': node_bitmaps, 'node_prefix_bytes': prefixes,
                'bitmap_bytes': bitmaps, 'descriptor_bytes': descriptors,
                'runtime_index_bytes': indices + bitmaps + descriptors + gate_index + active_flags,
                'active_flag_bytes': active_flags,
                'input_width': self.input_width, 'native_input_padding': (self.input_width + 15) // 16 * 16, 'axis_mode': 'none',
                'active_layers_from_update1': int(self.config.get('initial_active_prefix', 2)), 'active_prefix': self.active_prefix,
                'gate_coarse_levels_K': int(self.config['coarse_levels']),
                'full_cover_coarse_layers': self.config.get('full_cover_coarse_layers', [0, 1]),
                'plane_dims': self.plane_dims,
                'coordinate_version': COORDINATE_VERSION, 'layout_version': LAYOUT_VERSION,
                'addressing_version': ADDRESSING_VERSION,
                'query_cuda_launches': 1, 'backward_cuda_launches': 1,
                'forward_precision': 'P01 FP32 geometry and FP16 interpolation chain',
                'gradient_precision': 'FP16 feature gradient, FP32 interpolation weights and atomicAdd',
                'levels': [{key: value for key, value in level.items() if key != 'nodes'} for level in self.layout['levels']]}


def build_model(config, supports, plane_dims, plane_node_caps, **kwargs):
    if config['representation'] == 'hash_grid':
        from .hash_grid import HashGrid
        return HashGrid(config, supports, plane_dims, plane_node_caps)
    return PackedXYPlanes(config, supports, plane_dims, plane_node_caps, **kwargs)
