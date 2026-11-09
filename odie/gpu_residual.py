"""Stream live RGB residuals to exact CUDA greedy without full-image downloads.

FP32 differences/squares,
FP64 raster-order sums and the existing greedy kernel retain the CPU rule.
The score array stays on GPU; only the selected support/node masks come back.
"""
import ctypes as C
from functools import lru_cache
from hashlib import sha256
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import selection as ref
from .evaluation import coordinates
from .selection_cuda import Cuda, build as build_greedy


@lru_cache(maxsize=1)
def libraries():
    """Compile before timed validation; cache by source hash and device arch."""
    import fcntl
    source = Path(__file__).with_name('residual_cuda.cu')
    digest = sha256(source.read_bytes()).hexdigest()
    major, minor = torch.cuda.get_device_capability()
    cache = Path(os.environ.get('ODIE_BUILD_CACHE', str(Path.home()/'.cache/odie')))
    cache = cache / f'residual-{digest[:16]}-sm{major}{minor}'
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/'libodie_residual.so'
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            command = [str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'), '-shared', '-O3',
                       '--fmad=false', '--ftz=false', f'-arch=sm_{major}{minor}',
                       '-Xcompiler', '-fPIC', '-ccbin', os.environ['CXX'], str(source), '-o', str(library)]
            subprocess.run(command, check=True)
    cuda = Cuda(build_greedy())
    residual = C.CDLL(str(library))
    residual.score_chunk.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int64,
        C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_bool, C.c_uint64]
    return cuda, residual


class DeviceScores:
    def __init__(self, shape, resolution):
        self.shape, self.resolution = shape, resolution
        self.cuda, self.library = libraries()
        height, width = shape
        thresholds = np.asarray(ref.native_thresholds(resolution), dtype=np.float32)
        xmap = np.searchsorted(thresholds, np.arange(width, dtype=np.float32)/np.float32(width-1), side='right')
        self.ymap = np.searchsorted(thresholds, np.arange(height, dtype=np.float32)/np.float32(height-1), side='right')
        bounds = np.arange(resolution+1)
        xb = np.searchsorted(xmap, bounds).astype(np.int32)
        yb = np.searchsorted(self.ymap, bounds).astype(np.int32)
        self.counts = (np.diff(yb).astype(np.int64)[:,None]*np.diff(xb)[None,:]).reshape(-1)
        self.xb, self.yb = torch.from_numpy(xb).cuda(), torch.from_numpy(yb).cuda()
        self.totals = torch.zeros(resolution*resolution, dtype=torch.float64, device='cuda')

    def add(self, prediction, target, begin):
        prediction, target = prediction.contiguous(), target.contiguous()
        end, width = begin+prediction.shape[0], self.shape[1]
        first_y = int(self.ymap[begin//width])
        active = (int(self.ymap[(end-1)//width])-first_y+1)*self.resolution
        ptr = lambda tensor: C.c_void_p(tensor.data_ptr())
        self.cuda.check(self.library.score_chunk(ptr(prediction), ptr(target), begin, end,
            width, self.resolution, first_y, active, ptr(self.xb), ptr(self.yb), ptr(self.totals), target.dtype == torch.uint8,
            torch.cuda.current_stream().cuda_stream))


def select_device(cuda, ds, n, resolution, cap):
    """Select device-resident residual scores; include cleanup in timing."""
    points = ref.align8(n)
    sizes, offsets, tree_size = [], [], 0
    children = n
    while True:
        parents = (children+31)//32
        sizes.append(parents)
        offsets.append(tree_size)
        tree_size += parents*5
        if parents == 1:
            break
        children = parents
    sizes, offsets = np.array(sizes, dtype=np.int32), np.array(offsets, dtype=np.int32)
    allocations = []
    def allocate(size, host=None):
        pointer = C.c_void_p()
        cuda.check(cuda.rt.cudaMalloc(C.byref(pointer), size))
        allocations.append(pointer)
        if host is not None:
            cuda.check(cuda.rt.cudaMemcpy(pointer, host.ctypes.data, size, 1))
        return pointer
    started = time.perf_counter()
    try:
        dm, dc, dn = allocate(n*4), allocate(n), allocate(points)
        dt = allocate(tree_size*4)
        do, dz = allocate(offsets.nbytes, offsets), allocate(sizes.nbytes, sizes)
        st = allocate(4*8)
        status = np.zeros(4, dtype=np.int64)
        cuda.check(cuda.lib.selection_init(ds, n, points, dm, dc, dn, dt, st,
            offsets.ctypes.data, sizes.ctypes.data, len(sizes), 0))
        launches = 0
        while not status[2]:
            cuda.check(cuda.lib.selection_chunk(ds, n, points, resolution, cap, True,
                dm, dc, dn, dt, do, dz, len(sizes), 512, st, None, 0))
            cuda.check(cuda.rt.cudaMemcpy(status.ctypes.data, st, status.nbytes, 2))
            launches += 1
            if status[3]:
                raise AssertionError('CUDA marginal-cost invariant failed')
        selected, nodes = np.empty(n, dtype=np.uint8), np.empty(points, dtype=np.uint8)
        cuda.check(cuda.rt.cudaMemcpy(selected.ctypes.data, dc, selected.nbytes, 2))
        cuda.check(cuda.rt.cudaMemcpy(nodes.ctypes.data, dn, nodes.nbytes, 2))
        support, record = ref._result(selected.view(np.bool_), nodes.view(np.bool_),
            resolution, cap, 'B2 GPU residual / exact newly added closure-node parameters', int(status[1]))
        if record['closure_nodes'] != int(status[0]):
            raise AssertionError('Device and host closure counts differ')
    finally:
        for pointer in reversed(allocations):
            cuda.check(cuda.rt.cudaFree(pointer))
    record.update(cuda_end_to_end_seconds=time.perf_counter()-started, launches=launches,
                  score_gpu_to_cpu_copies=0, score_cpu_to_gpu_copies=0,
                  selection_backend='cuda_device_residual_exact_greedy')
    return support, record


@torch.no_grad()
def select_live(model, target, shape, resolution, cap, dim, batch_size, validate_scores=False):
    """Time setup + inference/scoring + greedy + cleanup; exclude compilation.

    validate_scores adds a score download after timing solely for the paired
    validation. Normal training neither downloads the scores nor a full image.
    """
    libraries()
    torch.cuda.synchronize()
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        scores = DeviceScores(shape, resolution)
        total = torch.zeros((), dtype=torch.float64, device='cuda')
        for begin in range(0, target.shape[0], batch_size):
            ids = torch.arange(begin, min(begin+batch_size, target.shape[0]), device=target.device)
            prediction = model(coordinates(shape, ids)).float()
            truth = target[ids]
            total += (prediction-truth).square().double().sum()
            # ByteTarget.cpu() divides with correct FP32 rounding; CUDA Tensor
            # scalar division uses a reciprocal and differs by 1 ULP for 126/256
            # byte values. The score kernel uses __fdiv_rn for RGB8 to preserve
            # the CPU residual reference exactly.
            score_target = target.data[begin:begin+ids.numel()] if target.data.dtype == torch.uint8 else truth
            scores.add(prediction, score_target, begin)
        torch.cuda.synchronize()
        scoring_seconds = time.perf_counter()-started
        mse = float(total.item()/target.numel())
        if not math.isfinite(mse) or not bool(torch.isfinite(scores.totals).all()):
            raise FloatingPointError('GPU live RGB residual is nonfinite')
        # The greedy selector uses stream 0; scoring must finish first.
        support, record = select_device(scores.cuda, C.c_void_p(scores.totals.data_ptr()),
                                         resolution*resolution, resolution, cap)
        record.update(plane_dim=dim, normalize_by_cost=True)
        counts = scores.counts
        device_totals = scores.totals if validate_scores else None
        del scores
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        downloaded = device_totals.cpu().numpy() if validate_scores else None
        timing = dict(pipeline_seconds=seconds, inference_and_scoring_seconds=scoring_seconds,
                      greedy_seconds=record['cuda_end_to_end_seconds'],
                      prediction_gpu_to_cpu_copies=0, gt_gpu_to_cpu_copies=0,
                      validation_only_score_download=bool(validate_scores),
                      note='Setup, inference, scoring, selection, host support/closure validation and cleanup; compile excluded')
        return ({'mse':mse,'psnr_db':-10*math.log10(mse) if mse else None},
                support, record, counts, timing, downloaded)
    finally:
        model.train(was_training)
