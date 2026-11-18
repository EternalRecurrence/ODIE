"""Common full-image RGB evaluation; FP32 differences, FP64 sum."""
import math
import time
import torch

def coordinates(shape, ids):
    height, width = shape
    return torch.stack((ids.remainder(width).float() / (width - 1),
                        torch.div(ids, width, rounding_mode='floor').float() / (height - 1)), dim=1)


@torch.no_grad()
def evaluate(model, target, shape, batch_size, return_image=False):
    evaluation_started = time.perf_counter()
    copy_wait_seconds = 0.
    was_training = model.training
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=target.device)
    image = torch.empty((target.shape[0], 3), dtype=torch.float32) if return_image else None
    for begin in range(0, target.shape[0], batch_size):
        ids = torch.arange(begin, min(begin + batch_size, target.shape[0]), device=target.device)
        prediction = model(coordinates(shape, ids)).float()
        total += (prediction - target[ids]).square().double().sum()
        if image is not None:
            copy_started = time.perf_counter()
            image[begin:begin + ids.numel()] = prediction.cpu()
            copy_wait_seconds += time.perf_counter() - copy_started
    mse = float(total.item() / target.numel())
    if not math.isfinite(mse):
        raise FloatingPointError('Full-image RGB evaluation is nonfinite')
    model.train(was_training)
    evaluate.last_timing = {'total_evaluation_seconds': time.perf_counter() - evaluation_started,
                           'prediction_cpu_copy_wait_seconds': copy_wait_seconds if return_image else None,
                           'prediction_cpu_transfer_seconds': None,
                           'copy_timing_basis': 'CPU copy wall time includes waiting for queued CUDA work; no extra synchronization'}
    return {'mse': mse, 'psnr_db': -10 * math.log10(mse) if mse else None}, image

