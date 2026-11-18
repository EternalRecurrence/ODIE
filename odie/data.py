"""Original RGB targets and pixel sampling for image fitting."""
import numpy as np
from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None


class ByteTarget:
    def __init__(self, data):
        if data.dtype != torch.uint8 or data.ndim != 2 or data.shape[1] != 3:
            raise ValueError('Expected RGB8 Nx3 target')
        self.data, self.shape, self.device = data, data.shape, data.device

    def __getitem__(self, ids):
        return self.data[ids].float() / 255.

    def data_ptr(self):
        return self.data.data_ptr()

    def numel(self):
        return self.data.numel()

    def detach(self):
        return self

    def cpu(self):
        return self.data.cpu().float().div_(255.)


def load_target(path, storage, device='cuda'):
    with Image.open(path) as image:
        rgb = np.array(image.convert('RGB'), dtype=np.uint8)
    shape = rgb.shape[:2]
    if min(shape) < 2:
        raise ValueError('Endpoint coordinates require both image axes >= 2')
    if storage == 'uint8':
        target = ByteTarget(torch.from_numpy(rgb.reshape(-1, 3)).to(device))
    elif storage == 'float32':
        values = rgb.astype(np.float32) / 255.
        target = torch.from_numpy(values.reshape(-1, 3)).to(device)
    else:
        raise ValueError('target_storage must be float32 or uint8')
    return target, shape


def full_batches(pixels, batch, generator, device='cuda'):
    """Visit a global permutation once, then pad with its beginning.

    Keep the permutation on the generator's device (CPU for giant images).
    Only the current batch is converted to int64 and transferred to the GPU.
    """
    dtype = torch.int32 if pixels <= 2**31-1 else torch.int64
    order = torch.randperm(pixels, dtype=dtype, device=generator.device, generator=generator)
    for begin in range(0, pixels, batch):
        ids = order[begin:begin+batch]
        if ids.numel() < batch:
            padding = torch.arange(batch-ids.numel(), device=order.device) % pixels
            ids = torch.cat((ids, order[padding]))
        yield ids.to(device=device, dtype=torch.int64)


def stratified_batches(shape, batch, generator, device='cuda'):
    """One random pixel per square stratum per step, as in I-NGP image fitting.

    Epochs count ceil(H*W/batch) fresh draws, not a traversal of all pixels.
    Pixel IDs use the model's existing coordinate convention downstream.
    """
    height, width = shape
    side = int(batch ** .5)
    if batch < 1 or batch & (batch - 1) or side * side != batch:
        raise ValueError('Stratified sampling requires a square power-of-two batch size')
    slots = torch.arange(batch, device=device)
    strata = torch.stack((slots % side, slots // side), dim=1).float() / side
    scale = torch.tensor((width, height), dtype=torch.float32, device=device)
    for _ in range((height * width + batch - 1) // batch):
        # Match stratify2_kernel: random/side + stratum/side, then pixel lookup.
        uv = torch.rand((batch, 2), device=device, generator=generator) / side + strata
        xy = (uv * scale).long()
        yield xy[:, 1].clamp_(0, height - 1) * width + xy[:, 0].clamp_(0, width - 1)


def shuffled_batches(pixels, batch, generator, mode='global', chunk=4194304, device='cuda', shape=None):
    if mode == 'global_full':
        yield from full_batches(pixels, batch, generator, device)
    elif mode == 'stratified':
        if shape is None or len(shape) != 2 or shape[0] * shape[1] != pixels or min(shape) < 2:
            raise ValueError('Stratified sampling requires the matching image shape')
        yield from stratified_batches(shape, batch, generator, device)
    elif mode == 'global':
        permutation = torch.randperm(pixels, device=device, generator=generator)
        for begin in range(0, pixels, batch):
            yield permutation[begin:begin + batch]
    elif mode == 'chunked':
        if chunk <= 0 or chunk % batch:
            raise ValueError('Chunk size must be a positive multiple of batch size')
        count = (pixels + chunk - 1) // chunk
        order = torch.randperm(count, device=device, generator=generator).cpu().tolist()
        for block in order:
            start = block * chunk
            size = min(chunk, pixels - start)
            permutation = torch.randperm(size, device=device, generator=generator)
            for begin in range(0, size, batch):
                yield permutation[begin:begin + batch] + start
    else:
        raise ValueError('sampling must be global, chunked, global_full or stratified')
