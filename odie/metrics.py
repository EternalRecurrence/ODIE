"""Standalone raw-float / RGB8 PSNR and SSIM, with bounded SSIM work buffers."""
import math
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def array_metrics(truth, prediction, stripe_rows=256, rgb8=False):
    from skimage.metrics import structural_similarity
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[2] != 3:
        raise ValueError('Expected matching HxWx3 images')
    height, width, _ = truth.shape
    window = min(7, min(height, width) if min(height, width) % 2 else min(height, width)-1)
    radius = window // 2
    error = 0.
    ssim_sums = np.zeros(3, dtype=np.float64)
    count = 0

    def normalized(array):
        if rgb8 and np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise FloatingPointError('Nonfinite evaluation input')
            # Match reconstruction.png export without a full-image RGB8 copy.
            array = np.rint(np.clip(array, 0., 1.) * 255).astype(np.uint8)
        return array.astype(np.float32) / 255. if array.dtype == np.uint8 else array.astype(np.float32, copy=False)

    for start in range(0, height, stripe_rows):
        a, b = normalized(truth[start:start+stripe_rows]), normalized(prediction[start:start+stripe_rows])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError('Nonfinite evaluation input')
        error += np.square(a.astype(np.float64)-b.astype(np.float64)).sum(dtype=np.float64)
    if window >= 3:
        for start in range(radius, height-radius, stripe_rows):
            end = min(start+stripe_rows, height-radius)
            for channel in range(3):
                a = normalized(truth[start-radius:end+radius, :, channel])
                b = normalized(prediction[start-radius:end+radius, :, channel])
                _, values = structural_similarity(a, b, data_range=1., win_size=window,
                    gaussian_weights=False, use_sample_covariance=True, full=True)
                valid = values[radius:-radius, radius:-radius]
                ssim_sums[channel] += valid.sum(dtype=np.float64)
            count += (end-start)*(width-2*radius)
    mse = float(error / truth.size)
    return dict(mse=mse, psnr_db=-10*math.log10(mse) if mse else None,
                perfect_reconstruction=mse == 0, ssim=float((ssim_sums/count).mean()) if count else None,
                ssim_window=window if count else None)


def image_metrics(input_path, prediction_path, stripe_rows=256, prediction_array=None):
    with Image.open(input_path) as image:
        truth = np.array(image.convert('RGB'), dtype=np.uint8)
    if prediction_array is not None:
        prediction = prediction_array
    elif Path(prediction_path).suffix.lower() == '.npy':
        prediction = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
    else:
        with Image.open(prediction_path) as image:
            prediction = np.array(image.convert('RGB'), dtype=np.uint8)
    result = array_metrics(truth, prediction, stripe_rows)
    result.update(shape_hwc=list(truth.shape), prediction_format='rgb8' if prediction.dtype == np.uint8 else 'raw_float32',
                  input=str(input_path), prediction=str(prediction_path),
                  definition='data_range=1, RGB channel mean, uniform window, sample covariance')
    return result
