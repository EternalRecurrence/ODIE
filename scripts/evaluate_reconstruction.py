"""Shared full-resolution RGB8 PSNR/SSIM and compression ratios."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from odie.metrics import array_metrics
from scripts._common.io import sha, write

Image.MAX_IMAGE_PIXELS = None


def compression(input_path, shape, files, odie_metrics=None):
    """The caller lists every file required to decode one selected checkpoint."""
    seen = set()
    records = []
    forbidden = Path(input_path).resolve()
    for name in files:
        path = Path(name).resolve(strict=True)
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if not path.is_file() or stat.st_size <= 0:
            raise ValueError(f'Expected nonempty inference file: {path}')
        if path == forbidden or identity in seen:
            raise ValueError(f'Duplicate inference file or input image: {path}')
        seen.add(identity)
        records.append(dict(path=str(path), bytes=stat.st_size, sha256=sha(path)))
    if not records:
        raise ValueError('List all inference files, including required support/index/config sidecars')
    total = sum(row['bytes'] for row in records)
    source = Path(input_path).stat().st_size
    raw = int(shape[0])*int(shape[1])*3
    result = dict(inference_files=records, complete_file_bytes=total,
                source_file_bytes=source, raw_rgb8_bytes=raw,
                cr_source_file=source/total, cr_raw_rgb8=raw/total,
                storage_fraction_source=total/source, storage_fraction_raw_rgb8=total/raw,
                definition='CR = original bytes / complete inference bytes; larger is better',
                accounting='One checkpoint plus all decode-required sidecars; exclude logs, reconstructions, caches and optimizer state')
    if odie_metrics is not None:
        parameters = odie_metrics['fp16_parameter_payload_bytes']
        activation = odie_metrics['support_storage']['payload_bytes']
        # Saved FP16 weights and compressed support streams; excludes metadata/container overhead.
        for name, size in (('parameters_only', parameters), ('parameters_and_activation', parameters+activation)):
            result[name] = dict(payload_bytes=size, cr_source_file=source/size, cr_raw_rgb8=raw/size)
    return result


def evaluate(input_path, prediction_path, files, checkpoint, layout='HWC', stripe_rows=256,
             quality_only=False, odie_metrics=None):
    if stripe_rows <= 0:
        raise ValueError('stripe_rows must be positive')
    with Image.open(input_path) as image:
        truth = np.array(image.convert('RGB'), dtype=np.uint8)
    prediction_path = Path(prediction_path)
    if prediction_path.suffix.lower() == '.npy':
        prediction = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
        if layout == 'CHW':
            prediction = prediction.transpose(1, 2, 0)
        if prediction.dtype != np.uint8 and not np.issubdtype(prediction.dtype, np.floating):
            raise ValueError('NPY must contain uint8 RGB or floating RGB values in the 0..1 convention')
    else:
        if layout != 'HWC':
            raise ValueError('CHW layout applies only to NPY arrays')
        with Image.open(prediction_path) as image:
            if image.mode != 'RGB':
                raise ValueError('Prediction image must be RGB; no implicit alpha/mode conversion')
            prediction = np.array(image, dtype=np.uint8)
    if Path(input_path).resolve() == prediction_path.resolve():
        raise ValueError('Prediction must be a separate reconstruction')
    if any(Path(p).resolve() == prediction_path.resolve() for p in files):
        raise ValueError('Reconstruction cannot be counted as an inference model')
    quality = array_metrics(truth, prediction, stripe_rows, rgb8=True)
    result = dict(schema='shared-image-evaluation-v1', checkpoint=checkpoint,
                  input=str(Path(input_path).resolve()), prediction=str(prediction_path.resolve()),
                  input_sha256=sha(Path(input_path)), prediction_sha256=sha(prediction_path),
                  shape_hwc=list(truth.shape), prediction_layout=layout,
                  prediction_format='rgb8' if prediction.dtype == np.uint8 else str(prediction.dtype),
                  psnr_pixel_format='rgb8',
                  ssim_pixel_format='rgb8',
                  preprocessing='RGB8: clip floating RGB to [0,1], rint(x*255) to uint8; original resolution; no resize, filtering or antialiasing',
                  ssim_definition='RGB channel mean; uniform window min(7, largest valid odd size); sample covariance; data_range=1',
                  mse=quality['mse'], psnr_db=quality['psnr_db'],
                  perfect_reconstruction=quality['perfect_reconstruction'],
                  ssim=quality['ssim'], ssim_window=quality['ssim_window'])
    if not quality_only:
        if not checkpoint:
            raise ValueError('Name the checkpoint used to produce this reconstruction')
        result['compression'] = compression(input_path, truth.shape, files, odie_metrics)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--model-files', type=Path, nargs='+', default=[])
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout', choices=['HWC','CHW'], default='HWC')
    parser.add_argument('--stripe-rows', type=int, default=256)
    parser.add_argument('--quality-only', action='store_true', help='PSNR/SSIM only; no compression claim without a file list')
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.prediction.resolve(), *(p.resolve() for p in args.model_files)}:
        parser.error('Output must not overwrite evaluation inputs')
    result = evaluate(args.input, args.prediction, args.model_files, args.checkpoint,
                      args.layout, args.stripe_rows, args.quality_only)
    write(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
