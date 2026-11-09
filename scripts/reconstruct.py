"""Reconstruct RGB from an ODIE inference checkpoint, separately from evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='RGB PNG or HWC float32 NPY')
    parser.add_argument('--batch-size', type=int, default=131072)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.suffix.lower() not in ('.png', '.npy'):
        parser.error('Use a positive batch size and a .png or .npy output')
    if args.output.exists():
        raise FileExistsError(args.output)
    from odie import runtime_env
    import numpy as np
    from PIL import Image
    import torch
    from odie.evaluation import coordinates
    from odie.model import build_model
    from odie.support_codec import restore_support_record
    from odie.train import copy_parameters

    artifact = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = artifact['configuration']
    record = restore_support_record(artifact)
    model = build_model(config, record['supports'], record['plane_dims'], record['plane_node_caps']).cuda().eval()
    model.set_active_prefix(config['n_levels'])
    copy_parameters(model, artifact['parameters'])
    shape = config['shape_hw']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = (np.lib.format.open_memmap(args.output, mode='w+', dtype=np.float32, shape=(*shape, 3))
              if args.output.suffix.lower() == '.npy' else np.empty((*shape, 3), dtype=np.uint8))
    pixels = values.reshape(-1, 3)
    with torch.no_grad():
        for begin in range(0, len(pixels), args.batch_size):
            ids = torch.arange(begin, min(begin+args.batch_size, len(pixels)), device='cuda')
            prediction = model(coordinates(shape, ids)).float().cpu().numpy()
            pixels[begin:begin+len(ids)] = (prediction if values.dtype == np.float32 else
                np.rint(np.clip(prediction, 0., 1.)*255).astype(np.uint8))
    if isinstance(values, np.memmap):
        values.flush()
    else:
        Image.fromarray(values).save(args.output)


if __name__ == '__main__':
    main()
