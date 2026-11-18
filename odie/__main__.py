"""python -m odie run|batch|evaluate|compress --help"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .config import load_config, resolve, source_hash
from scripts._common.io import update_run, run_log


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def run(args):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    path = args.input.resolve()
    config = load_config(args.config)
    if 'png_fraction' in config['budget'] and path.suffix.lower() != '.png':
        raise ValueError('png_fraction requires an original PNG; use an explicit budget for JPEG input')
    with Image.open(path) as image:
        shape = tuple(reversed(image.size))
    if shape[0]*shape[1] > 2**32-1:
        raise ValueError('Native pixel addressing requires fewer than 2^32 pixels')
    resolved = resolve(config, shape, path.stat().st_size)
    if args.dry_run:
        print(json.dumps(dict(input=str(path), shape_hw=shape, resolved=resolved), indent=2))
        return
    root = args.output.resolve()
    managed = (args.config.resolve() == root/'config.json' and (root/'run.json').is_file()
               and json.loads((root/'run.json').read_text()).get('method') == 'odie')
    if root.exists() and any(root.iterdir()) and not managed:
        raise FileExistsError(f'Output must be new or empty: {root}')
    if managed and any(p.name not in {'config.json','run.json','capacity.json','train.log'} for p in root.iterdir()):
        raise FileExistsError(f'Unexpected existing training outputs: {root}')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'config.json', config)
    with run_log(root):
        train(args, path, root, resolved, shape, managed)


def train(args, path, root, resolved, shape, managed):
    update_run(root,state='preparing' if managed else 'training')
    try:
        from . import runtime_env
        import torch
        from .train import fit
        from .model import make_layout
        from .optimizer import validate_schedule
        validate_schedule(resolved)
        resolved.update(input_sha256=source_hash(path), shape_hw=list(shape))
        update_run(root,configuration=resolved)
        supports = ([] if resolved['representation']=='hash_grid' else
                    [torch.full((r*r,), i < resolved['initial_active_prefix'], dtype=torch.bool)
                     for i, r in enumerate(resolved['resolutions'])])
        record = dict(supports=supports, plane_dims=resolved['plane_feature_dimensions'],
                      plane_node_caps=resolved['cell_caps'], metadata=dict(
                          input_sha256=resolved['input_sha256'], input_path=str(path),
                          extra_teacher_use=False, historical_mask_use=False,
                          initial_support=resolved['support_policy'] if resolved['representation']=='hash_grid' else 'full dense prefix, empty sparse levels',
                          selection_policy=resolved['dynamic_policy']))
        if resolved['representation'] != 'hash_grid':
            layout = make_layout(resolved, supports, record['plane_dims'], record['plane_node_caps'])
            if layout['params']+resolved['gate_parameters']+resolved['decoder_parameters'] != resolved['allocated_parameters']:
                raise AssertionError('Prepared layout and budget allocation disagree')
        source = Path(__file__).resolve().parent
        files = [p for p in source.rglob('*') if p.is_file() and p.suffix in ('.py', '.cu', '.cpp', '.cuh', '.txt')]
        update_run(root,runtime=dict(input_sha256=resolved['input_sha256'],
            source_sha256={str(p.relative_to(source)):source_hash(p) for p in files},
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda))
        fit(SimpleNamespace(input=path, configuration=resolved, support_record=record,
             output=root, run_id=root.name, axis_mode='none', kernel_a=0., kernel_omega=1.5*3.141592653589793,
             loss_scale=128., adam_eps=1e-15, parameter_cap=resolved['allocated_parameters'],
             fp16_parameter_budget=resolved['fp16_parameter_budget']))
        if not managed:
            update_run(root,state='completed')
    except BaseException as error:
        update_run(root,state='failed', error=f'{type(error).__name__}: {error}')
        raise


def batch(args):
    manifest = args.manifest.resolve()
    rows = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty list of input/config/output records')
    jobs = []
    for row in rows:
        if set(row) != {'input', 'config', 'output'}:
            raise ValueError('Each manifest row must contain exactly input, config and output')
        job = {k: str((manifest.parent/v).resolve()) for k, v in row.items()}
        if any(job['output'] == previous['output'] for previous in jobs):
            raise ValueError('Manifest contains duplicate output directories')
        jobs.append(job)
    # Validate every job before training the first one.
    for job in jobs:
        subprocess.run([sys.executable, '-m', 'odie', 'run', '--dry-run',
                        *[part for k, v in job.items() for part in ('--'+k, v)]], check=True)
    if args.dry_run:
        return
    for job in jobs:
        subprocess.run([sys.executable, '-u', '-m', 'odie', 'run',
                        *[part for k, v in job.items() for part in ('--'+k, v)]],check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('run', help='Train one image from a configuration')
    train.add_argument('--input', type=Path, required=True)
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--dry-run', action='store_true')
    queue = commands.add_parser('batch', help='Run a JSON experiment manifest sequentially, stop on failure')
    queue.add_argument('--manifest', type=Path, required=True)
    queue.add_argument('--dry-run', action='store_true')
    metric = commands.add_parser('evaluate', help='Evaluate a reconstruction PNG or raw NPY')
    metric.add_argument('--input', type=Path, required=True)
    metric.add_argument('--prediction', type=Path, required=True)
    metric.add_argument('--output', type=Path, required=True)
    metric.add_argument('--stripe-rows', type=int, default=256)
    compact = commands.add_parser('compress', help='Losslessly compress an existing checkpoint support mask on CPU')
    compact.add_argument('--checkpoint', type=Path, required=True)
    compact.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    elif args.command == 'batch':
        batch(args)
    elif args.command == 'compress':
        from .support_codec import compress_checkpoint
        print(json.dumps(compress_checkpoint(args.checkpoint, args.output), indent=2))
    else:
        from .metrics import image_metrics
        if args.stripe_rows <= 0:
            parser.error('--stripe-rows must be positive')
        result = image_metrics(args.input, args.prediction, args.stripe_rows)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
