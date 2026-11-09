"""Config-driven method entries; only ODIE reads a corresponding INGP count."""
import argparse
from copy import deepcopy
import math
import os
import signal
from pathlib import Path
import sys

from . import capacity
from .defaults import official
from .io import HERE, ROOT, now, read, sha, write, update_run, run_log
from .vendor import verify


def reference(path, digest, shape, group):
    path = Path(path)
    record = read(path/'capacity.json' if path.is_dir() else path)
    if record.get('method')!='ingp' or record.get('capacity_verified') is not True:
        raise ValueError('ODIE requires an actual native-verified INGP capacity.json, not a CPU estimate')
    if record.get('input_sha256')!=digest or record.get('shape_hw')!=list(shape):
        raise ValueError('INGP reference belongs to a different input image')
    if record.get('experiment_group')!=group:
        raise ValueError('INGP reference belongs to a different experiment group')
    if type(record.get('parameter_count')) is not int or record['parameter_count']<=0:
        raise ValueError('Invalid INGP parameter count')
    return record


def resolve(method, scene, args):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    requested = read(args.config)
    allowed={'experiment_group','capacity','training','evaluation','model','data'}
    if set(requested)-allowed:
        raise ValueError(f'Unknown config sections: {set(requested)-allowed}')
    for name in ('experiment_group','capacity','training','evaluation'):
        if name not in requested:
            raise ValueError(f'Missing explicit experiment setting: {name}')
    inp=args.input.resolve()
    with Image.open(inp) as image:
        shape=(image.height,image.width)
        if min(shape)<2:
            raise ValueError('Each image dimension must be at least two pixels')
    digest=sha(inp)
    config=deepcopy(requested)
    training=config['training']
    fields={
        'siren':{'epochs','batch_size','seed','lr'},
        'finer':{'epochs','batch_size','seed','lr'},
        'miner':{'epochs','scales','seed'},
        'neural_experts':{'epochs','batch_size','seed'},
        'ingp':{'epochs','batch_size'},
        'odie':{'epochs','batch_size','seed','unlock_epochs','epochs_per_stage','learning_rates','lr_schedule'},
    }
    if set(training)-fields[method]:
        raise ValueError(f'Unsupported training overrides: {set(training)-fields[method]}')
    for key in ('epochs','batch_size','scales'):
        if key in training and (type(training[key]) is not int or training[key]<1):
            raise ValueError(f'training.{key} must be a positive integer')
    if type(config['evaluation'].get('interval')) is not int or config['evaluation']['interval']<1:
        raise ValueError('evaluation.interval must be a positive integer')
    if method!='odie' and ('model' in config or 'data' in config):
        raise ValueError('These baseline entries inherit the pinned official model/data rules; no unused overrides are accepted')
    pretraining_input=getattr(args,'pretraining_input',None)
    pretraining_output=getattr(args,'pretraining_output',None)
    budget_input_dir=getattr(args,'budget_input_dir',None)
    if method=='neural_experts':
        if pretraining_input is None or not pretraining_input.is_file():
            raise ValueError('Neural Experts requires --pretraining-input (official protocol reference: kodim19)')
        if pretraining_output is None:
            raise ValueError('Neural Experts requires --pretraining-output shared by all image fits')
        shared=pretraining_output.resolve()
        output=args.output.resolve()
        if shared==output or shared.is_relative_to(output) or output.is_relative_to(shared):
            raise ValueError('Shared pretraining and individual fit outputs must be separate directories')
    elif any(value is not None for value in (pretraining_input,pretraining_output,budget_input_dir)):
        raise ValueError('Pretraining and mean-budget arguments are only for Neural Experts')
    if set(config['evaluation'])-({'interval','batch_size'} if method=='odie' else {'interval'}):
        raise ValueError('Unknown evaluation setting')
    if method=='miner' and config['evaluation']['interval']!=1:
        raise ValueError('MINER retains its native per-epoch logging')
    if type(training.get('epochs')) is not int or training['epochs']<1:
        raise ValueError('training.epochs must be a positive integer')
    if method!='miner' and (type(training.get('batch_size')) is not int or training['batch_size']<1):
        raise ValueError('training.batch_size must be explicit and positive')
    if type(config['evaluation'].get('interval')) is not int or config['evaluation']['interval']<1:
        raise ValueError('evaluation.interval must be positive')
    vendor_root=args.vendor_root.resolve()
    if args.ingp_run is not None:
        raise ValueError('Only ODIE may read --ingp-run; use its shared experiment entry')
    upstream=read(ROOT/'upstreams.lock.json')[method]
    repo=vendor_root/method
    if repo.exists() and any(repo.iterdir()):
        verify(vendor_root,method)
        official_settings=official(method,repo)
    elif args.dry_run:
        cache=read(ROOT/'configs/reference/official-defaults.json')[method]
        if cache['commit']!=upstream['commit']:
            raise ValueError('Cached official defaults do not match the lock')
        official_settings=cache['defaults']
    else:
        raise FileNotFoundError(f'Prepare the vendor checkout first: {repo}')
    resolved=dict(method=method,scene=scene,experiment_group=config['experiment_group'],input=str(inp),
        input_sha256=digest,shape_hw=list(shape),input_bytes=inp.stat().st_size,requested_config=requested,
        official_defaults=official_settings,upstream_commit=upstream['commit'] if upstream else None,
        output=str(args.output.resolve()),training=training,evaluation=config['evaluation'],
        vendor_root=str(vendor_root),capacity_basis=config['capacity']['basis'])
    spec=args.capacity or config['capacity'].get('budget')
    if not spec:
        raise ValueError('Provide --capacity or capacity.budget in the config')
    if spec.startswith('png:') and inp.suffix.lower()!='.png':
        raise ValueError('png: budgets require an original PNG input')
    precision_bytes=2 if method=='ingp' else 4
    if method=='neural_experts':
        if config['capacity'].get('scope')!='dataset_mean' or not spec.startswith('png:'):
            raise ValueError('Neural Experts uses a fixed network from the mean PNG-byte budget')
        image_ids=config['capacity']['image_ids']
        if (not image_ids or len(image_ids)!=len(set(image_ids))
                or any(not isinstance(name,str) or len(name)!=4 or not name.isdigit() for name in image_ids)):
            raise ValueError('Mean-budget image_ids must be unique four-digit DIV2K IDs')
        directory=budget_input_dir.resolve() if budget_input_dir is not None else inp.parent
        images=[]
        for image_id in image_ids:
            path=directory/(image_id+'.png')
            images.append(dict(input=str(path.resolve()),bytes=path.stat().st_size,sha256=sha(path)))
        if not any(row['sha256']==digest and Path(row['input']).name==inp.name for row in images):
            raise ValueError('The fitted image must belong to the mean-budget reference set')
        _,total_amount=capacity.budget(spec,sum(row['bytes'] for row in images),precision_bytes)
        amount=total_amount//len(images)
        target=amount//precision_bytes
        resolved['capacity_reference']=dict(scope='dataset_mean',images=images,
            mean_input_bytes=sum(row['bytes'] for row in images)/len(images),
            mean_parameter_budget_bytes=amount)
    else:
        target,amount=capacity.budget(spec,inp.stat().st_size,precision_bytes)
    resolved['capacity_input']=spec
    if config['capacity']['basis'] not in ('parameter_payload_bytes','trainable_parameters'):
        raise ValueError('Capacity must be a parameter budget; inference-file bytes are measured separately')
    if config['capacity']['basis']=='trainable_parameters' and not spec.startswith('params:'):
        raise ValueError('trainable_parameters capacity requires params:N')
    if method in ('siren','finer'):
        model=deepcopy(official_settings['model'])
        allocation=capacity.mlp(target,model['hidden_layers'])
        model.update(in_features=2,out_features=3,hidden_features=allocation['width'])
        resolved['model']=model
        resolved['training']={**official_settings['training'],**training}
    elif method=='neural_experts':
        allocation=capacity.neural_experts(target)
        resolved['model']=dict(expert_width=allocation['expert_width'],encoders_router_width=128,n_experts=4)
        resolved['pretraining']=dict(input=str(pretraining_input.resolve()),
            input_sha256=sha(pretraining_input),output=str(shared),excluded_from_fitting_time=True)
    elif method=='miner':
        settings=deepcopy(official_settings['config'])
        settings['TRAINING']['epochs']=training['epochs']
        allocation=capacity.miner(target,shape,settings,training['scales'],
            config['capacity']['full_allocation_factor'])
        settings['FOLDING'].update(ksize=allocation['ksize'],stride=allocation['stride'])
        resolved['native_config']=settings
    elif method=='ingp':
        if training['batch_size']%256:
            raise ValueError('INGP batch_size must be a multiple of 256')
        finest=max(shape)/2
        # These are the native MLP parameters, not file-format overhead.
        network_parameters=7168
        if scene=='div2k':
            allocation=capacity.ingp_div2k(target,finest)
        else:
            allocation=capacity.hash_grid(target,finest,network_parameters,
                max_base=config['capacity'].get('max_base',int(finest)))
        resolved['finest_resolution']=finest
        native=deepcopy(official_settings['config'])
        native['encoding'].update({k:allocation[k] for k in ('base_resolution','log2_hashmap_size')})
        # The pinned Image entry derives the scale after loading the
        # image. Keep the external estimate out of the native input.
        native['encoding'].pop('per_level_scale',None)
        updates=training['epochs']*math.ceil(math.prod(shape)/training['batch_size'])
        # Both image protocols retain the official decay positions as a
        # fraction of its default 35000 training updates.
        native['optimizer']['decay_start']=round(updates*20000/35000)
        native['optimizer']['decay_interval']=max(1,round(updates*10000/35000))
        resolved['native_config']=native
        build_override=args.pyngp_build or os.environ.get('ODIE_PYNGP_BUILD')
        resolved['pyngp_build']=str(build_override or repo/'build')
        resolved['pyngp_expected_sha256']=None if build_override else upstream.get('module_sha256')
    else:
        raise ValueError(method)
    allocation.update(target_bytes=amount,capacity_verified=False)
    resolved['allocation']=allocation
    resolved['expected_updates']=training['epochs']*math.ceil(math.prod(shape)/training.get('batch_size',math.prod(shape)))
    resolved['automatic_retry']=False
    return resolved


def evaluate_run(plan):
    """Pass the selected RGB8 export to the shared evaluator after training."""
    from scripts.evaluate_reconstruction import evaluate
    output=Path(plan['output'])
    method=plan['method']
    fit=output
    native=read(fit/'metrics.json')
    checkpoint=native['checkpoint']
    if method=='miner':
        files=[*sorted((fit/'scales').glob('*.pt')),fit/'settings.json']
        if len(files)<2:
            raise ValueError('MINER did not export scale weights')
    elif method=='ingp':
        files=[fit/'model_latest.msgpack',fit/'image_metadata.json']
    elif method=='odie':
        files=[fit/'model_latest_fp16.pt']
    else:
        files=[fit/f'model_{checkpoint}.pt']
    result=evaluate(Path(plan['input']),fit/'reconstruction.png',files,checkpoint,
                    odie_metrics=native if method=='odie' else None)
    write(output/'metrics.json',dict(training=native,evaluation=result))
    print(f"RGB8 evaluation: PSNR {result['psnr_db']} dB | SSIM {result['ssim']}",flush=True)


def main(method,scene,default_config=None):
    if method=='odie':
        from .odie import main as run_odie
        return run_odie(scene,default_config)
    parser=argparse.ArgumentParser(description=f'{method} / {scene}: pinned upstream implementation with ODIE integration patches')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=default_config or ROOT/f'configs/baselines/{method}_{scene}.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capacity',help='Override config capacity: 100MB, 50MiB, png:0.5, or params:N')
    parser.add_argument('--ingp-run',type=Path,help='ODIE only: native-verified capacity.json or its run directory')
    parser.add_argument('--vendor-root',type=Path,default=Path(os.environ.get('ODIE_VENDOR_ROOT',ROOT/'vendor')))
    parser.add_argument('--pyngp-build',type=Path,help='INGP only: compiled module from this pinned engine')
    parser.add_argument('--pretraining-input',type=Path,help='Neural Experts only: RGB segmentation pretraining image (kodim19)')
    parser.add_argument('--pretraining-output',type=Path,help='Neural Experts only: shared pretraining directory, created once and reused')
    parser.add_argument('--budget-input-dir',type=Path,help='Neural Experts only: directory containing the ten budget-reference PNGs; defaults to the input directory')
    parser.add_argument('--gpu-lock',type=Path,default=Path(os.environ.get('ODIE_GPU_LOCK',ROOT/'results/sequential-sweep.gpu.lock')))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='Read config/defaults and calculate CPU estimates; write nothing')
    mode.add_argument('--prepare-only',action='store_true',help='Write a run plan and verify native capacity, without training')
    args=parser.parse_args()
    plan=resolve(method,scene,args)
    if args.dry_run:
        import json
        print(json.dumps(plan,indent=2,ensure_ascii=False))
        return
    if sys.platform!='linux':
        parser.error('Training and dependency preparation require Linux')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    source_files={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    source_files.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')})
    if method=='odie':
        source_files.update({name:sha(ROOT/name) for name in read(ROOT/'upstreams.lock.json')['odie']['compression_files']})
    update_run(output,method=method,plan=plan,source=dict(files=source_files,argv=sys.argv,
        python=sys.executable,upstream_commit=plan['upstream_commit']),
        created_at=now(),state='waiting_for_gpu_lock',pid=os.getpid())
    import fcntl
    args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):
        raise SystemExit(128+signum)
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with run_log(output):
            try:
                with args.gpu_lock.open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    from .execute import execute
                    update_run(output,state='preparing')
                    execute(plan,args.prepare_only)
                    if not args.prepare_only:
                        update_run(output,state='evaluating')
                        evaluate_run(plan)
                update_run(output,state='prepared' if args.prepare_only else 'completed',finished_at=now())
            except BaseException as error:
                update_run(output,state='failed',error=f'{type(error).__name__}: {error}',finished_at=now())
                raise
    finally:
        for s,handler in previous.items():
            signal.signal(s,handler)
