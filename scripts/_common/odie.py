"""Shared ODIE entry: one implementation for main experiments and ablations."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
from scripts._common.io import HERE,ROOT,now,read,sha,write,update_run,run_log
from scripts._common.run import reference,evaluate_run


def use_ingp_encoding(config, ref):
    """Use the corresponding verified INGP encoding, including Image auto-scale."""
    import math
    from odie.config import float32
    native = ref['calibration']['selected']
    encoding = deepcopy(native['config']['encoding'])
    if encoding['otype'] != 'HashGrid':
        raise ValueError('The no-on-demand reference must use HashGrid')
    layers, base = encoding['n_levels'], encoding['base_resolution']
    if not encoding.get('per_level_scale'):
        ratio = float32(float32(max(ref['shape_hw'])/2)/base)
        encoding['per_level_scale'] = float32(math.exp(float32(float32(math.log(ratio))/(layers-1))))
    config['model']['hash_encoding'] = encoding
    if config['model']['router_enabled']:
        from odie.config import align
        # Keep the reference HashGrid unchanged; record the extra router separately.
        config['budget']['trainable_parameters'] += 32*(
            align(layers*encoding['n_features_per_level'],16)+align(layers,16))


def main(scene='div2k', default_config=None):
    from odie.config import DEFAULT,merge,resolve
    ablation=scene=='ablation'
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ingp-run',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=default_config or ROOT/f'configs/baselines/odie_{scene}.json')
    parser.add_argument('--vendor-root',type=Path,help='Accepted by the common serial entry; ODIE uses the project odie package')
    parser.add_argument('--gpu-lock',type=Path,default=Path(os.environ.get('ODIE_GPU_LOCK',ROOT/'results/sequential-sweep.gpu.lock')))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    request=read(args.config)
    variant=request.get('ablation')
    if ablation and variant not in ('tokyo_no_router','tokyo_no_router_no_on_demand','tokyo_no_on_demand','expert_count'):
        raise ValueError('Unknown ODIE ablation')
    if request['capacity']!={'basis':'ingp_parameters'}:
        raise ValueError('ODIE reads the corresponding INGP parameter budget')
    with Image.open(args.input) as image:
        shape=[image.height,image.width]
    digest=sha(args.input)
    if ablation:
        if variant in ('tokyo_no_router','tokyo_no_router_no_on_demand','tokyo_no_on_demand'):
            if (digest!='c6c9f2288d8506998b2cf098f10d81b3e118703b9ecd27d4c075f2ec822fc16b' or
                request['model'].get('router_enabled') is not (variant=='tokyo_no_on_demand') or request['model']['layers']!=16):
                raise ValueError('Tokyo ablations require the original image, L16 and the matching router setting')
            if (variant!='tokyo_no_router') != (request['model'].get('on_demand',True) is False):
                raise ValueError('The no-on-demand ablations must explicitly disable on_demand')
        elif request['model']['layers'] not in (8,12,20) or request['model'].get('router_enabled',True) is not True:
            raise ValueError('Expert-count ablation requires 8/12/20 levels with router enabled; reuse the main experiment for 16 experts')
    ref=reference(args.ingp_run,digest,shape,request['experiment_group'])
    overrides={k:deepcopy(v) for k,v in request.items() if k not in ('experiment_group','capacity','ablation')}
    extra_model={k:overrides.get('model',{}).pop(k) for k in
        ('resolutions','feature_dimensions','max_vertices_per_level','hash_encoding') if k in overrides.get('model',{})}
    unlock=overrides['training'].pop('unlock_epochs',None)
    if unlock is not None:
        overrides['training']['epochs_per_stage']=1
    config=merge(DEFAULT,overrides)
    config['model'].update(extra_model)
    config['budget']={'trainable_parameters':ref['parameter_count']}
    if config['model']['on_demand'] is False:
        use_ingp_encoding(config,ref)
    if unlock is not None:
        sparse=config['model']['layers']-resolve(config,shape,args.input.stat().st_size)['initial_active_prefix']
        if not 1<=sparse<=unlock:
            raise ValueError('unlock_epochs must provide at least one epoch per sparse expert')
        config['training']['epochs_per_stage']=[unlock//sparse+(i<unlock%sparse) for i in range(sparse)]
    allocation=resolve(config,shape,args.input.stat().st_size)
    plan=dict(method='odie',scene=scene,ablation=variant,input=str(args.input.resolve()),input_sha256=digest,
        shape_hw=shape,experiment_group=request['experiment_group'],upstream_commit=None,
        output=str(args.output.resolve()),ingp_reference=ref,capacity_basis='ingp_parameters',
        allocation=dict(parameter_count=allocation['allocated_parameters'],target_bytes=allocation['fp16_parameter_budget'],
                        capacity_verified=False),unified_configuration=allocation)
    if not config['model']['on_demand']:
        plan['router_parameter_overhead']=allocation['gate_parameters']
    if args.dry_run:
        import json
        print(json.dumps(plan,indent=2))
        return
    if sys.platform!='linux':
        parser.error('Training requires Linux with NVIDIA CUDA')
    runtime=ROOT
    lock=read(ROOT/'upstreams.lock.json')['odie']
    files=lock['files']
    for relative,digest in files.items():
        if sha(runtime/'odie'/relative)!=digest:
            raise ValueError(f'ODIE source differs from the pinned experiment version: {relative}')
    output=Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    write(output/'config.json',config)
    update_run(output,method='odie',plan=plan,requested_config=request,
        created_at=now(),state='waiting_for_gpu_lock',pid=os.getpid(),
        source=dict(source_root=str(runtime.relative_to(ROOT)),
        files=files,
        entry_sha256=sha(Path(__file__)),ingp_reference_sha256=sha(args.ingp_run/'capacity.json' if args.ingp_run.is_dir() else args.ingp_run)))
    import fcntl
    from scripts._common.execute import command,capacity_record
    args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    with run_log(output):
        try:
            with args.gpu_lock.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                capacity_record(plan,verified=False,reference_parameters=ref['parameter_count'],
                    alignment_slack=allocation['parameter_budget']-allocation['allocated_parameters'],
                    router_parameter_overhead=plan.get('router_parameter_overhead',0))
                if not args.prepare_only:
                    env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONUNBUFFERED='1')
                    command([sys.executable,'-u','-m','odie','run','--input',plan['input'],
                        '--config',output/'config.json','--output',output],runtime,plan,env=env)
                    metrics=read(output/'metrics.json')
                    if metrics['allocated_parameter_count']!=allocation['allocated_parameters']:
                        raise ValueError('Fitted ODIE parameter count differs from the resolved allocation')
                    capacity_record(plan,verified=True,reference_parameters=ref['parameter_count'],
                        alignment_slack=allocation['parameter_budget']-allocation['allocated_parameters'],
                        router_parameter_overhead=plan.get('router_parameter_overhead',0))
                    update_run(output,state='evaluating')
                    evaluate_run(plan)
            update_run(output,state='prepared' if args.prepare_only else 'completed',finished_at=now())
        except BaseException as error:
            update_run(output,state='failed',error=f'{type(error).__name__}: {error}',finished_at=now())
            raise
