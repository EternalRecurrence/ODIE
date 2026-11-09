"""Launch frozen training entries with resolved experiment inputs."""
import configparser
from copy import deepcopy
import gc
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .io import HERE, ROOT, read, sha, write, update_run
from .vendor import verify
from .cache import input_cache


def capacity_record(plan, count=None, verified=False, **extra):
    record=dict(method=plan['method'],experiment_group=plan['experiment_group'],
        input_sha256=plan['input_sha256'],shape_hw=plan['shape_hw'],
        parameter_count=count if count is not None else plan['allocation'].get('parameter_count'),
        capacity_verified=verified,capacity_basis=plan['capacity_basis'],
        target_bytes=plan['allocation']['target_bytes'],allocation=plan['allocation'],
        upstream_commit=plan['upstream_commit'],**extra)
    if 'capacity_reference' in plan:
        record['capacity_reference']=plan['capacity_reference']
    write(Path(plan['output'])/'capacity.json',record)
    return record


def command(argv, cwd, plan, label='train', env=None):
    output=Path(plan['output'])
    commands=read(output/'run.json').get('commands',{})
    commands[label]=dict(argv=[str(a) for a in argv],cwd=str(cwd))
    update_run(output,commands=commands,state='training' if label=='train' else 'preparing')
    print(f"Starting {label}: {' '.join(str(a) for a in argv)}",flush=True)
    with (output/'train.log').open('a') as stream:
        process=subprocess.Popen([str(a) for a in argv],cwd=cwd,env=env,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        def terminate(signum,frame):
            # Never call Popen.wait inside a signal handler: it can interrupt
            # another wait while its internal waitpid lock is held.
            raise SystemExit(128+signum)
        previous={s:signal.signal(s,terminate) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            process.wait()
        except BaseException:
            for s in previous:
                signal.signal(s,signal.SIG_IGN)
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            raise
        finally:
            for s,handler in previous.items():
                signal.signal(s,handler)
    commands[label]['returncode']=process.returncode
    update_run(output,commands=commands)
    if process.returncode:
        raise RuntimeError(f'{label} failed ({process.returncode}); see {output}/train.log')


def verify_ingp(plan):
    import msgpack
    build=Path(plan['pyngp_build'])
    modules=list(build.glob('pyngp*.so'))
    if len(modules)!=1:
        raise FileNotFoundError(f'Expected exactly one built official pyngp module in {build}')
    digest=sha(modules[0])
    if plan['pyngp_expected_sha256'] and digest!=plan['pyngp_expected_sha256']:
        raise ValueError('Built pyngp module does not match the expected SHA256')
    sys.path.insert(0,str(build))
    import pyngp as ngp
    tb=ngp.Testbed(ngp.TestbedMode.Image)
    config=deepcopy(plan['native_config'])
    # Official automatic scale needs the actual image dimensions first.
    data=Path(plan['output'])/'tokyo.bin' if plan['scene']=='gigapixel' else Path(plan['input'])
    tb.load_training_data(str(data))
    scratch=Path(plan['output'])/'capacity-probe.msgpack'
    estimate=deepcopy(plan['allocation'])
    probes=[]
    while True:
        tb.reload_network_from_json(config)
        count=tb.n_params()
        tb.save_snapshot(str(scratch),False,False)
        packed=msgpack.unpackb(scratch.read_bytes(),raw=False,strict_map_key=False)
        native=packed['snapshot']
        assert native['n_params']==count and len(native['params_binary'])==count*2
        assert native['params_type']=='__half' and 'optimizer' not in native
        size=len(msgpack.packb(packed,use_bin_type=True))
        scratch.unlink()
        probes.append(dict(base_resolution=config['encoding']['base_resolution'],
            parameter_count=count,package_bytes=size))
        if count*2<=plan['allocation']['target_bytes']:
            break
        # Float rounding in the official automatic scale may differ from the
        # CPU estimate. Capacity-only calibration is recorded, before training.
        if config['encoding']['base_resolution']<=1:
            raise ValueError('Native automatically scaled grid cannot fit the capacity')
        config['encoding']['base_resolution']-=1
        # Release the previous image/trainer before any replacement allocation.
        # A second live trainer can exceed VRAM for the Vela image even with this budget.
        del tb
        gc.collect()
        ngp.free_temporary_memory()
        tb=ngp.Testbed(ngp.TestbedMode.Image)
        tb.load_training_data(str(data))
    selected=dict(config=config,parameter_count=count,parameter_bytes=count*2,
        encoding_parameters=tb.n_encoding_params(),params_type='__half',package_bytes=size,
        optimizer_state=False,compression=False,
        required_sidecar='image_metadata.json; dimensions and official Image color settings')
    del tb
    gc.collect()
    ngp.free_temporary_memory()
    from .capacity import grid_count
    _,scale=grid_count(config['encoding']['base_resolution'],config['encoding']['log2_hashmap_size'],plan['finest_resolution'])
    plan['native_config']=config
    plan['allocation'].update(parameter_count=count,capacity_verified=True,package_bytes=size,
        base_resolution=config['encoding']['base_resolution'],per_level_scale=scale)
    calibration=dict(status='calibrated',selected=selected,
        shape_hwc=[*plan['shape_hw'],3],source_commit=plan['upstream_commit'],input_sha256=plan['input_sha256'],
        quality_based_selection=False,cpu_estimate=estimate,native_probes=probes,
        resolution_policy='official Image automatic max(H,W)/2; no per_level_scale input')
    capacity_record(plan,count,True,parameter_bytes=count*2,
        native_module_sha256=digest,native_module=str(modules[0]),package_bytes=size,
        calibration=calibration)
    write(Path(plan['output'])/'config.json',config)
    return selected


def prepare_neural_pretraining(plan, repo, model, prepare_only, env):
    """Create one official RGB pretraining run, then reuse its complete state."""
    import yaml
    from .native_baselines import check_pretrained, neural_pretraining_config, RGB_PRETRAINING_CONFIG
    output=Path(plan['output'])
    shared=Path(plan['pretraining']['output'])
    image=Path(plan['pretraining']['input'])
    if sha(image)!=plan['pretraining']['input_sha256']:
        raise ValueError('Pretraining image changed after plan resolution')
    config=neural_pretraining_config(repo,plan['allocation']['expert_width'],image,plan['training']['seed'])
    text=yaml.safe_dump(config,sort_keys=False)
    config_path=shared/'requested-config.yaml'
    checkpoint=shared/'trained_models'/f"{config['MODEL']['model_name']}_model_{config['TRAINING']['num_epochs']-1}.pth"
    argv=[sys.executable,'-u',repo/'image_reconstruction/train_rgbimage.py',
        '--config',config_path,'--logdir',shared.parent,'--identifier',shared.name,'--image_id',image.name]
    identity=dict(input_sha256=plan['pretraining']['input_sha256'],
        config_sha256=hashlib.sha256(text.encode()).hexdigest(),
        upstream_commit=plan['upstream_commit'],patch_sha256=sha(ROOT/'patches/neural_experts.patch'),
        parameter_count=plan['allocation']['parameter_count'])
    record=dict(**plan['pretraining'],identity=identity,path=str(checkpoint),status='planned',
        official_config=RGB_PRETRAINING_CONFIG,official_config_sha256=sha(repo/RGB_PRETRAINING_CONFIG),
        rgb_defaults_sha256=sha(repo/'configs/config_RGB.yaml'),epochs=config['TRAINING']['num_epochs'])
    commands=read(output/'run.json').get('commands',{})
    commands['pretrain']=dict(argv=[str(a) for a in argv],cwd=str(repo/'image_reconstruction'))
    update_run(output,commands=commands)
    if prepare_only:
        return record

    import fcntl
    shared.parent.mkdir(parents=True,exist_ok=True)
    with (shared.parent/(shared.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=shared/'pretraining.json'
        if receipt.exists():
            saved=read(receipt)
            if saved.get('status')!='completed' or saved.get('identity')!=identity:
                raise ValueError('Shared pretraining is incomplete or uses a different input/config/source; use a new directory')
            if sha(checkpoint)!=saved['sha256']:
                raise ValueError('Shared pretraining checkpoint changed')
            loaded=check_pretrained(model,checkpoint)
            record={**saved,**loaded,'reused':True}
            print(f'Reusing pretrained Neural Experts model: {checkpoint}',flush=True)
        else:
            if shared.exists() and any(shared.iterdir()):
                raise ValueError('Shared pretraining output must be new or have a completed receipt')
            shared.mkdir(parents=True,exist_ok=True)
            config_path.write_text(text,encoding='utf-8')
            write(receipt,{**record,'status':'training'})
            try:
                command(argv,repo/'image_reconstruction',plan,'pretrain',env)
                metrics=read(shared/'metrics.json')
                if (metrics['optimizer_steps']!=config['TRAINING']['num_epochs']
                        or metrics['parameter_count']!=plan['allocation']['parameter_count']):
                    raise ValueError('RGB pretraining capacity/update count mismatch')
                record.update(check_pretrained(model,checkpoint))
                record.update(status='completed',training_loop_seconds=metrics['training_loop_seconds'],reused=False)
                write(receipt,record)
            except BaseException as error:
                write(receipt,{**record,'status':'failed','error':f'{type(error).__name__}: {error}'})
                raise
    commands=read(output/'run.json')['commands']
    commands['pretrain']['reused']=record['reused']
    update_run(output,commands=commands)
    return record


def execute(plan, prepare_only):
    method=plan['method']
    output=Path(plan['output'])
    repo,receipt=verify(Path(plan['vendor_root']),method)
    os.environ['ODIE_OFFICIAL_REPO']=str(repo)
    worker_env=dict(os.environ,PYTHONPATH=str(repo),PYTHONUNBUFFERED='1')
    update_run(output,vendor=receipt)
    capacity_record(plan)
    # Some official entries require a new directory. Flatten their completed
    # outputs without changing the official directory setup logic.
    fit=output/'.training'
    t=plan['training']
    if method in ('siren','finer','neural_experts'):
        from .native_baselines import model_for_plan
        model,native=model_for_plan(plan,repo)
        count=sum(p.numel() for p in model.parameters())
        assert count==plan['allocation']['parameter_count']
        extra={}
        if method=='neural_experts':
            extra['pretraining']=prepare_neural_pretraining(plan,repo,model,prepare_only,worker_env)
            native['MODEL']['manager_pt_path']=extra['pretraining']['path']
            write(output/'pretraining.json',extra['pretraining'])
        del model
        capacity_record(plan,count,True,parameter_bytes=count*4,**extra)
        if method!='neural_experts':
            write(output/'config.json',dict(model=plan['model'],training=t,evaluation=plan['evaluation']))
        if method=='siren':
            entry='experiment_scripts/train_img.py'
            argv=[sys.executable,'-u',repo/entry,'--input',plan['input'],
                '--logging_root',output,'--experiment_name','.training',
                '--num_epochs',t['epochs'],'--batch_size',t['batch_size'],
                '--hidden_features',plan['allocation']['width'],'--seed',t['seed'],
                '--epochs_til_ckpt',plan['evaluation']['interval']]
            for key in ('lr',):
                if key in plan['requested_config']['training']:
                    argv += ['--'+key,t[key]]
        elif method=='finer':
            entry='run_finer.sh'
            argv=['bash',repo/entry,'--input',plan['input'],'--logdir',output,
                '--exp_name','.training','--num_epochs',t['epochs'],
                '--batch_size',t['batch_size'],'--seed',t['seed'],
                '--hidden_features',plan['allocation']['width'],
                '--steps_til_summary',plan['evaluation']['interval']]
            for key in ('lr',):
                if key in plan['requested_config']['training']:
                    argv += ['--'+key,t[key]]
            # The official shell invokes "python". Bind it to this run's
            # selected environment, even when launched from a different shell.
            worker_env['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
        else:
            import yaml
            entry='image_reconstruction/train_rgbimage.py'
            native['seed']=t['seed']
            native['DATA']['dataset_path']=str(Path(plan['input']).parent)
            native['DATA']['pixel_samples']=True
            native['TRAINING'].update(num_epochs=t['epochs'],batch_size=t['batch_size'],
                eval_interval=plan['evaluation']['interval'])
            (output/'config.yaml').write_text(yaml.safe_dump(native,sort_keys=False))
            argv=[sys.executable,'-u',repo/entry,'--config',output/'config.yaml',
                '--logdir',output,'--identifier','.training','--image_id',Path(plan['input']).name]
        cwd=repo/'image_reconstruction' if method=='neural_experts' else repo
        native_entry(plan,repo,entry,argv,cwd,prepare_only,worker_env)
        if not prepare_only:
            metrics=read(fit/'metrics.json')
            if metrics['parameter_count']!=count or metrics['optimizer_steps']!=plan['expected_updates']:
                raise ValueError('Official entry did not complete the requested capacity/updates')
    elif method=='miner':
        conf=configparser.ConfigParser()
        conf.read_dict({s:{k:str(v) for k,v in values.items()} for s,values in plan['native_config'].items()})
        with (output/'config.ini').open('w') as stream:
            conf.write(stream)
        entry='miner_image.py'
        argv=[sys.executable,'-u',repo/entry,'--input',plan['input'],
            '--config',output/'config.ini','--output',fit,
            '--scales',t['scales'],'--seed',t['seed']]
        native_entry(plan,repo,entry,argv,repo,prepare_only,worker_env)
        if not prepare_only:
            measured=read(fit/'metrics.json')['parameter_count']
            target=plan['allocation']['target_bytes']
            # Residual-driven block allocation only approximately matches the target.
            capacity_record(plan,measured,True,parameter_bytes=measured*4,
                capacity_policy='approximate',within_target=measured*4<=target,
                target_difference_fraction=measured*4/target-1)
    elif method=='ingp':
        return ingp(plan, repo, prepare_only)
    else:
        raise ValueError(method)
    if not prepare_only:
        for path in fit.iterdir():
            target=output/path.name
            if target.exists():
                if path.is_file() and target.is_file() and sha(path)==sha(target):
                    path.unlink()
                    continue
                raise FileExistsError(target)
            shutil.move(str(path),str(target))
        fit.rmdir()


def native_entry(plan,repo,entry,argv,cwd,prepare_only,env):
    """Record and launch an existing upstream file, never an added worker."""
    commands=read(Path(plan['output'])/'run.json').get('commands',{})
    commands['train']=dict(argv=[str(a) for a in argv],cwd=str(cwd))
    update_run(plan['output'],entry=entry,
        patch_sha256=sha(ROOT/'patches'/f"{plan['method']}.patch"),
        commands=commands)
    if not prepare_only:
        command(argv,cwd,plan,env=env)


def ingp(plan, repo, prepare_only):
    """Budget and launch the existing official Image entry; no training loop."""
    from contextlib import nullcontext
    output = Path(plan['output'])
    env = dict(os.environ, PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join([plan['pyngp_build'], str(repo/'scripts')]))
    data = Path(plan['input'])
    cache = input_cache(plan) if plan['scene'] == 'gigapixel' else nullcontext()
    with cache as converted:
        if converted is not None:
            data = converted
            command([sys.executable, '-u', repo/'scripts/convert_image.py',
                     '--input', plan['input'], '--output', data, '--stripe_rows', 128],
                    repo, plan, 'convert', env)
        verify_ingp(plan)
        update_run(output,plan=plan)
        argv = [sys.executable, '-u', repo/'scripts/run.py', '--scene', data,
                '--network', output/'config.json','--n_steps',plan['expected_updates'],
                '--width',plan['shape_hw'][1],'--height',plan['shape_hw'][0],
                '--save_snapshot',output/'model_latest.msgpack',
                '--training_batch_size',plan['training']['batch_size'],
                '--screenshot_dir',output,'--screenshot_spp',1]
        native_entry(plan,repo,'scripts/run.py',argv,repo,prepare_only,env)
        if prepare_only:
            return
        metrics = read(output/'metrics.json')
        if metrics['parameter_count'] != plan['allocation']['parameter_count']:
            raise ValueError('Native training count differs from calibrated INGP capacity')
        if metrics['optimizer_steps'] != plan['expected_updates']:
            raise ValueError('Official Image entry did not complete the requested updates')
