"""Run a manifest strictly serially; each worker owns and cleans its input cache."""
import argparse
import gzip
import hashlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._common.io import ROOT,now,read,write,update_run
from scripts._common.cache import cleanup


def jobs(manifest,vendor_root):
    result=[]
    outputs=set()
    for row in read(manifest):
        if set(row)-{'method','scene','input','config','output','capacity','ingp_run','pyngp_build','pretraining_input','pretraining_output','budget_input_dir','python','archive_reconstruction'}:
            raise ValueError('Unknown serial manifest field')
        if row['method']!='odie' and 'ingp_run' in row:
            raise ValueError('Only ODIE may read an INGP reference')
        script=ROOT/'scripts'/f'run_{row["method"]}_{row["scene"]}.py'
        if not script.is_file():
            raise ValueError(f'Unsupported experiment: {script.name}')
        python=row.get('python',sys.executable)
        if not Path(python).is_file():
            raise FileNotFoundError(f'Missing Python runtime: {python}')
        argv=[python,'-u',str(script),'--vendor-root',str(vendor_root)]
        paths={}
        for key in ('input','config','output','ingp_run','pyngp_build','pretraining_input','pretraining_output','budget_input_dir'):
            if key in row:
                paths[key]=(manifest.parent/row[key]).resolve()
                argv.extend(['--'+key.replace('_','-'),str(paths[key])])
        if not paths['input'].is_file() or not paths['config'].is_file():
            raise FileNotFoundError(f'Missing input/config for {row["output"]}')
        if row['method']=='neural_experts':
            if 'pretraining_input' not in paths or not paths['pretraining_input'].is_file():
                raise FileNotFoundError('Neural Experts requires a pretraining_input image')
            if 'pretraining_output' not in paths:
                raise ValueError('Neural Experts requires a shared pretraining_output directory')
        elif any(key in paths for key in ('pretraining_input','pretraining_output','budget_input_dir')):
            raise ValueError('Pretraining and mean-budget arguments are only for Neural Experts')
        if paths['output'] in outputs or (paths['output'].exists() and any(paths['output'].iterdir())):
            raise ValueError(f'Output must be unique, new or empty: {paths["output"]}')
        if row['method']=='odie':
            reference=paths.get('ingp_run')
            if reference is None or (not reference.exists() and reference not in outputs):
                raise ValueError('ODIE must reference an existing INGP record or a preceding INGP job')
        outputs.add(paths['output'])
        if 'capacity' in row:
            argv.extend(['--capacity',row['capacity']])
        result.append(dict(argv=argv,output=paths['output'],method=row['method'],
            archive_reconstruction=row.get('archive_reconstruction',False)))
    return result


def archive_reconstruction(output):
    """Keep exact floating output in gzip; verify bytes before releasing raw NPY."""
    source=output/'reconstruction.npy'
    # New runs already save losslessly compressed RGB8 PNG.
    if (output/'reconstruction.png').is_file() or (output/'fit/reconstruction.png').is_file():
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    destination=source.with_suffix('.npy.gz')
    digest=hashlib.sha256()
    with source.open('rb') as raw,destination.open('xb') as packed:
        with gzip.GzipFile(filename='',mode='wb',fileobj=packed,compresslevel=1,mtime=0) as archive:
            for block in iter(lambda:raw.read(8*1024**2),b''):
                digest.update(block)
                archive.write(block)
    restored=hashlib.sha256()
    with gzip.open(destination,'rb') as archive:
        for block in iter(lambda:archive.read(8*1024**2),b''):
            restored.update(block)
    if restored.digest()!=digest.digest():
        raise RuntimeError('Lossless reconstruction archive failed byte verification')
    write(output/'reconstruction_archive.json',dict(raw_name=source.name,
        raw_bytes=source.stat().st_size,raw_sha256=digest.hexdigest(),
        archive_name=destination.name,archive_bytes=destination.stat().st_size,
        lossless_byte_roundtrip_verified=True))
    source.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--vendor-root',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    plan=jobs(args.manifest.resolve(),args.vendor_root.resolve())
    if args.dry_run:
        for job in plan:
            print(subprocess.list2cmdline(job['argv']))
        return
    for job in plan:
        root=job['output']
        print(f'Starting {job["method"]}: {root}',flush=True)
        root.parent.mkdir(parents=True,exist_ok=True)
        # Entries create their own train.log after validating the output.
        # Keep startup output in a temporary stream until it can be appended.
        with tempfile.TemporaryFile(mode='w+') as stream:
            child=subprocess.Popen(job['argv'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            def stop(signum,frame):
                raise SystemExit(128+signum)
            previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
            stopped=None
            try:
                code=child.wait()
            except SystemExit as error:
                stopped=error
                for s in previous:
                    signal.signal(s,signal.SIG_IGN)
                child.terminate()
                # Entry cleanup has its own 30-second worker termination bound.
                try:
                    code=child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise RuntimeError('Entry did not terminate; stop and inspect its worker before cleaning caches')
            finally:
                for s,handler in previous.items():
                    signal.signal(s,handler)
            root.mkdir(parents=True,exist_ok=True)
            stream.seek(0)
            with (root/'train.log').open('a') as log:
                shutil.copyfileobj(stream,log)
        cleanup(root)
        if (root/'tokyo.bin').exists():
            raise RuntimeError(f'Input cache remains after worker exit; stop before the next experiment: {root}')
        update_run(root,serial=dict(returncode=code,cache_absent=True,finished_at=now()))
        if stopped:
            raise stopped
        if code:
            raise RuntimeError(f'Experiment failed; serial execution stopped: {root}/train.log')
        if job['archive_reconstruction']:
            archive_reconstruction(root)
        print(f'Completed {job["method"]}: {root}',flush=True)


if __name__=='__main__':
    main()
