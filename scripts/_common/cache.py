"""Own only this run's disposable half4 input; never delete original images."""
from contextlib import contextmanager
from pathlib import Path
import shutil

from .io import now,read,update_run


def cleanup(output):
    output=Path(output).resolve()
    cache=output/'tokyo.bin'
    run=read(output/'run.json') if (output/'run.json').exists() else {}
    owner=run.get('input_cache')
    if not owner:
        if cache.exists():
            raise RuntimeError(f'Refuse to remove an unowned cache: {cache}')
        return
    if owner.get('path')!=str(cache) or owner.get('output')!=str(output) or cache.is_symlink():
        raise RuntimeError('Input cache ownership/path mismatch')
    if not cache.exists() and owner.get('cache_absent'):
        return
    size=cache.stat().st_size if cache.exists() else 0
    cache.unlink(missing_ok=True)
    update_run(output,input_cache=dict(owner,bytes=size,cache_absent=True,finished_at=now()))


@contextmanager
def input_cache(plan):
    output=Path(plan['output']).resolve()
    cache=output/'tokyo.bin'
    if cache.exists() or cache.is_symlink():
        raise FileExistsError(f'Never overwrite an existing input cache: {cache}')
    needed=8+plan['shape_hw'][0]*plan['shape_hw'][1]*8+12*1024**3
    if shutil.disk_usage(output).free<needed:
        raise RuntimeError('Insufficient disk for this half4 input cache plus 12 GiB reserve')
    update_run(output,input_cache=dict(path=str(cache),output=str(output),
        input_sha256=plan['input_sha256'],created_at=now()))
    try:
        yield cache
    finally:
        cleanup(output)
