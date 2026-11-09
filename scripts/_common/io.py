"""Small shared helpers; planning never imports Torch or starts training."""
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def update_run(output, **fields):
    path = Path(output)/'run.json'
    value = read(path) if path.exists() else {}
    value.update(fields)
    write(path, value)


@contextmanager
def run_log(output):
    """Collect entry and child-process output in the same log."""
    import sys
    class LogStream:
        def __init__(self, console, log):
            self.console, self.log = console, log
        def isatty(self):
            return self.console.isatty()
        def write(self, value):
            if self.isatty():
                self.console.write(value)
            return self.log.write(value)
        def flush(self):
            self.console.flush()
            self.log.flush()
    with (Path(output)/'train.log').open('a', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(LogStream(sys.stdout,stream)), redirect_stderr(LogStream(sys.stderr,stream)):
            try:
                yield
            except BaseException:
                import traceback
                traceback.print_exc()
                raise


def record(output):
    """Bind a run to its official implementation and actually imported project files."""
    import os
    import sys
    from .vendor import verify
    repo = Path(os.environ['ODIE_OFFICIAL_REPO']).resolve()
    _, receipt = verify(repo.parent, repo.name)
    files = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and (path.is_relative_to(repo) or path.is_relative_to(ROOT)):
                files[str(path)] = sha(path)
    update_run(output, runtime_sources=dict(
        official=receipt, argv=sys.argv, executable=sys.executable,
        entry_sha256=sha(Path(sys.argv[0])), imported_files=files))
