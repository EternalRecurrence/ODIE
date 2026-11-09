"""Prepare pinned official dependencies and apply the separately tracked patches."""
import argparse
from pathlib import Path
import subprocess
import sys

from .io import ROOT, read, sha, write


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def patch_files(method):
    locked = read(ROOT/'upstreams.lock.json')[method]['patches']
    files = [ROOT/'patches'/name for name in locked]
    for path in files:
        if sha(path) != locked[path.name]:
            raise ValueError(f'{method}: patch hash mismatch: {path.name}')
    return files


def submodules(repo):
    subprocess.run(['git', '-C', str(repo), 'submodule', 'update', '--init', '--recursive'], check=True)


def verify(root, method):
    lock = read(ROOT/'upstreams.lock.json')[method]
    repo = Path(root)/method
    receipt = read(repo/'.odie-vendor.json')
    if receipt['upstream_commit'] != lock['commit'] or git(repo, 'rev-parse', 'HEAD') != lock['commit']:
        raise ValueError(f'{method}: incorrect upstream commit')
    expected = {p.name:sha(p) for p in patch_files(method)}
    if receipt['patches'] != expected:
        raise ValueError(f'{method}: patches changed; prepare a new vendor directory')
    for relative, digest in lock['implementation_files'].items():
        if sha(repo/relative) != digest:
            raise ValueError(f'{method}: implementation mismatch: {relative}')
    changed = set(git(repo, 'diff', 'HEAD', '--name-only', '--ignore-submodules=all').splitlines())
    if changed != set(lock['implementation_files']):
        raise ValueError(f'{method}: unexpected implementation diff: {changed}')
    if method == 'ingp':
        actual = [' '.join(line.strip().split()[:2]) for line in git(repo, 'submodule', 'status', '--recursive').splitlines()]
        expected_modules = [' '.join(line.strip().split()[:2]) for line in lock['submodules'].splitlines()]
        if actual != expected_modules:
            raise ValueError('I-NGP submodules differ from the pinned revisions')
    return repo, receipt


def prepare(root, methods):
    if sys.platform != 'linux':
        raise RuntimeError('Prepare official dependencies on Linux')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for method in methods:
        lock = read(ROOT/'upstreams.lock.json')[method]
        repo = root/method
        if (repo/'.odie-vendor.json').exists():
            verify(root, method)
            print(f'{method}: existing checkout verified', flush=True)
            continue
        if repo.exists() and any(repo.iterdir()):
            if git(repo, 'rev-parse', 'HEAD') != lock['commit'] or git(repo, 'diff', 'HEAD', '--name-only', '--ignore-submodules=all'):
                raise ValueError(f'Use an empty directory or a clean pinned checkout: {repo}')
        else:
            subprocess.run(['git', 'clone', '--no-checkout', lock['url'], str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', lock['commit']], check=True)
        if method == 'ingp':
            submodules(repo)
        for patch in patch_files(method):
            subprocess.run(['git', '-C', str(repo), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(repo), 'apply', str(patch)], check=True)
        receipt = dict(upstream_commit=lock['commit'], url=lock['url'],
            patches={p.name:sha(p) for p in patch_files(method)}, implementation_files=lock['implementation_files'])
        write(repo/'.odie-vendor.json', receipt)
        verify(root, method)
        print(f'{method}: pinned official implementation verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'vendor')
    parser.add_argument('--methods', nargs='+', required=True,
                        choices=[k for k in read(ROOT/'upstreams.lock.json') if k != 'odie'])
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        for method in args.methods:
            verify(args.root, method)
            print(f'{method}: verified')
    else:
        prepare(args.root, args.methods)


if __name__ == '__main__':
    main()
