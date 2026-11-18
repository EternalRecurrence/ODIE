"""Build one canonical native source, cached by model/storage specialization."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


def native_library(config):
    if os.name != 'posix':
        raise RuntimeError('Native training currently requires Linux with CUDA and TCNN')
    import fcntl
    from .runtime_env import cuda_architecture
    arch = cuda_architecture()
    source = Path(__file__).resolve().parent / 'native'
    user_cmake = Path.home()/'.local/bin/cmake'
    cmake = os.environ.get('CMAKE', str(user_cmake) if user_cmake.is_file() else 'cmake')
    ingp=Path(__file__).resolve().parents[1]/'vendor'/'ingp'
    tcnn = Path(os.environ.get('TCNN_SOURCE_DIR', str(ingp/'dependencies/tiny-cuda-nn')))
    prebuilt = Path(os.environ.get('TCNN_PREBUILT_DIR', str(ingp/'build/dependencies/tiny-cuda-nn')))
    dependencies = [tcnn/'include/tiny-cuda-nn/optimizers/adam.h',
                    prebuilt/'libtiny-cuda-nn.a', prebuilt/'libtiny-cuda-nn-resources.a',
                    prebuilt/'dependencies/fmt/libfmt.a']
    if any(not p.is_file() for p in dependencies):
        raise FileNotFoundError('Set TCNN_SOURCE_DIR and TCNN_PREBUILT_DIR to the installed TCNN source and static build')
    definitions = dict(ODIE_LEVELS=config['n_levels'], ODIE_RAW_WIDTH=sum(config['plane_feature_dimensions']),
                       ODIE_ROUTER_ENABLED=int(config.get('router_enabled', True)),
                       ODIE_RGB8=int(config['target_storage'] == 'uint8'),
                       ODIE_GATE_K=config['gate_coarse_levels_K'],
                       ODIE_SOFTPLUS=int(config['router_output_activation'] == 'softplus'))
    manifest = dict(definitions=definitions, source={p.name: sha256(p.read_bytes()).hexdigest()
                    for p in sorted(source.iterdir()) if p.is_file()},
                    external={str(p): sha256(p.read_bytes()).hexdigest() for p in dependencies},
                    architecture=arch,
                    compiler={k: os.environ.get(k) for k in ('CC', 'CXX', 'CUDA_HOME')})
    digest = sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    cache = Path(os.environ.get('ODIE_BUILD_CACHE', str(Path.home()/'.cache/odie')))/digest
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/'libodie_runtime_probe.so'
    with (cache/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            command = [cmake, '-S', str(source), '-B', str(cache),
                       f'-DCMAKE_CUDA_ARCHITECTURES={arch}', f'-DTCNN_MIN_GPU_ARCH={arch}',
                       f'-DTCNN_SOURCE_DIR={tcnn}', f'-DTCNN_PREBUILT_DIR={prebuilt}',
                       f'-DCMAKE_CUDA_COMPILER={os.environ["CUDA_HOME"]}/bin/nvcc',
                       f'-DCMAKE_CXX_COMPILER={os.environ["CXX"]}',
                       f'-DCMAKE_CUDA_HOST_COMPILER={os.environ["CXX"]}',
                       *[f'-D{k}={v}' for k, v in definitions.items()]]
            with (cache/'build.log').open('w') as log:
                for argv in (command, [cmake, '--build', str(cache), '-j', os.environ.get('MAX_JOBS', '2')]):
                    result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'Native build failed; see {cache / "build.log"}')
            (cache/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return library
