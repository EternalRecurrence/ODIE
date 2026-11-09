"""Build the pinned I-NGP engine and the TCNN libraries used by ODIE."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common.io import ROOT
from scripts._common.vendor import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor-root', type=Path, default=ROOT/'vendor')
    args = parser.parse_args()
    from odie import runtime_env
    from odie.runtime_env import cuda_architecture
    repo, _ = verify(args.vendor_root, 'ingp')
    arch = cuda_architecture()
    cmake = os.environ.get('CMAKE', 'cmake')
    subprocess.run([cmake, '-S', str(repo), '-B', str(repo/'build'),
        '-DNGP_BUILD_WITH_GUI=OFF', '-DNGP_BUILD_WITH_OPTIX=OFF',
        '-DNGP_BUILD_WITH_VULKAN=OFF', '-DNGP_BUILD_WITH_PYTHON_BINDINGS=ON',
        f'-DTCNN_CUDA_ARCHITECTURES={arch}',
        f'-DCMAKE_CUDA_COMPILER={Path(os.environ["CUDA_HOME"])/"bin/nvcc"}',
        f'-DCMAKE_CXX_COMPILER={os.environ["CXX"]}',
        f'-DCMAKE_CUDA_HOST_COMPILER={os.environ["CXX"]}',
        f'-DPYTHON_EXECUTABLE={sys.executable}'], check=True)
    subprocess.run([cmake, '--build', str(repo/'build'), '--target', 'pyngp',
        '-j', os.environ.get('MAX_JOBS', '2')], check=True)


if __name__ == '__main__':
    main()
