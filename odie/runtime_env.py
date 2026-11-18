"""Discover the Linux CUDA toolchain without machine-specific defaults."""
import os
from pathlib import Path
import shutil
import sys

if sys.platform != 'linux':
    raise RuntimeError('ODIE native training requires Linux and NVIDIA CUDA')

for key, value in {'MAX_JOBS': '2', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
    os.environ.setdefault(key, value)
if 'CUDA_HOME' not in os.environ:
    nvcc = shutil.which('nvcc')
    if nvcc is None:
        raise RuntimeError('Set CUDA_HOME or put the CUDA toolkit nvcc on PATH')
    os.environ['CUDA_HOME'] = str(Path(nvcc).resolve().parents[1])
for key, executable in [('CC', 'gcc'), ('CXX', 'g++')]:
    os.environ.setdefault(key, shutil.which(executable) or executable)
for directory in (str(Path(os.environ['CUDA_HOME'])/'bin'), str(Path(sys.executable).parent)):
    if directory not in os.environ.get('PATH', '').split(os.pathsep):
        os.environ['PATH'] = directory + os.pathsep + os.environ.get('PATH', '')


def cuda_architecture():
    requested = os.environ.get('ODIE_CUDA_ARCHITECTURE')
    if requested:
        if not requested.isdigit():
            raise ValueError('ODIE_CUDA_ARCHITECTURE must be an integer such as 86')
        return requested
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f'{major}{minor}'
