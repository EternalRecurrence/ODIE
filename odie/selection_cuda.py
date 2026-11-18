"""Build and bind the exact CUDA greedy support selector."""
import ctypes as C
from hashlib import sha256
import os
from pathlib import Path
import subprocess



def build():
    import fcntl
    from . import runtime_env
    from .runtime_env import cuda_architecture
    source = Path(__file__).with_name('native')/'selection_cuda.cu'
    arch = cuda_architecture()
    digest = sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(os.environ.get('ODIE_BUILD_CACHE', str(Path.home()/'.cache/odie')))/f'selection-{digest}-sm{arch}'
    cache.mkdir(parents=True, exist_ok=True)
    library = cache/'libodie_selection.so'
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            subprocess.run([str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'), '-shared', '-O3',
                '--fmad=false', f'-arch=sm_{arch}', '-Xcompiler', '-fPIC',
                '-ccbin', os.environ['CXX'], str(source), '-o', str(library)], check=True)
    return library


class Cuda:
    def __init__(self, library):
        self.rt = C.CDLL(str(Path(os.environ['CUDA_HOME'])/'lib64/libcudart.so'))
        self.lib = C.CDLL(str(library))
        self.rt.cudaGetErrorString.argtypes = [C.c_int]
        self.rt.cudaGetErrorString.restype = C.c_char_p
        self.rt.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        self.rt.cudaFree.argtypes = [C.c_void_p]
        self.rt.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        self.rt.cudaEventCreate.argtypes = [C.POINTER(C.c_void_p)]
        self.rt.cudaEventRecord.argtypes = [C.c_void_p, C.c_void_p]
        self.rt.cudaEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]
        self.rt.cudaEventDestroy.argtypes = [C.c_void_p]
        ptr, integer = C.c_void_p, C.c_int
        self.lib.selection_init.argtypes = [ptr, integer, integer, ptr, ptr, ptr, ptr, ptr,
                                           ptr, ptr, integer, C.c_uint64]
        self.lib.selection_chunk.argtypes = [ptr, integer, integer, integer, integer, C.c_bool,
                                            ptr, ptr, ptr, ptr, ptr, ptr, integer, integer, ptr, ptr, C.c_uint64]

    def check(self, result):
        if result:
            raise RuntimeError(self.rt.cudaGetErrorString(result).decode())

