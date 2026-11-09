"""Lossless layerwise support storage; legacy bitmap checkpoints remain readable."""
from functools import lru_cache
from hashlib import sha256
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings
import zlib

import numpy as np
import torch

STORAGE_VERSION = 'layerwise-bitmap-deflate-jbig-v1'


@lru_cache(maxsize=1)
def jbig_tool():
    """Build the optional CPU bridge once, outside the CUDA build cache."""
    compiler = shutil.which('g++') if os.name == 'posix' else None
    if compiler is None:
        warnings.warn('JBIG unavailable: g++ and libjbig headers/library are required; exporting bitmap/DEFLATE masks.')
        return None
    source = Path(__file__).with_name('support_jbig.cpp')
    digest = sha256(source.read_bytes() + compiler.encode()).hexdigest()[:20]
    cache = Path(os.environ.get('ODIE_BUILD_CACHE', str(Path.home()/'.cache/odie')))/('jbig-'+digest)
    cache.mkdir(parents=True, exist_ok=True)
    tool = cache/'jbig_codec'
    if not tool.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            candidate = Path(temporary)/'jbig_codec'
            result = subprocess.run([compiler, '-O2', '-std=c++11', str(source), '-ljbig', '-o', str(candidate)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                warnings.warn('JBIG unavailable; exporting bitmap/DEFLATE masks. Compiler: '+result.stderr.decode(errors='replace')[-1000:])
                return None
            os.replace(candidate, tool)
    return tool


def _jbig(operation, resolution, data, tool):
    result = subprocess.run([str(tool), operation, str(resolution), str(resolution)], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError('JBIG '+operation+' failed: '+result.stderr.decode(errors='replace'))
    return result.stdout


def encode_support(support):
    if support.device.type != 'cpu' or support.dtype != torch.bool or support.ndim != 1 or support.numel() < 1:
        raise ValueError('Support must be a nonempty flat CPU bool tensor')
    values = support.numpy()
    total, count = values.size, int(values.sum())
    resolution = math.isqrt(total)
    if resolution**2 != total:
        raise ValueError('Support must describe a square grid')
    record = dict(num_cells=total, resolution=resolution, selected_count=count)
    if count in (0, total):
        return dict(record, encoding='all' if count else 'none')
    bitmap = np.packbits(values, bitorder='little').tobytes()
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    candidates = {'bitmap': bitmap, 'bitmap_deflate': compressor.compress(bitmap)+compressor.flush()}
    tool = jbig_tool()
    if tool is not None:
        rows = np.packbits(values.reshape(resolution, resolution), axis=1, bitorder='big').tobytes()
        candidates['jbig1'] = _jbig('encode', resolution, rows, tool)
    # Raw bitmap comes first, so ties keep the simplest representation.
    encoding, data = min(candidates.items(), key=lambda item: len(item[1]))
    record.update(encoding=encoding, data=torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()))
    if not torch.equal(decode_support(record), support):
        raise AssertionError('Selected mask codec changed support membership')
    return record


def decode_support(record):
    total, resolution, count = record['num_cells'], record['resolution'], record['selected_count']
    if (type(total) is not int or total < 1 or type(resolution) is not int or resolution < 1 or resolution**2 != total
            or type(count) is not int or not 0 <= count <= total):
        raise ValueError('Invalid support geometry or selected count')
    encoding = record['encoding']
    if encoding in ('all', 'none'):
        if count != (total if encoding == 'all' else 0):
            raise ValueError('Constant mask count mismatch')
        return torch.full((total,), encoding == 'all', dtype=torch.bool)
    payload = record['data']
    if not isinstance(payload, torch.Tensor) or payload.dtype != torch.uint8 or payload.ndim != 1:
        raise ValueError('Mask payload must be a flat uint8 tensor')
    data = payload.detach().cpu().numpy().tobytes()
    if encoding == 'jbig1':
        tool = jbig_tool()
        if tool is None:
            raise RuntimeError('Reading this JBIG checkpoint requires g++ and libjbig headers/library')
        rows = _jbig('decode', resolution, data, tool)
        if len(rows) != resolution*((resolution+7)//8):
            raise ValueError('Decoded JBIG size mismatch')
        packed = np.frombuffer(rows, dtype=np.uint8).reshape(resolution, -1)
        if resolution % 8 and bool((packed[:, -1] & ((1 << (8-resolution % 8))-1)).any()):
            raise ValueError('JBIG row padding must be zero')
        values = np.unpackbits(packed, axis=1, bitorder='big', count=resolution).reshape(-1)
    elif encoding in ('bitmap', 'bitmap_deflate'):
        if encoding == 'bitmap_deflate':
            decoder = zlib.decompressobj(-15)
            try:
                data = decoder.decompress(data, (total+7)//8+1)
            except zlib.error as error:
                raise ValueError('Invalid DEFLATE mask') from error
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('Incomplete or trailing DEFLATE data')
        if len(data) != (total+7)//8 or (total % 8 and data[-1] >> (total % 8)):
            raise ValueError('Bitmap length or padding mismatch')
        values = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little', count=total)
    else:
        raise ValueError('Unknown support encoding: '+str(encoding))
    if int(values.sum()) != count:
        raise ValueError('Decoded selected count mismatch')
    return torch.from_numpy(values.astype(bool))


def encode_supports(supports):
    return [encode_support(support) for support in supports]


def decode_supports(artifact):
    counts = artifact['support_num_cells']
    if 'support_encodings' in artifact:
        if artifact.get('support_storage_version') != STORAGE_VERSION:
            raise ValueError('Unsupported support storage version')
        if len(artifact['support_encodings']) != len(counts):
            raise ValueError('Encoded layer count mismatch')
        supports = [decode_support(record) for record in artifact['support_encodings']]
    else:
        bitmaps = artifact['support_bitmaps']
        if len(bitmaps) != len(counts):
            raise ValueError('Legacy bitmap layer count mismatch')
        supports = []
        for bits, count in zip(bitmaps, counts):
            if (type(count) is not int or count < 1 or bits.dtype != torch.uint8 or bits.ndim != 1
                    or bits.numel() != (count+7)//8):
                raise ValueError('Invalid legacy bitmap')
            supports.append(torch.from_numpy(np.unpackbits(bits.detach().cpu().numpy(), bitorder='little', count=count).astype(bool)))
    if [support.numel() for support in supports] != counts:
        raise ValueError('Support sizes differ from checkpoint geometry')
    resolutions = artifact['configuration'].get('resolutions')
    expected = [] if artifact['configuration']['representation']=='hash_grid' else [r*r for r in resolutions] if resolutions is not None else None
    if expected is not None and counts != expected:
        raise ValueError('Support sizes differ from model configuration')
    return supports


def restore_support_record(artifact):
    return dict(artifact['support_payload'], supports=decode_supports(artifact))


def support_storage_statistics(encodings):
    layers = [dict(level=i, resolution=record['resolution'], selected_count=record['selected_count'],
                   encoding=record['encoding'], bitmap_bytes=(record['num_cells']+7)//8,
                   payload_bytes=record['data'].numel() if 'data' in record else 0)
              for i, record in enumerate(encodings)]
    return dict(version=STORAGE_VERSION, selection_rule='smallest available serialized stream; raw bitmap wins ties',
                layers=layers, bitmap_bytes=sum(row['bitmap_bytes'] for row in layers),
                payload_bytes=sum(row['payload_bytes'] for row in layers))


def compress_checkpoint(source, destination):
    """Repack existing weights and metadata without overwriting the original."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    artifact = torch.load(source, map_location='cpu', weights_only=True)
    supports = decode_supports(artifact)
    encodings = encode_supports(supports)
    compact = {key: value for key, value in artifact.items() if key not in ('support_bitmaps', 'support_encodings', 'support_storage_version')}
    compact.update(support_storage_version=STORAGE_VERSION, support_encodings=encodings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(compact, stream)
    restored = torch.load(destination, map_location='cpu', weights_only=True)
    if not all(torch.equal(a, b) for a, b in zip(supports, decode_supports(restored))):
        raise AssertionError('Repacked checkpoint changed supports')
    if not all(torch.equal(value.contiguous().view(torch.uint8), restored['parameters'][name].contiguous().view(torch.uint8))
               for name, value in artifact['parameters'].items()):
        raise AssertionError('Repacked checkpoint changed weights')
    return dict(source=str(source), output=str(destination), source_bytes=source.stat().st_size,
                checkpoint_bytes=destination.stat().st_size,
                fp16_parameter_payload_bytes=sum(p.numel()*p.element_size() for p in restored['parameters'].values()),
                support_storage=support_storage_statistics(encodings), roundtrip_verified=True)
