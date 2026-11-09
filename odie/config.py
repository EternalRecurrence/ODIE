"""Resolve user-facing configuration into a complete, auditable run plan.

This module needs only the Python standard library, including for --dry-run.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
import struct
from pathlib import Path

from .schedule import validate_lr_schedule


DEFAULT = {
    'model': {'layers': 16, 'base_resolution': 8, 'max_resolution': 8192,
              'features': 2, 'finest_features': 2, 'router_unscaled_levels': 0,
              'router_activation': 'sigmoid', 'router_enabled': True, 'on_demand': True,
              'resolution_rounding': 'round', 'budget_remainder': 'finest'},
    'data': {'target_storage': 'uint8', 'sampling': 'stratified', 'chunk_pixels': 4194304},
    'training': {'backend': 'native', 'epochs': 50, 'epochs_per_stage': 1,
                 'batch_size': 65536, 'seed': 1729,
                 'learning_rates': {'grid': .01, 'router': .001, 'decoder': .001},
                 'lr_schedule': {'type': 'late_cosine', 'constant_through_epoch': 30,
                                 'minimum_factor': .1}},
    'regions': {'selection': 'error_per_parameter', 'backend': 'cuda_device'},
    'budget': {'png_fraction': .5},
    'evaluation': {'interval': 5, 'batch_size': 131072, 'save_npy': False,
                   'ssim': 'off', 'ssim_stripe_rows': 256},
}


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    """Reject typos while keeping the example files compact."""
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    # Mutually exclusive budget forms need no default key to survive merging.
    budget = raw.pop('budget', DEFAULT['budget'])
    if set(budget) not in ({'png_fraction'}, {'parameter_payload_bytes'}, {'trainable_parameters'}):
        raise ValueError('Budget needs exactly one of png_fraction, parameter_payload_bytes, trainable_parameters')
    # Explicit lists are optional, rather than additional required settings.
    extra_model = {k: raw.get('model', {}).pop(k) for k in
                   ('resolutions', 'feature_dimensions', 'max_vertices_per_level', 'hash_encoding') if k in raw.get('model', {})}
    config = merge(DEFAULT, raw)
    config['budget'] = budget
    config['model'].update(extra_model)
    return config


def align(value, multiple=8):
    return (value + multiple - 1) // multiple * multiple


def integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum or "inf"}]')
    return value


def source_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def hash_grid_layout(encoding):
    """TCNN FP32 grid_scale/grid_resolution and eight-entry-aligned capacities."""
    levels = integer(encoding['n_levels'], 'HashGrid n_levels', 2, 20)
    base = integer(encoding['base_resolution'], 'HashGrid base_resolution', 2)
    exponent = integer(encoding['log2_hashmap_size'], 'log2_hashmap_size', 3, 30)
    growth = float32(encoding['per_level_scale'])
    if not math.isfinite(growth) or growth <= 1:
        raise ValueError('HashGrid per_level_scale must be finite and greater than one')
    log_growth = float32(math.log2(growth))
    scales = [float32(float32(float32(2**float32(i*log_growth))*base)-1) for i in range(levels)]
    resolutions = [math.ceil(s)+1 for s in scales]
    return resolutions, [min(align(r*r), 1 << exponent) for r in resolutions]


def resolve(config, shape, input_bytes):
    m, t, d, e = (config[k] for k in ('model', 'training', 'data', 'evaluation'))
    on_demand = m['on_demand']
    if type(on_demand) is not bool:
        raise ValueError('on_demand must be bool')
    if not on_demand:
        encoding = m['hash_encoding']
        if encoding['otype'] != 'HashGrid':
            raise ValueError('The no-on-demand ablation requires HashGrid')
        resolutions, hash_caps = hash_grid_layout(encoding)
        m = dict(m, layers=encoding['n_levels'], base_resolution=resolutions[0],
                 max_resolution=resolutions[-1], resolutions=resolutions,
                 feature_dimensions=[encoding['n_features_per_level']]*encoding['n_levels'])
    elif 'hash_encoding' in m:
        raise ValueError('hash_encoding requires on_demand=false')
    levels = integer(m['layers'], 'layers', 2, 20)
    base = integer(m['base_resolution'], 'base_resolution', 2)
    maximum = max(shape) // 2 if m['max_resolution'] == 'half_long_edge' else m['max_resolution']
    maximum = integer(maximum, 'max_resolution', base)
    if m['resolution_rounding'] not in ('floor','round') or m['budget_remainder'] not in ('unused','finest'):
        raise ValueError('Invalid resolution_rounding or budget_remainder')
    # Floor-mode resolutions use floor(exp(log(base)+i*log(max/base)/(L-1))).
    resolutions = m.get('resolutions', [int(math.exp(math.log(base) + i / (levels-1) * math.log(maximum/base)))
                                        for i in range(levels)])
    if 'resolutions' not in m and m['resolution_rounding']=='round':
        resolutions=[round(base*(maximum/base)**(i/(levels-1))) for i in range(levels)]
    if 'resolutions' not in m:
        resolutions[0], resolutions[-1] = base, maximum
    if len(resolutions) != levels or any(type(r) is not int or r < 2 or align(r*r) > 2**31-1 for r in resolutions):
        raise ValueError('resolutions must contain one valid native resolution per level')
    if resolutions != sorted(resolutions):
        raise ValueError('resolutions must be nondecreasing')
    dims = m.get('feature_dimensions', [m['features']]*(levels-1) + [m['finest_features']])
    if len(dims) != levels or any(type(f) is not int or f not in (1, 2, 4) for f in dims):
        raise ValueError('feature_dimensions must contain F1/F2/F4 per level')
    k = integer(m['router_unscaled_levels'], 'router_unscaled_levels', 0, levels)
    if type(m['router_enabled']) is not bool:
        raise ValueError('router_enabled must be true or false')
    if m['router_activation'] not in ('sigmoid', 'softplus'):
        raise ValueError('router_activation must be sigmoid or softplus')
    epochs = integer(t['epochs'], 'epochs')
    batch = integer(t['batch_size'], 'batch_size', 1, 65536)
    integer(t['seed'], 'seed', 0, 2**31-1)
    if t['backend'] not in ('native', 'reference'):
        raise ValueError('backend must be native or reference')
    if d['target_storage'] not in ('uint8', 'float32') or d['sampling'] not in ('global', 'chunked', 'global_full', 'stratified'):
        raise ValueError('Invalid target storage or sampling')
    if d['sampling'] == 'stratified' and (batch & (batch - 1) or math.isqrt(batch) ** 2 != batch):
        raise ValueError('Stratified sampling requires a square power-of-two batch size')
    integer(d['chunk_pixels'], 'chunk_pixels')
    if d['sampling'] == 'chunked' and d['chunk_pixels'] % batch:
        raise ValueError('chunk_pixels must be a multiple of batch_size')
    integer(e['interval'], 'evaluation.interval')
    integer(e['batch_size'], 'evaluation.batch_size')
    integer(e['ssim_stripe_rows'], 'ssim_stripe_rows')
    if type(e['save_npy']) is not bool or e['ssim'] not in ('off', 'raw', 'png'):
        raise ValueError('save_npy must be bool; ssim must be off, raw or png')
    policy = {'error_per_parameter': 'B2', 'error': 'E', 'mean_error': 'B1'}.get(config['regions']['selection'])
    if policy is None:
        raise ValueError('Unknown region selection policy')
    region_backend = config['regions'].get('backend', 'cpu')
    if region_backend not in ('cpu', 'cuda_device'):
        raise ValueError('Region backend must be cpu or cuda_device')
    if region_backend == 'cuda_device' and policy != 'B2':
        raise ValueError('CUDA device residual selection currently supports error_per_parameter only')
    budget = config['budget']
    if 'trainable_parameters' in budget:
        target = integer(budget['trainable_parameters'], 'trainable_parameters')
        payload = target * 2
    elif 'parameter_payload_bytes' in budget:
        payload = integer(budget['parameter_payload_bytes'], 'parameter_payload_bytes')
        target = payload // 2
    else:
        fraction = budget['png_fraction']
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('png_fraction must be in (0, 1]')
        payload = int(input_bytes * fraction)
        target = payload // 2
    width = align(sum(dims), 16)
    gate = width*32 + 32*align(levels, 16) if m['router_enabled'] else 0
    decoder = width*64 + 64*64 + 64*16
    full = [align(r*r) for r in resolutions]
    if target <= gate + decoder:
        raise ValueError('Budget cannot hold the router and decoder')
    # Largest common vertex cap C: saturated dense levels release their unused share.
    low, high = 0, max(full)//8
    user_cap = m.get('max_vertices_per_level')
    if user_cap is not None:
        integer(user_cap, 'max_vertices_per_level', 8)
        high = min(high, user_cap//8)
    while low < high:
        middle = (low + high + 1)//2
        cost = gate + decoder + sum(min(n, middle*8)*f for n, f in zip(full, dims))
        if cost <= target:
            low = middle
        else:
            high = middle - 1
    caps = [min(n, low*8) for n in full]
    if m['budget_remainder']=='finest':
        remainder=target-gate-decoder-sum(n*f for n,f in zip(caps,dims))
        # Every dataset uses the same aligned allocation. A sub-quantum
        # remainder or a completely full finest level stays unallocated.
        limit=min(full[-1],user_cap//8*8) if user_cap is not None else full[-1]
        caps[-1]+=min(remainder//(8*dims[-1])*8,limit-caps[-1])
    if not on_demand:
        caps = hash_caps
        if gate + decoder + sum(n*f for n,f in zip(caps,dims)) != target:
            raise ValueError('HashGrid and heads must exactly match the trainable parameter budget')
    if gate + decoder + sum(n*f for n, f in zip(caps, dims)) > 2**31-1:
        raise ValueError('Native packed parameter offsets must fit int32')
    dense = ([i for i, (cap, n) in enumerate(zip(caps, full)) if cap == n]
             if on_demand else list(range(levels)))
    if dense != list(range(len(dense))) or len(dense) < 2 or min(caps) < 8:
        raise ValueError('Budget must support at least two initial fully covered levels')
    stage = t['epochs_per_stage']
    if type(stage) is not int and not isinstance(stage, list):
        raise ValueError('epochs_per_stage must be an integer or list')
    stages = ([integer(stage, 'epochs_per_stage')] * (levels-len(dense)) if type(stage) is int else list(stage))
    if len(stages) != levels-len(dense) or any(type(x) is not int or x <= 0 for x in stages):
        raise ValueError('epochs_per_stage list must contain one positive epoch count per sparse level')
    unlock = sum(stages)
    if unlock >= epochs:
        raise ValueError('Training must contain at least one all-level joint epoch')
    rates = t['learning_rates']
    out = dict(representation='xy_planes', n_levels=levels, base_resolution=base,
               max_resolution=maximum, resolutions=resolutions, plane_feature_dimensions=dims,
               feature_dim=m['features'], feature_init_std=.001, coarse_levels=k, gate_coarse_levels_K=k,
               gate_topology='shared', gate_position='feature', router_output_activation=m['router_activation'],
               router_enabled=m['router_enabled'],
               gate_hidden_width=32, gate_hidden_layers=1, decoder_hidden_width=64, decoder_hidden_layers=2,
               gate_parameters=gate, decoder_parameters=decoder, final_epochs=epochs,
               stage_epochs=stages, initial_active_prefix=len(dense), unlock_epochs=unlock,
               joint_epochs=epochs-unlock, batch_size=batch, seed=t['seed'],
               grid_lr=rates['grid'], gate_lr=rates['router'], decoder_lr=rates['decoder'],
               learning_rate_schedule=dict(t['lr_schedule'], final_epoch=epochs),
               eval_interval=e['interval'], eval_batch_size=e['batch_size'],
               cell_caps=caps, full_cover_coarse_layers=dense, dynamic_policy=policy,
               selection_ranking=config['regions']['selection'], region_backend=region_backend, support_policy='online current-prefix RGB residual',
               parameter_budget=target, fp16_parameter_budget=payload,
               allocated_parameters=gate+decoder+sum(n*f for n, f in zip(caps, dims)),
               target_storage=d['target_storage'], sampling=d['sampling'], chunk_pixels=d['chunk_pixels'],
               backend=t['backend'], save_npy=e['save_npy'], ssim=e['ssim'], ssim_stripe_rows=e['ssim_stripe_rows'])
    validate_lr_schedule(out)
    if not on_demand:
        out.update(representation='hash_grid', on_demand=False, hash_encoding=deepcopy(encoding),
                   dynamic_policy=None, region_backend=None, selection_ranking=None,
                   support_policy='full spatial coverage; official TCNN HashGrid')
    return out
