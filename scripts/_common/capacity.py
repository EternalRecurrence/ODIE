"""Parameter-only capacity allocation. No PSNR or reconstruction selection."""
from copy import deepcopy
import math
import re

def budget(spec, input_bytes, bytes_per_parameter):
    if spec.startswith('png:'):
        fraction = float(spec[4:])
        if not 0 < fraction <= 1:
            raise ValueError('PNG budget fraction must be in (0,1]')
        amount = int(input_bytes*fraction)
        return amount//bytes_per_parameter, amount
    if spec.startswith('params:'):
        count = int(spec[7:])
        if count<=0:
            raise ValueError('Parameter budget must be positive')
        return count,count*bytes_per_parameter
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(B|KB|MB|GB|KiB|MiB|GiB)',spec)
    if not match:
        raise ValueError('Capacity must be e.g. 100MB, 50MiB, png:0.5, or params:1000000')
    unit = {'B':1,'KB':1000,'MB':1000**2,'GB':1000**3,'KiB':1024,'MiB':1024**2,'GiB':1024**3}[match[2]]
    amount = int(float(match[1])*unit)
    if amount<=0:
        raise ValueError('Capacity must be positive')
    return amount//bytes_per_parameter,amount


def maximal_width(target, count):
    low,high=0,math.isqrt(target)+1
    while low<high:
        mid=(low+high+1)//2
        if count(mid)<=target:
            low=mid
        else:
            high=mid-1
    if low<1:
        raise ValueError('Capacity cannot hold this model')
    return low,count(low)


def mlp(target, hidden_layers):
    width,count=maximal_width(target,lambda w:hidden_layers*w*w+(hidden_layers+6)*w+3)
    return dict(width=width,parameter_count=count)


def neural_experts(target):
    width,count=maximal_width(target,lambda w:8*w*w+536*w+166288)
    return dict(expert_width=width,parameter_count=count)


def grid_count(base, exponent, finest, levels=16, features=2):
    scale=(finest/base)**(1/(levels-1))
    nodes=sum(min(((math.ceil(base*scale**i-1)+1)**2+7)//8*8,1<<exponent) for i in range(levels))
    return nodes*features,scale


def hash_grid(target, finest, overhead, levels=16, features=2, max_base=128):
    chosen=None
    for exponent in range(1,25):
        count,_=grid_count(1,exponent,finest,levels,features)
        if count+overhead<=target:
            chosen=exponent
    if chosen is None:
        raise ValueError('Capacity cannot hold hash model')
    low,high=1,min(max_base,int(finest))
    while low<high:
        mid=(low+high+1)//2
        count,_=grid_count(mid,chosen,finest,levels,features)
        if count+overhead<=target:
            low=mid
        else:
            high=mid-1
    count,scale=grid_count(low,chosen,finest,levels,features)
    return dict(base_resolution=low,log2_hashmap_size=chosen,per_level_scale=scale,
                parameter_count=count+overhead,capacity_verified=False)


def ingp_div2k(target,finest):
    candidates=[]
    # Search hash exponents 15..20 and base resolutions 8..80 for the largest
    # parameter count within the target; ties prefer bases nearest 16.
    for exponent in range(15,21):
        low,high=8,80
        while low<=high:
            base=(low+high)//2
            count,scale=grid_count(base,exponent,finest)
            count+=7168
            if count<=target:
                candidates.append(dict(base_resolution=base,log2_hashmap_size=exponent,
                    per_level_scale=scale,parameter_count=count,capacity_verified=False))
                low=base+1
            else:
                high=base-1
    if not candidates:
        raise ValueError('Capacity is below the pinned DIV2K search domain')
    return max(candidates,key=lambda r:(r['parameter_count'],-abs(r['base_resolution']-16)))


def miner(target, shape, defaults, scales, full_allocation_factor):
    # Only MINER has residual-dependent block allocation. This allowance bounds
    # full coverage, not the unknown fraction of blocks retained after training.
    if not math.isfinite(full_allocation_factor) or full_allocation_factor < 1:
        raise ValueError('MINER full_allocation_factor must be finite and at least 1')
    limit=math.floor(target*full_allocation_factor)
    f=defaults['NETWORK']['nfeat']
    depth=defaults['NETWORK']['nlayers']
    costs=[depth*w*w+(depth+6)*w+3 for w in [f*4]+[f]*(scales-1)]
    factor=2**(scales-1)
    def upper(k):
        coarse=math.ceil(shape[0]/(factor*k))*math.ceil(shape[1]/(factor*k))
        return sum(coarse*4**i*c for i,c in enumerate(costs))
    low,high=1,max(shape)
    if upper(high)>limit:
        raise ValueError('Capacity cannot hold even one MINER coarse block')
    while low<high:
        mid=(low+high)//2
        if upper(mid)<=limit:
            high=mid
        else:
            low=mid+1
    return dict(ksize=low,stride=low,parameter_upper_bound=upper(low),
                target_parameters=target,full_allocation_factor=full_allocation_factor,
                full_allocation_limit=limit,capacity_verified=False,
                policy='smallest block size within full-allocation allowance; report actual count after training')


def odie(target, model):
    from copy import deepcopy
    from odie.config import DEFAULT,resolve
    config=deepcopy(DEFAULT)
    config['model'].update(model)
    config['budget']={'trainable_parameters':target}
    result=resolve(config,(model['max_resolution']*2,)*2,0)
    full=[(r*r+7)//8*8 for r in result['resolutions']]
    return dict(resolutions=result['resolutions'],plane_dims=result['plane_feature_dimensions'],
        plane_node_caps=result['cell_caps'],native_full_node_slots=full,
        full_cover_levels=result['full_cover_coarse_layers'],initial_active_prefix=result['initial_active_prefix'],
        parameter_count=result['allocated_parameters'],gate_parameters=result['gate_parameters'],
        decoder_parameters=result['decoder_parameters'])
