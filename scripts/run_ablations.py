"""Tokyo without router, plus DIV2K10 with 8/12/20 experts, strictly serial.

The 16-expert comparison reuses the DIV2K main-experiment results.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._common.io import ROOT,read,sha,write,now
from scripts._common.run import reference
from odie.config import DEFAULT,merge,resolve

IDS=('0028','0036','0063','0188','0337','0374','0394','0460','0545','0767')


def configuration(request):
    overrides={k:deepcopy(v) for k,v in request.items() if k not in ('experiment_group','capacity','ablation')}
    extra={k:overrides.get('model',{}).pop(k) for k in
        ('resolutions','feature_dimensions','max_vertices_per_level') if k in overrides.get('model',{})}
    config=merge(DEFAULT,overrides)
    config['model'].update(extra)
    return config


def plan(root,div2k,ingp_root,tokyo,tokyo_ref):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    jobs=[]
    def add(name,image,request,ref_path):
        with Image.open(image) as picture:
            shape=[picture.height,picture.width]
        digest=sha(image)
        ref=reference(ref_path,digest,shape,request['experiment_group'])
        config=configuration(request)
        config['budget']={'trainable_parameters':ref['parameter_count']}
        resolved=resolve(config,shape,image.stat().st_size)
        jobs.append(dict(name=name,input=str(image.resolve()),input_sha256=digest,
            ingp_run=str(ref_path.resolve()),ingp_parameters=ref['parameter_count'],config=request,
            resolved=resolved,output=str(root/'runs'/name)))
    request=read(ROOT/'configs/ablations/tokyo_no_router.json')
    digest=sha(tokyo)
    if digest!='c6c9f2288d8506998b2cf098f10d81b3e118703b9ecd27d4c075f2ec822fc16b':
        raise ValueError('Router ablation is only for the original Tokyo image')
    add('Tokyo_no_router',tokyo,request,tokyo_ref)
    for levels in (8,12,20):
        for image_id in IDS:
            add(f'DIV2K_{image_id}_L{levels}',div2k/(image_id+'.png'),
                read(ROOT/f'configs/ablations/div2k_L{levels}.json'),ingp_root/image_id/'ingp')
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--div2k-ingp-root',type=Path,required=True,help='Contains IMAGE_ID/ingp/capacity.json')
    parser.add_argument('--tokyo-ingp-run',type=Path,required=True)
    parser.add_argument('--div2k-dir',type=Path,required=True)
    parser.add_argument('--tokyo-input',type=Path,required=True)
    parser.add_argument('--vendor-root',type=Path,default=ROOT/'vendor')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Use a new or empty ablation output: {root}')
    jobs=plan(root,args.div2k_dir,args.div2k_ingp_root,args.tokyo_input,args.tokyo_ingp_run)
    preview=[dict(name=j['name'],reference_parameters=j['ingp_parameters'],
        allocated_parameters=j['resolved']['allocated_parameters'],
        levels=j['resolved']['n_levels'],router=j['resolved']['router_enabled'],
        unlock_epochs=j['resolved']['unlock_epochs']) for j in jobs]
    print(json.dumps(preview,indent=2),flush=True)
    if args.dry_run:
        return
    if sys.platform!='linux':
        parser.error('Ablation training requires Linux with NVIDIA CUDA')
    write(root/'ablation_plan.json',dict(created_at=now(),jobs=jobs,
        reporting_checkpoint='latest',router_scope='Tokyo without router only',
        parameter_policy='same per-image native-verified INGP budget; router savings allocated to grids'))
    rows=[]
    for job in jobs:
        path=root/'configs'/(job['name']+'.json')
        write(path,job['config'])
        rows.append(dict(method='odie',scene='ablation',input=job['input'],config=str(path),
            output=job['output'],ingp_run=job['ingp_run'],python=sys.executable))
    write(root/'manifest.json',rows)
    # The serial entry forwards signals, shares the main GPU lock and verifies
    # cache cleanup before starting the next experiment.
    os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/run_baselines.py'),
        '--manifest',str(root/'manifest.json'),'--vendor-root',str(args.vendor_root)])


if __name__=='__main__':
    main()
