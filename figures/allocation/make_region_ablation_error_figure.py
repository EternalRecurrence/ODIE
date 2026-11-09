"""Render aligned region reconstructions and shared-scale RGB error maps.

Run: python figures/allocation/make_region_ablation_error_figure.py --config figure.json
Requires numpy and Pillow. Config paths are relative to the config file.
Each source accepts either an original-resolution `image` or a `crops` mapping.
Optional `support` supplies per-region NPZ `values`, 1-based `experts`, and a
`label`; their binary union replaces the lower-left legend panel. Also exports
separate error comparisons inside and outside that same support for all models.
Those comparisons are also combined with reconstructions in a three-row figure.
Whole-image PSNR comes exclusively from the supplied full-image metrics JSON;
crop PSNR is measured separately and is never substituted for the whole image.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size, bold=False):
    for name in (f'C:/Windows/Fonts/arial{"bd" if bold else ""}.ttf',
                 f'DejaVuSans{"-Bold" if bold else ""}.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def support_error_comparisons(loaded, region, active, lut, limit, output, panels_dir):
    """Use the Full support as a fixed partition for every reconstruction."""
    tile, gap, left = 384, 16, 20
    width = left+len(loaded)*(tile+gap)-gap+112
    crop = loaded[0]['crops'][region]
    crop_h = round(tile*crop.height/crop.width)
    records = []
    for name, selected in (('inside',active), ('outside',~active)):
        count = int(selected.sum())
        fraction = float(selected.mean())
        canvas = Image.new('RGB',(width,crop_h+170),'white')
        draw = ImageDraw.Draw(canvas)
        region_label = 'Active region' if name=='inside' else 'Inactive region'
        draw.text((left,10),f'{region_label} | '
                  f'{100*fraction:.1f}% of crop',font=get_font(24,True),fill='black')
        draw.text((left,42),'Identical RGB RMSE scale for both methods',
                  font=get_font(19),fill='#444444')
        for col,item in enumerate(loaded):
            x = left+col*(tile+gap)
            draw.text((x+tile/2,88),item['label'].replace('\n',' '),
                      font=get_font(25),fill='black',anchor='ms')
            if col:
                error = np.load(panels_dir/f'{region}-{col:02d}-error.npy')
                pixels = lut[np.rint(np.clip(error/limit,0,1)*255).astype(np.uint8)].copy()
                mse = float(np.mean(error[selected]**2)) if count else None
                rmse = math.sqrt(mse) if mse is not None else None
                psnr = 10*math.log10(255**2/mse) if mse else None
                records.append(dict(region=region,subset=name,label=item['label'],
                    pixels=count,fraction=fraction,mse_rgb8=mse,rmse_rgb8=rmse,
                    psnr_db=psnr,perfect_reconstruction=(mse==0)))
                caption = (f'RMSE: {rmse:.3f} | PSNR: {psnr:.3f} dB' if psnr is not None
                           else 'Perfect reconstruction' if count else 'Empty subset')
            else:
                pixels = np.asarray(crop).copy()
                caption = 'Reference pixels in the selected subset'
            pixels[~selected] = 180
            panel = Image.fromarray(pixels)
            panel.save(panels_dir/f'{region}-{name}-{col:02d}.png')
            canvas.paste(panel.resize((tile,crop_h),Image.Resampling.NEAREST),(x,100))
            draw.text((x+tile/2,100+crop_h+30),caption,font=get_font(19),
                      fill='black',anchor='ms')
        lx,ly,bw,bh = width-96,156,20,crop_h-112
        gradient = np.tile(lut[np.rint(np.linspace(255,0,bh)).astype(int)][:,None],(1,bw,1))
        canvas.paste(Image.fromarray(gradient),(lx,ly))
        draw.text((lx+bw/2,ly-15),'RMSE',font=get_font(18),fill='black',anchor='ms')
        for f in (0,.5,1):
            draw.text((lx+bw+8,ly+f*(bh-1)),f'{limit*(1-f):g}'+('+' if f==0 else ''),
                      font=get_font(18),fill='black',anchor='lm')
        canvas.save(output.with_name(f'{output.stem}-{region}-{name}-error.png'))

    # Compact overview: reconstruction, supported pixels, unsupported pixels.
    row_step = crop_h+90
    combined = Image.new('RGB',(width,120+2*row_step+crop_h+60),'white')
    draw = ImageDraw.Draw(combined)
    for col,item in enumerate(loaded):
        x = left+col*(tile+gap)
        draw.text((x+tile/2,32),item['label'].replace('\n',' '),font=get_font(27),
                  fill='black',anchor='ms')
        if col:
            draw.text((x+tile/2,68),f'Image PSNR: {item["full_psnr_db"]:.3f} dB',
                      font=get_font(22),fill='black',anchor='ms')
        combined.paste(item['crops'][region].resize((tile,crop_h),Image.Resampling.NEAREST),
                       (x,120))
    draw.text((left,108),'Reconstruction',font=get_font(22,True),
              fill='black',anchor='ls')
    for row,(name,selected) in enumerate((('inside',active),('outside',~active)),1):
        y = 120+row*row_step
        region_label = 'Active region' if name=='inside' else 'Inactive region'
        draw.text((left,y-30),f'{region_label} '
                  f'({100*selected.mean():.1f}%)',
                  font=get_font(22,True),fill='black')
        for col,item in enumerate(loaded):
            x = left+col*(tile+gap)
            with Image.open(panels_dir/f'{region}-{name}-{col:02d}.png') as panel:
                combined.paste(panel.resize((tile,crop_h),Image.Resampling.NEAREST),(x,y))
            if col:
                record = next(r for r in records if r['subset']==name and r['label']==item['label'])
                caption = (f'RMSE: {record["rmse_rgb8"]:.3f} | PSNR: {record["psnr_db"]:.3f} dB'
                           if record['psnr_db'] is not None else 'Perfect or empty subset')
            else:
                caption = 'Reference in selected subset'
            draw.text((x+tile/2,y+crop_h+30),caption,font=get_font(19),
                      fill='black',anchor='ms')
    # Shared legend for both error rows; masked-out pixels are never colored.
    lx,ly,bw,bh = width-96,120+row_step+56,20,crop_h-112
    combined.paste(Image.fromarray(gradient),(lx,ly))
    draw.text((lx+bw/2,ly-15),'RMSE',font=get_font(18),fill='black',anchor='ms')
    for f in (0,.5,1):
        draw.text((lx+bw+8,ly+f*(bh-1)),f'{limit*(1-f):g}'+('+' if f==0 else ''),
                  font=get_font(18),fill='black',anchor='lm')
    combined.save(output.with_name(f'{output.stem}-{region}-three-row.png'))
    return records


def generate(config, base):
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else base/path

    limit = float(config.get('error_max', 20))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('error_max must be finite and positive')
    regions = config['regions']
    if not regions or not config['models']:
        raise ValueError('At least one region and model are required')
    for key, box in regions.items():
        if (len(box) != 4 or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
            raise ValueError(f'Invalid xyxy coordinates for {key}: {box}')

    sources = [dict(config['reference'], label='Reference'), *config['models']]
    loaded = []
    for index, item in enumerate(sources):
        crops = {}
        if 'image' in item:
            # Explicit large-image opt-in, with dimensions/crop bounds checked below.
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(resolve(item['image'])) as im:
                for region, box in regions.items():
                    if box[2] > im.width or box[3] > im.height:
                        raise ValueError(f'{region} is outside {item["image"]}')
                    crops[region] = im.crop(box).convert('RGB')
        else:
            for region in regions:
                with Image.open(resolve(item['crops'][region])) as im:
                    if im.mode != 'RGB':
                        raise ValueError('Pre-extracted crops must be RGB8')
                    crops[region] = im.copy()
        for region, box in regions.items():
            if crops[region].size != (box[2]-box[0], box[3]-box[1]):
                raise ValueError(f'{item["label"]}: crop size mismatch in {region}')
        record = {'label':item['label'], 'crops':crops, 'source':item}
        if index:
            metric_path = resolve(item['metrics'])
            metric_bytes = metric_path.read_bytes()
            evaluation = json.loads(metric_bytes)['evaluation']
            psnr = float(evaluation['psnr_db'])
            if not math.isfinite(psnr):
                raise ValueError(f'Invalid whole-image PSNR in {metric_path}')
            record.update(full_psnr_db=psnr, evaluation=evaluation,
                          metric_path=str(metric_path),
                          metric_sha256=hashlib.sha256(metric_bytes).hexdigest())
        loaded.append(record)

    support = config.get('support')
    tile, gap = 384, 16
    left, right = (20,112) if support else (92,20)
    width = left+len(loaded)*(tile+gap)-gap+right
    title_h, band_h, row_gap = 128, 32, 24
    region_h = 2*(tile+band_h)+row_gap
    height = title_h+len(regions)*region_h+20
    canvas = Image.new('RGB', (width,height), 'white')
    draw = ImageDraw.Draw(canvas)
    display_mm = float(config.get('display_width_mm', 178))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_mm}mm" '
           f'height="{display_mm*height/width:.5f}mm" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="white"/>']

    def text(x, y, value, size=26, bold=False, anchor='middle', color='#111111'):
        draw.text((x,y),value,font=get_font(size,bold),fill=color,
                  anchor={'middle':'ms','start':'ls','end':'rs'}[anchor])
        svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                   f'font-family="Arial,DejaVu Sans,sans-serif" font-size="{size}" '
                   f'font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(value)}</text>')

    def put(im, x, y, w=tile, h=tile):
        canvas.paste(im.resize((w,h),Image.Resampling.NEAREST),(x,y))
        buffer = io.BytesIO()
        im.save(buffer,format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
                   f'style="image-rendering:pixelated" href="data:image/png;base64,{encoded}"/>')

    anchors = np.array([[12,7,30],[52,15,93],[111,25,110],[176,44,91],
                        [229,81,57],[251,145,52],[252,209,111],[252,253,191]],float)
    lut = np.stack([np.interp(np.linspace(0,7,256),np.arange(8),anchors[:,c])
                    for c in range(3)],axis=1).astype(np.uint8)
    output = resolve(config['output'])
    if output.suffix.lower() != '.svg':
        raise ValueError('output must end in .svg')
    output.parent.mkdir(parents=True,exist_ok=True)
    panels_dir = output.parent/(output.stem+'-panels')
    panels_dir.mkdir(exist_ok=True)
    manifest = {'regions_xyxy':regions,
                'whole_image_psnr_source':'evaluation.psnr_db in each supplied metrics JSON',
                'error_definition':'Per-pixel sqrt(mean((prediction-reference)^2 over RGB)), RGB8 units',
                'error_limits':[0,limit],
                'error_scaling':'Identical for every region and method; values above maximum saturate',
                'crop_resampling':'Nearest neighbor; no filtering or enhancement',
                'methods':[], 'local_metrics':[], 'support':[], 'support_metrics':[]}
    for col, item in enumerate(loaded):
        x = left+col*(tile+gap)
        lines = item['label'].split('\n')
        for i,line in enumerate(lines):
            text(x+tile/2,34+i*32,line,27,bold=(col==len(loaded)-1))
        if col:
            text(x+tile/2,108,f'Image PSNR: {item["full_psnr_db"]:.3f} dB',24)
            manifest['methods'].append({k:v for k,v in item.items() if k!='crops'})
        else:
            text(x+tile/2,108,'RGB8 reference',23,color='#555555')

    for row,(region,box) in enumerate(regions.items()):
        start = title_h+row*region_h
        crop_y, error_y = start+band_h, start+2*band_h+tile
        crop_h = round(tile*(box[3]-box[1])/(box[2]-box[0]))
        if crop_h > tile:
            raise ValueError('Portrait crops require a wider output layout')
        truth = np.asarray(loaded[0]['crops'][region],dtype=np.float64)
        if not support:
            text(42,crop_y+tile/2,region,32,True)
        text(left,start+24,'Reconstruction' if support else f'Region {region}',
             24,True,anchor='start')
        if not support:
            text(left,error_y-9,'RGB error (RMSE)',24,anchor='start')
        for col,item in enumerate(loaded):
            x = left+col*(tile+gap)
            crop = item['crops'][region]
            put(crop,x,crop_y,tile,crop_h)
            crop.save(panels_dir/f'{region}-{col:02d}-crop.png')
            if col:
                if support:
                    text(x+tile/2,error_y-9,'RGB error (RMSE)',24)
                delta = np.asarray(crop,dtype=np.float64)-truth
                error = np.sqrt(np.mean(delta**2,axis=2))
                heat = Image.fromarray(lut[np.rint(np.clip(error/limit,0,1)*255).astype(np.uint8)])
                put(heat,x,error_y,tile,crop_h)
                heat.save(panels_dir/f'{region}-{col:02d}-error.png')
                np.save(panels_dir/f'{region}-{col:02d}-error.npy',error)
                mse = float(np.mean(delta**2))
                manifest['local_metrics'].append(dict(region=region,label=item['label'],
                    crop_psnr_db=10*math.log10(255**2/mse) if mse else None,
                    crop_mse_rgb8=mse, fraction_above_color_limit=float(np.mean(error>limit)),
                    sha256_pixels=hashlib.sha256(crop.tobytes()).hexdigest()))
            elif support:
                spec = support[region]
                values_path = resolve(spec['values'])
                with np.load(values_path) as values:
                    visible = values['visible'].reshape(crop.height,crop.width,-1)
                experts = spec['experts']
                if (not experts or any(type(e) is not int or not 1 <= e <= visible.shape[-1]
                                       for e in experts)):
                    raise ValueError('support experts must be valid 1-based indices')
                active = visible[...,np.array(experts)-1].astype(bool).any(axis=-1)
                mask = Image.fromarray(active.astype(np.uint8)*255).convert('RGB')
                put(mask,x,error_y,tile,crop_h)
                mask.save(panels_dir/f'{region}-support.png')
                text(x+tile/2,error_y-9,spec['label'],24)
                manifest['support'].append(dict(region=region,source=str(values_path),
                    sha256=hashlib.sha256(values_path.read_bytes()).hexdigest(),
                    experts=experts,definition='Binary union of spatial support; white=active, black=inactive',
                    active_fraction=float(active.mean())))
            else:
                lx,ly,bw,bh = x+40,error_y+tile//2, tile-80,22
                gradient = np.tile(lut[np.rint(np.linspace(0,255,bw)).astype(int)][None],(bh,1,1))
                put(Image.fromarray(gradient),lx,ly,bw,bh)
                text(x+tile/2,ly-24,'RGB RMSE',26)
                for f in (0,.5,1):
                    text(lx+f*bw,ly+bh+32,f'{limit*f:g}'+('+' if f==1 else ''),24)
                text(x+tile/2,ly+bh+75,'Shared scale (RGB8)',22)

        if support:
            manifest['support_metrics'].extend(support_error_comparisons(
                loaded,region,active,lut,limit,output,panels_dir))
            # One common color scale, outside the aligned mask/error panels.
            lx,ly,bw,bh = width-right+16,error_y+56,20,crop_h-112
            gradient = np.tile(lut[np.rint(np.linspace(255,0,bh)).astype(int)][:,None],(1,bw,1))
            put(Image.fromarray(gradient),lx,ly,bw,bh)
            text(lx+bw/2,ly-18,'RMSE',18)
            for f in (0,.5,1):
                text(lx+bw+8,ly+f*(bh-1)+6,
                     f'{limit*(1-f):g}'+('+' if f==0 else ''),18,anchor='start')
            text(lx+bw/2,ly+bh+30,'RGB8',17)

    svg.append('</svg>')
    output.write_text('\n'.join(svg)+'\n',encoding='utf-8')
    canvas.save(output.with_suffix('.png'))
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    # Also save a compact reconstruction-only preview from the exact same panels.
    compact = Image.new('RGB',(width,title_h+len(regions)*(tile+band_h)+20),'white')
    compact.paste(canvas.crop((0,0,width,title_h)),(0,0))
    for row in range(len(regions)):
        y = title_h+row*region_h
        compact.paste(canvas.crop((0,y,width,y+band_h+tile)),(0,title_h+row*(tile+band_h)))
    compact.save(output.with_name(output.stem+'-reconstruction.png'))
    print(output)
    for row in manifest['local_metrics']:
        print(row['region'],row['label'].replace('\n',' '),f"local PSNR {row['crop_psnr_db']:.3f}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    generate(json.loads(args.config.read_text(encoding='utf-8')),args.config.resolve().parent)


if __name__ == '__main__':
    main()
