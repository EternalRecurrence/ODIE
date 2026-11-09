"""Compose full images, aligned crops, and shared-scale RGB error maps.

Requires numpy and Pillow. Run with --help, or use a JSON config whose relative
paths resolve against the config directory. SVG embeds PNG panels, so it has no
external image dependencies; labels, borders, and colorbar remain vector objects.
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


def read_rgb(path):
    with Image.open(path) as im:
        if im.mode not in ('RGB', 'RGBA', 'L', 'P'):
            raise ValueError(f'Expected an 8-bit RGB-compatible image: {path} ({im.mode})')
        return im.convert('RGB')


def font(size, bold=False):
    for name in (('C:/Windows/Fonts/timesbd.ttf' if bold else 'C:/Windows/Fonts/times.ttf'),
                 ('DejaVuSerif-Bold.ttf' if bold else 'DejaVuSerif.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def generate(config, base):
    def resolve(value):
        p = Path(value)
        return (p if p.is_absolute() else base/p).resolve()

    reference_path = resolve(config['reference'])
    reference = read_rgb(reference_path)
    box = tuple(config['crop'])
    w, h = reference.size
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('crop must be four integer coordinates [left, top, right, bottom]')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f'Crop {box} is outside reference image {reference.size}')
    limit = float(config.get('error_max', 20))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('error_max must be finite and positive (RGB8 units)')
    if not config['models']:
        raise ValueError('At least one model image is required')
    panels = [('Reference', reference, reference_path)]
    for item in config['models']:
        path = resolve(item['image'])
        im = read_rgb(path)
        if im.size != reference.size:
            raise ValueError(f'{path}: {im.size} differs from reference {reference.size}')
        panels.append((item['label'], im, path))

    tile, gap, left, right = 256, 12, 140, 10
    display_width = float(config.get('display_width_mm', 171.45))
    font_pt = float(config.get('font_pt', 9))
    if not all(math.isfinite(value) and value > 0 for value in (display_width, font_pt)):
        raise ValueError('display_width_mm and font_pt must be finite and positive')
    width = left+len(panels)*(tile+gap)-gap+right
    label_size = font_pt*width/(display_width*72/25.4)
    line_height = round(label_size*1.12)
    # Wrap at the actual panel width, including epoch suffixes and long names.
    def wrap_label(label):
        lines = []
        for paragraph in label.split('\n'):
            line = ''
            for word in paragraph.split():
                candidate = (line+' '+word).strip()
                if line and font(round(label_size)).getlength(candidate) > tile:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        return lines
    overview_h = round(tile*h/w)
    crop_h = round(tile*(y2-y1)/(x2-x1))
    top = 12+line_height*max(len(wrap_label(label)) for label, _, _ in panels)
    crop_y = top+overview_h+20
    error_y = crop_y+crop_h+20
    height = error_y+crop_h+line_height
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{display_width}mm" height="{display_width*height/width:.5f}mm" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="white"/>']

    def text(x, y, value, size=23, bold=False, color='#111111', anchor='middle'):
        size = label_size
        # Use the same baseline and font family in SVG and PNG preview.
        draw.text((x, y), value, font=font(round(size), bold), fill=color,
                  anchor={'middle':'ms', 'end':'rs', 'start':'ls'}[anchor])
        svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Times New Roman,Times,serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(value)}</text>')

    def put(im, x, y, dw, dh, nearest=False):
        canvas.paste(im.resize((dw, dh), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS), (x, y))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        data = base64.b64encode(buf.getvalue()).decode('ascii')
        svg.append(f'<image x="{x}" y="{y}" width="{dw}" height="{dh}" preserveAspectRatio="none" style="image-rendering:{"pixelated" if nearest else "auto"}" xlink:href="data:image/png;base64,{data}"/>')

    def rect(x, y, rw, rh, color, stroke=1):
        draw.rectangle((x, y, x+rw, y+rh), outline=color, width=stroke)
        svg.append(f'<rect x="{x}" y="{y}" width="{rw}" height="{rh}" fill="none" stroke="{color}" stroke-width="{stroke}"/>')

    anchors = np.array([[12,7,30],[52,15,93],[111,25,110],[176,44,91],
                        [229,81,57],[251,145,52],[252,209,111],[252,253,191]], float)
    lut = np.stack([np.interp(np.linspace(0,7,256), np.arange(8), anchors[:,c]) for c in range(3)], axis=1).astype(np.uint8)
    output = resolve(config['output'])
    if output.suffix.lower() != '.svg':
        raise ValueError('output must end in .svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_dir = output.parent/(output.stem+'-panels')
    crop_dir.mkdir(exist_ok=True)
    target = np.asarray(reference.crop(box), dtype=np.float64)
    manifest = {'reference':str(reference_path), 'crop_xyxy':box, 'image_size':[w,h],
                'error_definition':'sqrt(mean((prediction - reference)^2, RGB channels)), RGB8 units',
                'error_limits':[0,limit], 'clipping':'errors above maximum saturate; no per-panel normalization',
                'crop_resampling':'nearest neighbor; no image enhancement',
                'style': {'display_width_mm': display_width, 'font_pt': font_pt}, 'panels':[]}
    region_color = '#b56b00'
    text(left-10, top+overview_h/2, 'Full', 19, anchor='end')
    text(left-10, crop_y+crop_h/2, 'Crop', 19, anchor='end')
    text(left-10, error_y+crop_h/2, 'Error', 19, anchor='end')
    for j, (label, im, path) in enumerate(panels):
        x = left+j*(tile+gap)
        lines = wrap_label(label)
        for n, line in enumerate(lines):
            text(x+tile/2, label_size+n*line_height, line, 23, label.startswith('Ours'))
        # Embed the reduced overview to keep the portable SVG compact.
        put(im.resize((tile,overview_h), Image.Resampling.LANCZOS),x,top,tile,overview_h)
        rect(x+x1/w*tile, top+y1/h*overview_h, (x2-x1)/w*tile, (y2-y1)/h*overview_h, region_color, 2)
        crop = im.crop(box)
        crop.save(crop_dir/f'{j:02d}-crop.png')
        put(crop, x, crop_y, tile, crop_h, nearest=True)
        rect(x,crop_y,tile,crop_h,region_color)
        rec = {'label':label, 'image':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if j:
            delta = np.asarray(crop,dtype=np.float64)-target
            error = np.sqrt(np.mean(delta**2,axis=2))
            heat = Image.fromarray(lut[np.rint(np.clip(error/limit,0,1)*255).astype(np.uint8)])
            heat.save(crop_dir/f'{j:02d}-error.png')
            np.save(crop_dir/f'{j:02d}-error.npy', error)
            put(heat,x,error_y,tile,crop_h,nearest=True)
            mse = float(np.mean(delta**2))
            rec.update(crop_mse=mse, crop_psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                       mean_pixel_rgb_rms=float(error.mean()), fraction_clipped=float(np.mean(error>limit)))
        manifest['panels'].append(rec)

    # The reference has no reconstruction error; use its third-row space as legend.
    lx=left+30; ly=error_y+crop_h//2-15; bw=tile-60; bh=18
    text(left+tile/2,ly-line_height,'RGB error',23)
    svg.append('<defs><linearGradient id="errorbar">')
    for idx, color in enumerate(lut):
        svg.append(f'<stop offset="{idx/255:.6f}" stop-color="rgb({color[0]},{color[1]},{color[2]})"/>')
    svg.append('</linearGradient></defs>')
    svg.append(f'<rect x="{lx}" y="{ly}" width="{bw}" height="{bh}" fill="url(#errorbar)"/>')
    gradient = np.tile(lut[np.rint(np.linspace(0,255,bw)).astype(int)][None],(bh,1,1))
    canvas.paste(Image.fromarray(gradient),(lx,ly))
    for f in (0,.5,1):
        text(lx+f*bw,ly+bh+line_height,f'{limit*f:g}'+('+' if f==1 else ''),21)
    text(left+tile/2,ly+bh+2*line_height,'Shared scale',21)
    svg.append('</svg>')
    output.write_text('\n'.join(svg)+'\n',encoding='utf-8')
    canvas.save(output.with_suffix('.png'))
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='JSON config; paths resolve relative to the config file')
    parser.add_argument('--reference',help='Reference full-resolution RGB8 image')
    parser.add_argument('--model',action='append',nargs=2,metavar=('LABEL','IMAGE'),help='Repeat in desired column order')
    parser.add_argument('--crop',nargs=4,type=int,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    parser.add_argument('--error-max',type=float,default=20,help='Common upper color limit in RGB8 units (default: 20)')
    parser.add_argument('--output',help='Output .svg file; also writes PNG, JSON, and individual crops/errors')
    args=parser.parse_args()
    if args.config:
        if any(v is not None for v in (args.reference,args.model,args.crop,args.output)):
            parser.error('Use either --config or explicit image/crop/output arguments')
        generate(json.loads(args.config.read_text(encoding='utf-8')),args.config.resolve().parent)
    else:
        if not all((args.reference,args.model,args.crop,args.output)):
            parser.error('--reference, --model, --crop, and --output are required without --config')
        if Path(args.output).suffix.lower() != '.svg':
            parser.error('--output must end in .svg')
        generate(dict(reference=args.reference,models=[dict(label=l,image=p) for l,p in args.model],
                      crop=args.crop,error_max=args.error_max,output=args.output),Path.cwd())


if __name__=='__main__':
    main()
