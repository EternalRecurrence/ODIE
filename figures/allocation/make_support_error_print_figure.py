"""Reflow a generated support/error comparison for an IEEE single column.

Reads the original generator's panels and metric receipt, without recomputing
errors or changing masks. Produces an SVG with embedded pixels and 8 pt vector
labels. Run after make_region_ablation_error_figure.py with the same config.
"""
import argparse
import base64
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image


def generate(config_path, output, width_mm=88.57, font_pt=8):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    original = (config_path.parent / config['output']).resolve()
    receipt = json.loads(original.with_suffix('.json').read_text(encoding='utf-8'))
    panels = original.parent / (original.stem + '-panels')
    methods = ['Reference'] + [item['label'] for item in config['models']]
    tile, gap, left, right = 384, 20, 20, 116
    width = left + len(methods) * (tile + gap) - gap + right
    font = font_pt * width / (width_mm * 72 / 25.4)
    line = font * 1.15
    # Keep the square crop and a separate shared legend in the margin.
    region, = config['regions']
    crop_y = round(5 * line)
    second_y = round(crop_y + tile + 1.8 * line)
    third_y = round(second_y + tile + 4 * line)
    height = round(third_y + tile + 2.8 * line)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
           f'height="{width_mm * height / width}mm" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="white"/>']

    def text(x, y, value, bold=False, anchor='middle'):
        svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                   f'font-family="Times New Roman" font-size="{font}" '
                   f'font-weight="{"bold" if bold else "normal"}">{escape(value)}</text>')

    def image(path, x, y):
        with Image.open(path) as source:
            if source.size != (tile, tile):
                raise ValueError('This print layout requires square 384 px panels')
            data = base64.b64encode(path.read_bytes()).decode('ascii')
        svg.append(f'<image x="{x}" y="{y}" width="{tile}" height="{tile}" '
                   f'href="data:image/png;base64,{data}"/>')

    for col, method in enumerate(methods):
        x = left + col * (tile + gap)
        text(x + tile / 2, line, method, col == len(methods) - 1)
        if col:
            metric = receipt['methods'][col - 1]['full_psnr_db']
            text(x + tile / 2, 2.2 * line, 'Image PSNR')
            text(x + tile / 2, 3.2 * line, f'{metric:.3f} dB')
        image(panels / f'{region}-{col:02d}-crop.png', x, crop_y)
    text(left, crop_y - .35 * line, 'Reconstruction', True, 'start')
    for subset, y, label in [('inside', second_y, 'Active region'),
                              ('outside', third_y, 'Inactive region')]:
        records = [r for r in receipt['support_metrics']
                   if r['region'] == region and r['subset'] == subset]
        text(left, y - .35 * line, f'{label} ({records[0]["fraction"] * 100:.1f}%)', True, 'start')
        for col, method in enumerate(methods):
            x = left + col * (tile + gap)
            image(panels / f'{region}-{subset}-{col:02d}.png', x, y)
            if col:
                record, = [r for r in records if r['label'] == method]
                text(x + tile / 2, y + tile + 1.15 * line, f'RMSE: {record["rmse_rgb8"]:.3f}')
                text(x + tile / 2, y + tile + 2.15 * line, f'PSNR: {record["psnr_db"]:.3f} dB')
            else:
                text(x + tile / 2, y + tile + 1.15 * line, 'Reference in')
                text(x + tile / 2, y + tile + 2.15 * line, 'selected subset')

    # Match the original generator's RGB8 LUT and clipping, with no normalization.
    anchors = np.array([[12, 7, 30], [52, 15, 93], [111, 25, 110], [176, 44, 91],
                        [229, 81, 57], [251, 145, 52], [252, 209, 111], [252, 253, 191]], float)
    lut = np.stack([np.interp(np.linspace(0, 7, 256), np.arange(8), anchors[:, c])
                    for c in range(3)], axis=1).astype(np.uint8)
    gradient = np.tile(lut[np.rint(np.linspace(255, 0, tile)).astype(int)][:, None], (1, 20, 1))
    buffer = io.BytesIO()
    Image.fromarray(gradient).save(buffer, format='PNG')
    data = base64.b64encode(buffer.getvalue()).decode('ascii')
    lx = width - right + 14
    svg.append(f'<image x="{lx}" y="{second_y}" width="20" height="{tile}" '
               f'href="data:image/png;base64,{data}"/>')
    text(lx + 25, second_y - .35 * line, 'RMSE')
    for fraction in (0, .5, 1):
        text(lx + 28, second_y + fraction * (tile - 1) + font / 3,
             f'{config.get("error_max", 20) * (1 - fraction):g}' + ('+' if fraction == 0 else ''),
             anchor='start')
    svg.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.config.resolve(), args.output.resolve())
