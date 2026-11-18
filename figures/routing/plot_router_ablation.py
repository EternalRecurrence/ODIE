"""Plot exported router diagnostics; no model loading or GPU is required.

NPZ inputs contain gates/visible (H, W, experts) and rgb (H, W, 3).
Gates are independent weights in [0, 1]; visible includes active-level flags.
RGB is uint8 or floating point in the [0, 1] output convention.
Expert numbers on the command line are one-based, with inclusive ranges.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.font_manager import fontManager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_diagnostics(path):
    with np.load(path, allow_pickle=False) as data:
        gates, visible, rgb = (data[key] for key in ('gates', 'visible', 'rgb'))
    if gates.ndim != 3 or visible.shape != gates.shape or visible.dtype != np.bool_:
        raise ValueError('gates and boolean visible must have the same HWC shape')
    if not np.isfinite(gates).all() or np.any((gates < 0) | (gates > 1)):
        raise ValueError('gates must contain finite weights in [0, 1]')
    if rgb.shape != (*gates.shape[:2], 3) or not np.isfinite(rgb).all():
        raise ValueError('rgb must contain finite HWC RGB values matching the masks')
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        raise ValueError('rgb must be floating point or uint8')
    return gates, visible, rgb


def parse_group(text, n_experts):
    try:
        label, interval = text.split('=', 1)
        first, last = map(int, interval.split('-'))
        if not label or not 1 <= first <= last <= n_experts:
            raise ValueError
    except ValueError:
        raise ValueError(f'Expected LABEL=FIRST-LAST within 1..{n_experts}: {text}') from None
    return label, first, last


def plot(args):
    gates, on, on_rgb = load_diagnostics(args.with_router)
    _, off, off_rgb = load_diagnostics(args.without_router)
    with Image.open(args.reference) as image:
        reference = np.array(image.convert('RGB'))
    height, width = gates.shape[:2]
    if reference.shape != on_rgb.shape or off_rgb.shape != on_rgb.shape:
        raise ValueError('Reference and both diagnostic crops must have identical dimensions')
    groups = [parse_group(text, min(on.shape[-1], off.shape[-1])) for text in args.groups]
    if args.layout in ('compact', 'errors') and len(groups) not in (1, 2):
        raise ValueError('Compact layout needs one or two routing groups; activation uses the last group')

    metadata = [json.loads(path.read_text(encoding='utf-8')) if path else {}
                for path in (args.with_metadata, args.without_metadata)]
    if all('box_xyxy' in meta for meta in metadata):
        if metadata[0]['box_xyxy'] != metadata[1]['box_xyxy']:
            raise ValueError('Diagnostic crops use different full-image coordinates')
    for meta, mask in zip(metadata, (on, off)):
        if 'shape_hw' in meta and meta['shape_hw'] != [height, width]:
            raise ValueError('Metadata crop dimensions do not match the arrays')
        if 'resolutions' in meta and len(meta['resolutions']) != mask.shape[-1]:
            raise ValueError('Metadata resolution count does not match expert count')

    means, totals, counts, off_counts, on_masks, off_masks = [], [], [], [], [], []
    for _, first, last in groups:
        selected = slice(first-1, last)
        count = on[..., selected].sum(axis=-1)
        total = (gates[..., selected]*on[..., selected]).sum(axis=-1)
        if args.mean_denominator == 'group':
            mean = total/(last-first+1)
        else:
            mean = np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float32),
                             where=count > 0)
        means.append(mean)
        totals.append(total)
        counts.append(count)
        off_count = off[..., selected].sum(axis=-1)
        off_counts.append(off_count)
        on_masks.append(count > 0)
        off_masks.append(off_count > 0)

    normalized_sum = args.routing_statistic == 'normalized-sum'
    routing_values = [total/total.max() if total.max() > 0 else np.zeros_like(total)
                      for total in totals] if normalized_sum else means
    routing_title = 'normalized router weight sum' if normalized_sum else 'mean router weight'

    errors = [np.sqrt(np.mean((rgb.astype(np.float64)-reference.astype(np.float64))**2, axis=-1))
              for rgb in (off_rgb, on_rgb)]
    error_metrics = [dict(mse=float(np.mean(error**2)),
                          psnr_db=float(10*np.log10(255**2/np.mean(error**2))) if error.any() else None,
                          mean_pixel_rgb_rms=float(error.mean()),
                          fraction_above_error_max=float(np.mean(error > args.error_max))) for error in errors]

    for path in args.font_files:
        fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': args.font_family, 'font.size': args.font_size,
                         'axes.titlesize': args.font_size, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'})
    cmap = LinearSegmentedColormap.from_list(
        'black_red_yellow_white', ['#000000', '#ff0000', '#ffff00', '#ffffff'])
    cmap.set_bad('#000000')
    binary = ListedColormap(['black', 'white'])
    compact = args.layout in ('compact', 'errors')
    error_layout = args.layout == 'errors'
    rows, columns = (2, len(groups)+2) if compact else (len(groups)+1, 3)
    legend_panel = compact and columns == 4
    narrow = args.width <= 4
    fig = plt.figure(figsize=(args.width, args.width*rows/columns*2.65/2.36))
    grid = fig.add_gridspec(rows, columns, left=.045, right=.95 if legend_panel else .90, bottom=.075, top=.91,
                            wspace=.16 if narrow else .08, hspace=.65 if narrow else .40)
    axes = [[fig.add_subplot(grid[row, col]) for col in range(columns)] for row in range(rows)]
    panels = [[('Reference', reference, None), ('w/o router', off_rgb, None),
               ('Full', on_rgb, None)]]
    if compact:
        if legend_panel:
            panels[0].append(('Legend', None, None))
        panels.append([
            (f'{groups[-1][0]} activation w/o ({off_masks[-1].mean()*100:.2f}%)', off_masks[-1], binary),
            (f'{groups[-1][0]} activation w/ ({on_masks[-1].mean()*100:.2f}%)', on_masks[-1], binary),
            *[(routing_title, values, cmap) for values in routing_values]])
        if error_layout:
            panels[1][:2] = [
                (f'RGB error w/o ({error_metrics[0]["psnr_db"]:.2f} dB)', errors[0], plt.get_cmap('magma')),
                (f'RGB error Full ({error_metrics[1]["psnr_db"]:.2f} dB)', errors[1], plt.get_cmap('magma'))]
    else:
        for mean, off_mask, on_mask in zip(routing_values, off_masks, on_masks):
            panels.append([
                (f'Activation w/o router ({off_mask.mean()*100:.2f}%)', off_mask, binary),
                (f'Activation w/ router ({on_mask.mean()*100:.2f}%)', on_mask, binary),
                (routing_title, mean, cmap)])
    for row in range(rows):
        for col, (title, values, color) in enumerate(panels[row]):
            ax = axes[row][col]
            if values is None:
                ax.set_axis_off()
                ax.set_title(title, pad=5)
                if error_layout:
                    ax.text(.1, .85, 'RGB error (RGB8 RMS)', transform=ax.transAxes, fontsize=args.font_size)
                    error_bar = fig.colorbar(
                        plt.cm.ScalarMappable(norm=plt.Normalize(0, args.error_max), cmap='magma'),
                        cax=ax.inset_axes([.1, .69, .8, .055]), orientation='horizontal',
                        ticks=[0, args.error_max/2, args.error_max], extend='max')
                    error_bar.ax.tick_params(labelsize=args.font_size, length=2)
                    ax.text(.1, .53, 'Shared error scale; values above max saturate',
                            transform=ax.transAxes, fontsize=7)
                else:
                    for y, value, label in ((.78, 'black', 'Inactive (0)'), (.62, 'white', 'Active (1)')):
                        ax.add_patch(Rectangle((.1, y-.04), .12, .09, facecolor=value,
                                               edgecolor='black', linewidth=.7, transform=ax.transAxes))
                        ax.text(.28, y, label, transform=ax.transAxes, va='center', fontsize=args.font_size)
                ax.text(.1, .40, routing_title, transform=ax.transAxes, fontsize=args.font_size)
                bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
                                   cax=ax.inset_axes([.1, .24, .8, .055]), orientation='horizontal',
                                   ticks=[0, .5, 1])
                bar.ax.tick_params(labelsize=args.font_size, length=2)
                ax.text(.1, .08, 'Sum / crop maximum' if normalized_sum else 'One shared scale; no normalization',
                        transform=ax.transAxes, fontsize=7)
                continue
            error_panel = error_layout and row == 1 and col < 2
            shown = ax.imshow(values, cmap=color, vmin=0 if color else None,
                              vmax=(args.error_max if error_panel else 1) if color else None,
                              interpolation='nearest')
            if not legend_panel and row > 0 and col == 2:
                box = ax.get_position()
                bar = fig.colorbar(shown, cax=fig.add_axes([.925, box.y0, .014, box.height]),
                                   ticks=[0, .5, 1])
                bar.ax.tick_params(labelsize=args.font_size, length=2)
                bar.set_label('Normalized sum' if normalized_sum else 'Mean weight', fontsize=args.font_size, labelpad=3)
            if narrow and error_layout and row == 1:
                title = title.replace(' (', '\n(')
                if col == 2:
                    title = 'Normalized\nrouter weight sum' if normalized_sum else 'Mean\nrouter weight'
            ax.set_title(title, pad=5, fontweight='bold' if (row, col) == (0, 2) else 'normal')
            ax.set_axis_off()
            ax.add_patch(Rectangle((-.5, -.5), width-.001, height-.001, fill=False,
                                   edgecolor=args.border_color, linewidth=.7))
    labels = ['(a) Reference and reconstructions'] + [
        f'({chr(98+index)}) {label} experts (E{first}-E{last}): activation and routing'
        for index, (label, first, last) in enumerate(groups)]
    if compact:
        labels = ['(a) Reference and reconstructions',
                  f'(b) {groups[-1][0]} spatial activation and {"/".join(group[0] for group in groups)} routing']
    if error_layout:
        labels[1] = ('(b) Reconstruction errors and normalized router weight sum' if normalized_sum
                     else '(b) Reconstruction errors and mean router weights')
        if not legend_panel:
            box = axes[0][-1].get_position()
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, args.error_max), cmap='magma'),
                               cax=fig.add_axes([.925, box.y0, .014, box.height]),
                               ticks=[0, args.error_max/2, args.error_max], extend='max')
            bar.ax.tick_params(labelsize=args.font_size, length=2)
            bar.set_label('RGB8 RMS error', fontsize=args.font_size, labelpad=3)
    for row, label in enumerate(labels):
        if narrow:
            label = ['(a) Reconstructions', '(b) Errors and router weights'][row] if error_layout else label
        fig.text(.045, axes[row][0].get_position().y1+(.15 if narrow and row else .09 if narrow else .052), label, fontsize=args.font_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = [Path(f'{args.output}.{extension}') for extension in ('png', 'pdf', 'svg')]
    for path in outputs:
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    data_path = Path(f'{args.output}.data.npz')
    np.savez_compressed(data_path, active_means=np.stack(means, axis=-1),
                        routing_sums=np.stack(totals, axis=-1),
                        routing_values=np.stack(routing_values, axis=-1),
                        active_counts=np.stack(counts, axis=-1),
                        without_router_active_counts=np.stack(off_counts, axis=-1),
                        activation_masks_with_router=np.stack(on_masks, axis=-1),
                        activation_masks_without_router=np.stack(off_masks, axis=-1),
                        rgb_error_without_router=errors[0], rgb_error_with_router=errors[1])
    outputs.append(data_path)
    inputs = [args.with_router, args.without_router, args.reference,
              args.with_metadata, args.without_metadata, *args.font_files]
    record = dict(
        schema='router-ablation-figure-v7', script_sha256=sha(__file__),
        layout=args.layout,
        activation_group_shown=groups[-1][0] if compact else 'all groups',
        inputs=[dict(file=path.name, sha256=sha(path)) for path in inputs if path],
        shape_hw=[height, width],
        box_xyxy=metadata[0].get('box_xyxy'),
        checkpoint_sha256=[meta.get('checkpoint_sha256') for meta in metadata],
        routing_groups=[dict(label=label, experts=[first, last]) for label, first, last in groups],
        mean=('sum(gate * visible) / number of experts in group' if args.mean_denominator == 'group'
              else 'sum(gate * visible) / sum(visible), per pixel and group'),
        mean_denominator=args.mean_denominator,
        routing_statistic=args.routing_statistic,
        routing_sum_ranges=[dict(min=float(total.min()), max=float(total.max())) for total in totals],
        routing_display=('sum(gate * visible) / max over crop of sum(gate * visible), per group; all-zero stays zero'
                         if normalized_sum else 'mean'),
        error=dict(definition='sqrt(mean((prediction-reference)^2, RGB)), RGB8 units',
                   limits=[0, args.error_max], colormap='magma',
                   without_router=error_metrics[0], with_router=error_metrics[1]),
        activation_mask='any(visible) within the group, binary 0/1; same groups as routing',
        no_active_expert='Zero' if args.mean_denominator == 'group' else 'NaN in data; black in figure',
        processing=('Sum normalized by each group maximum over the input crop; native arrays; no smoothing'
                    if normalized_sum else 'Fixed [0,1] scale; native arrays; no smoothing or per-panel normalization'),
        style=dict(font_family=args.font_family, width_inches=args.width, dpi=args.dpi,
                   border_color=args.border_color),
        coverage=[dict(label=group[0], experts=list(group[1:]),
                       with_router=float(on_mask.mean()), without_router=float(off_mask.mean()))
                  for group, on_mask, off_mask in zip(groups, on_masks, off_masks)],
        outputs={path.name: sha(path) for path in outputs})
    Path(f'{args.output}.provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {args.output}.{{png,pdf,svg,data.npz,provenance.json}}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-router', type=Path, required=True, help='NPZ diagnostics with router')
    parser.add_argument('--without-router', type=Path, required=True, help='NPZ diagnostics without router')
    parser.add_argument('--reference', type=Path, required=True, help='Reference crop at native resolution')
    parser.add_argument('--with-metadata', type=Path, help='With-router export provenance JSON')
    parser.add_argument('--without-metadata', type=Path, help='Without-router export provenance JSON')
    parser.add_argument('--groups', nargs='+', required=True, help='LABEL=FIRST-LAST, e.g. Middle=7-12 Fine=13-16')
    parser.add_argument('--mean-denominator', choices=('active', 'group'), default='active',
                        help='active: divide by active count; group: fixed group size, inactive weights are zero')
    parser.add_argument('--routing-statistic', choices=('mean', 'normalized-sum'), default='mean',
                        help='normalized-sum: sum visible weights, then divide by the maximum over the input crop')
    parser.add_argument('--layout', choices=('groups', 'compact', 'errors'), default='groups',
                        help='compact/errors: one or two groups; errors replaces masks with RGB error')
    parser.add_argument('--error-max', type=float, default=20, help='Shared RGB8 RMS error color limit')
    parser.add_argument('--output', type=Path, required=True, help='Output filename stem, without extension')
    parser.add_argument('--font-family', default='Times New Roman')
    parser.add_argument('--font-files', type=Path, nargs='*', default=[], help='Optional local TTF/OTF files')
    parser.add_argument('--width', type=float, default=7.08, help='Figure width in inches')
    parser.add_argument('--font-size', type=float, default=9, help='Text size in points at generated width')
    parser.add_argument('--dpi', type=int, default=240)
    parser.add_argument('--border-color', default='#7a4a9b')
    args = parser.parse_args()
    if args.width <= 0 or args.dpi <= 0:
        parser.error('width and dpi must be positive')
    if not np.isfinite(args.error_max) or args.error_max <= 0:
        parser.error('error-max must be finite and positive')
    plot(args)


if __name__ == '__main__':
    main()
