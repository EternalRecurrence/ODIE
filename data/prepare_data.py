"""Download and prepare the paper's ten DIV2K and six gigapixel inputs."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = None


INPUTS = [{'id': '0028',
  'path': 'data/div2k/0028.png',
  'sha256': '3001a82078238b2436a6ce8be9a5d7b5125d7fd7ce94c4a3c9b4ab1051df6867',
  'shape_hwc': [1992, 2040, 3]},
 {'id': '0036',
  'path': 'data/div2k/0036.png',
  'sha256': '20fcb9d1d5656076152c45069eb9c11b7ff39ffd5f953ad285be34e8ecd36d4d',
  'shape_hwc': [1536, 2040, 3]},
 {'id': '0063',
  'path': 'data/div2k/0063.png',
  'sha256': '7ea1957be776d54b8eab579d7c59e9d7b6f8f667cc96da9498e39bfa29340df3',
  'shape_hwc': [1584, 2040, 3]},
 {'id': '0188',
  'path': 'data/div2k/0188.png',
  'sha256': '20758ead8003c01a119650dfadcc5aee334e1b4e5059c972bfc2a0cc1d347283',
  'shape_hwc': [1548, 2040, 3]},
 {'id': '0337',
  'path': 'data/div2k/0337.png',
  'sha256': '5aef9cb469fda0f0e25b35d033eea6d70b69cb571323913c4063e940d8aaaa36',
  'shape_hwc': [2040, 1632, 3]},
 {'id': '0374',
  'path': 'data/div2k/0374.png',
  'sha256': '3b9d7edbaa6a7c18ff384f45904a119102401425cd1f2800a8b985f5bc40488b',
  'shape_hwc': [2040, 1632, 3]},
 {'id': '0394',
  'path': 'data/div2k/0394.png',
  'sha256': 'b70207f2b306a8afb3840af0a65e2decbce9122e975f68da21586962f09d38cf',
  'shape_hwc': [1536, 2040, 3]},
 {'id': '0460',
  'path': 'data/div2k/0460.png',
  'sha256': '6454e494f736e463bef6baeca9c0a9c30bf2724afc09c324339b8a5fcb19fbaa',
  'shape_hwc': [2040, 1944, 3]},
 {'id': '0545',
  'path': 'data/div2k/0545.png',
  'sha256': '0946aeab167f83bfd80af99d1f813e382cebc3cb0c8dee924c2e43d71cec5e8e',
  'shape_hwc': [2040, 1992, 3]},
 {'id': '0767',
  'path': 'data/div2k/0767.png',
  'sha256': '0bab2ec4906b275e52073167820e1efc599d50baadd36ee3c39ce084e45375a9',
  'shape_hwc': [1572, 2040, 3]},
 {'id': 'dobeln',
  'path': 'data/gigapixel/dobeln.png',
  'sha256': '3addd3e2b59ef5ee923f35e7e26e163a0a13bc35772c288330f777a44c9dd196',
  'shape_hwc': [26800, 44500, 3]},
 {'id': 'london',
  'path': 'data/gigapixel/london.jpg',
  'sha256': '6bb08643e936a2b0bc8e00c14e7c9cd71e1fc9498cca84dba1114c6f96c741c0',
  'shape_hwc': [18589, 63000, 3]},
 {'id': 'mars',
  'path': 'data/gigapixel/mars.png',
  'sha256': '3048396b8933e99543eee59a0d36f710a9185ac3b082e1b90997fc74e83be919',
  'shape_hwc': [18300, 60416, 3]},
 {'id': 'pearl',
  'path': 'data/gigapixel/pearl.png',
  'sha256': '7667b07202855b55bdb0db8e0a8f878c4248a9278428b7a354f666e451348896',
  'shape_hwc': [35500, 34000, 3]},
 {'id': 'tokyo',
  'path': 'data/gigapixel/tokyo.jpg',
  'sha256': 'c6c9f2288d8506998b2cf098f10d81b3e118703b9ecd27d4c075f2ec822fc16b',
  'shape_hwc': [21450, 56718, 3]},
 {'id': 'vela',
  'path': 'data/gigapixel/vela.tif',
  'sha256': '62248116e99a6736aecff4a607c627960d0bcfa946a90275ccfb62d03a6cc0c7',
  'shape_hwc': [35881, 35786, 3]}]

SOURCES = {'div2k': {'source_url': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
           'filename': 'DIV2K_train_HR.zip'},
 'gigapixel': {'london': {'filename': 'london-kings-reach-panorama-2.jpg',
                          'source_url': 'https://upload.wikimedia.org/wikipedia/commons/9/96/Kings_reach_panorama_2.jpg',
                          'sha256': '6bb08643e936a2b0bc8e00c14e7c9cd71e1fc9498cca84dba1114c6f96c741c0',
                          'width': 63000,
                          'height': 18589,
                          'bytes': 98723877},
               'pearl': {'filename': 'pearl-digital-profoundism-3.601gp.jpg',
                         'source_url': 'https://upload.wikimedia.org/wikipedia/commons/3/30/The_most_complete_and_accurate_reconstruction_of_Girl_with_a_Pearl_Earring-21_Gigapixel-Digital_Profoundism-Demo.jpg',
                         'sha256': '24a90aa25276c9cd61bfa41e1cd5acb33a10a229ab779749cb8fe9b33eba788e',
                         'width': 55399,
                         'height': 65000,
                         'bytes': 425240928},
               'vela': {'filename': 'vela-supernova-noirlab2406a.tif',
                        'source_url': 'https://storage.noirlab.edu/media/archives/images/original/noirlab2406a.tif',
                        'sha256': '62248116e99a6736aecff4a607c627960d0bcfa946a90275ccfb62d03a6cc0c7',
                        'width': 35786,
                        'height': 35881,
                        'bytes': 2997725320},
               'dobeln': {'filename': 'dobeln-orthophoto.tif',
                          'source_url': 'https://upload.wikimedia.org/wikipedia/commons/e/e8/Stadt_D%C3%B6beln_%28Landesamt_f%C3%BCr_Geobasisinformation_Sachsen%29.tif?download=1',
                          'sha256': '4ddefec6f1c905b727456f55d33d93f97a51d685280cf2d53b8062bcabe93451',
                          'width': 65535,
                          'height': 39321,
                          'bytes': 3570542924},
               'mars': {'filename': 'mars-acorn.tif',
                        'source_url': 'https://drive.usercontent.google.com/download?id=1Ro1lWxRsl97Jbzm9EA2k9nUEyyVUwxEu&export=download&confirm=t',
                        'sha256': 'd0e77686a72030eafce70768d0f3726bf0c61da440058dbcdab61693b88c53c9',
                        'width': 60416,
                        'height': 22958,
                        'bytes': 1762040264},
               'tokyo': {'filename': 'tokyo.jpg',
                         'source_page': 'https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837/sizes/o/',
                         'sha256': 'c6c9f2288d8506998b2cf098f10d81b3e118703b9ecd27d4c075f2ec822fc16b',
                         'width': 56718,
                         'height': 21450}}}

CROPS = {'mars': {'box_xyxy': [0, 3200, 60416, 21500]},
 'pearl': {'box_xyxy': [15500, 7500, 49500, 43000]},
 'dobeln': {'box_xyxy': [13000, 6500, 57500, 33300]}}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, record):
    actual = sha(path)
    if actual != record['sha256']:
        raise ValueError(f'{path.name}: SHA256 mismatch; expected {record["sha256"]}, got {actual}. '
                         'Existing files are not overwritten. Check source and PNG encoder versions.')
    expected = record.get('shape_hwc')
    size = (expected[1], expected[0]) if expected else (record['width'], record['height'])
    with Image.open(path) as image:
        if image.size != size or image.mode != 'RGB':
            raise ValueError(f'{path.name}: expected RGB {size}, got {image.mode} {image.size}')
    return dict(sha256=actual, bytes=path.stat().st_size, shape_hwc=[size[1], size[0], 3])


def download(url, target, expected_bytes=None):
    """Stream into a resumable partial file; never overwrite a completed file."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'User-Agent': 'ODIE-data-preparation/1.0', 'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    print(f'Download {target.name} (resume offset {offset} bytes)', flush=True)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = getattr(response, 'status', None) == 206
        if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Server returned an unexpected byte range')
        length = response.headers.get('Content-Length')
        received = 0
        progress = 0
        with partial.open('ab' if resumed else 'wb') as output:
            while chunk := response.read(8 * 1024**2):
                output.write(chunk)
                received += len(chunk)
                if received - progress >= 256 * 1024**2:
                    print(f'  {target.name}: {partial.stat().st_size / 1024**2:.0f} MiB', flush=True)
                    progress = received
        if length is not None and received != int(length):
            raise ValueError(f'{target.name}: incomplete HTTP response; rerun to resume')
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError(f'{target.name}: downloaded byte count differs from recorded source')
    partial.replace(target)


def tokyo_url(page):
    request = urllib.request.Request(page, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = html.unescape(response.read(4 * 1024**2).decode('utf-8')).replace('\\/', '/')
    matches = re.findall(r'https://(?:live|farm\d+)\.staticflickr\.com/[^\s"<>]*29314390837_[a-zA-Z0-9]+_o\.jpg', text)
    if not matches:
        raise ValueError('Flickr did not expose the Tokyo original. Download Original from the source page '
                         'as tokyo.jpg and use --source-dir, or pass its direct URL with --tokyo-url.')
    return matches[0]


def target_path(data_dir, record):
    relative = Path(record['path'])
    if relative.parts[0] != 'data' or '..' in relative.parts:
        raise ValueError('Input path must be relative to data/')
    return data_dir.joinpath(*relative.parts[1:])


def install_stream(stream, target, record):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.preparing')
    with partial.open('wb') as output:
        shutil.copyfileobj(stream, output, length=8 * 1024**2)
    result = verify(partial, record)
    partial.replace(target)
    return result


def prepare_div2k(args, records, source):
    missing = []
    results = []
    for record in records:
        target = target_path(args.data_dir, record)
        if target.exists():
            results.append(dict(id=record['id'], status='passed', **verify(target, record)))
        else:
            missing.append(record)
    if not missing:
        return results
    archive = args.div2k_archive or args.data_dir / '.downloads' / source['filename']
    if args.div2k_archive and not archive.is_file():
        raise FileNotFoundError(archive)
    if not args.div2k_archive:
        download(source['source_url'], archive)
    with zipfile.ZipFile(archive) as zipped:
        for record in missing:
            # Extract only the selected exact member, without path traversal or
            # re-encoding; stream reading also verifies its ZIP CRC.
            with zipped.open(f'DIV2K_train_HR/{record["id"]}.png') as stream:
                info = install_stream(stream, target_path(args.data_dir, record), record)
            results.append(dict(id=record['id'], status='passed', **info))
            print(f'Verified DIV2K {record["id"]}', flush=True)
    if not args.div2k_archive and not args.keep_downloads:
        archive.unlink()
    return results


def prepare_gigapixel(args, record, source, crop):
    target = target_path(args.data_dir, record)
    if target.exists():
        return dict(id=record['id'], status='passed', **verify(target, record))
    original = args.source_dir / source['filename'] if args.source_dir else None
    if original is not None and not original.is_file():
        raise FileNotFoundError(f'{original}: place the recorded original here, or omit --source-dir to download')
    downloaded = original is None
    if downloaded:
        original = args.data_dir / '.downloads' / source['filename']
        if not original.exists():
            url = source.get('source_url')
            if not url:
                url = args.tokyo_url or tokyo_url(source['source_page'])
            download(url, original, source.get('bytes'))
    verify(original, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if crop:
        box = crop['box_xyxy']
        if not (0 <= box[0] < box[2] <= source['width'] and 0 <= box[1] < box[3] <= source['height']):
            raise ValueError('Crop exceeds original image bounds')
        partial = target.with_name(target.name + '.preparing')
        with Image.open(original) as image:
            with image.crop(tuple(box)) as region:
                if region.mode != 'RGB':
                    raise ValueError('Recorded originals must be RGB; no implicit pixel conversion')
                region.save(partial, format='PNG', compress_level=1, optimize=False)
        info = verify(partial, record)
        partial.replace(target)
        if downloaded and not args.keep_downloads:
            original.unlink()
    elif downloaded and not args.keep_downloads:
        # Full-image inputs can be moved instead of duplicated or re-encoded.
        original.replace(target)
        info = verify(target, record)
    else:
        with original.open('rb') as stream:
            info = install_stream(stream, target, record)
    print(f'Verified {record["id"]}', flush=True)
    return dict(id=record['id'], status='passed', **info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=['div2k', 'tokyo', 'london', 'vela', 'mars', 'pearl', 'dobeln'],
                        help='Default: DIV2K and all six gigapixel images, processed serially')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--source-dir', type=Path, help='Use existing gigapixel originals named in this script and data/README.md')
    parser.add_argument('--div2k-archive', type=Path, help='Use an existing official DIV2K_train_HR.zip')
    parser.add_argument('--tokyo-url', help='Direct original JPEG URL if Flickr original-page lookup is unavailable')
    parser.add_argument('--keep-downloads', action='store_true', help='Keep downloaded archives/originals after successful preparation')
    parser.add_argument('--verify-only', action='store_true', help='Check prepared files without downloading or cropping')
    parser.add_argument('--plan', action='store_true', help='Print sources, crop boxes and targets without downloads or writes')
    args = parser.parse_args()
    records = INPUTS
    sources = SOURCES
    crops = CROPS
    selected = args.only or ['div2k', 'tokyo', 'london', 'vela', 'mars', 'pearl', 'dobeln']
    records = [r for r in records if ('div2k' if r['id'].isdigit() else r['id']) in selected]
    if args.plan:
        for record in records:
            key = record['id']
            source = sources['div2k'] if key.isdigit() else sources['gigapixel'][key]
            print(json.dumps(dict(id=key, target=str(target_path(args.data_dir, record)),
                source=source, crop=crops.get(key, {}).get('box_xyxy')), ensure_ascii=False))
        return
    results = []
    try:
        if args.verify_only:
            for record in records:
                results.append(dict(id=record['id'], status='passed', **verify(target_path(args.data_dir, record), record)))
        else:
            if 'div2k' in selected:
                results.extend(prepare_div2k(args, [r for r in records if r['id'].isdigit()], sources['div2k']))
            for record in records:
                if not record['id'].isdigit():
                    key = record['id']
                    results.append(prepare_gigapixel(args, record, sources['gigapixel'][key], crops.get(key)))
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f'Data preparation failed: {error}\n')
    print(json.dumps(dict(status='passed', inputs=results), indent=2))


if __name__ == '__main__':
    main()
