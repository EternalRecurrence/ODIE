# Input data

Full source images are not included. `prepare_data.py` contains the download sources, crop coordinates, output paths, input SHA256 values and RGB dimensions. Keep the ten DIV2K PNGs unchanged: re-encoding changes byte-based capacity targets.

## Download and prepare

Run from the repository root with Python >= 3.10 and Pillow installed. The script downloads the official DIV2K training ZIP, extracts only the ten selected PNGs without re-encoding, then prepares the six gigapixel images serially:

```bash
python data/prepare_data.py --plan
python data/prepare_data.py
```

To prepare a subset or check existing inputs:

```bash
python data/prepare_data.py --only div2k
python data/prepare_data.py --only mars pearl dobeln
python data/prepare_data.py --verify-only
```

Every prepared input must match the recorded RGB dimensions and SHA256. Existing mismatched files cause an error and are not overwritten. Downloads stream to `data/.downloads/` and resume `.part` files when the source supports HTTP byte ranges; a server ignoring ranges restarts that partial download. After successful verification the script removes its downloaded intermediate archive/original, or moves an uncropped original directly to its target. Use `--keep-downloads` to retain intermediates. Files supplied with the following local-input options are never deleted:

```bash
python data/prepare_data.py --only div2k --div2k-archive /path/to/DIV2K_train_HR.zip
python data/prepare_data.py --only mars pearl dobeln --source-dir /path/to/originals
python data/prepare_data.py --only tokyo --source-dir /path/to/originals
```

Original filenames are recorded in the script's `SOURCES`: `tokyo.jpg`, `london-kings-reach-panorama-2.jpg`, `vela-supernova-noirlab2406a.tif`, `mars-acorn.tif`, `pearl-digital-profoundism-3.601gp.jpg`, and `dobeln-orthophoto.tif`. `INPUTS` contains final identities and `CROPS` contains the three half-open crop boxes. The script saves cropped RGB pixels without resizing, using PNG `compress_level=1, optimize=False`, to preserve the experiment inputs. PNG byte identity can still depend on Pillow/zlib and JPEG decoding versions; mismatched files are rejected.

Tokyo's URL is resolved from the source page's Original image link, never a thumbnail. If Flickr blocks automatic access, download Original from that page as `tokyo.jpg` and use `--source-dir`, or provide its direct JPEG URL with `--tokyo-url`. The original is always checked against its recorded SHA256.

The full official DIV2K ZIP is downloaded even though only ten images are extracted. Cropping gigapixel images with Pillow decodes the original in memory; use a machine with at least 32 GiB RAM for the largest source. Processing is serial and defaults to releasing intermediate download files after each successful image. These commands prepare external datasets; the downloaded data stays ignored by Git.

## DIV2K

Download the original high-resolution DIV2K training images from https://data.vision.ee.ethz.ch/cvl/DIV2K/ . Retained IDs are 0028, 0036, 0063, 0188, 0337, 0374, 0394, 0460, 0545 and 0767. Put them in `data/div2k/ID.png`. Preserve the original PNG bytes. See the dataset source for its usage terms and cite the DIV2K/NTIRE papers.

## Gigapixel sources

| Image | Source | Attribution / recorded terms |
|---|---|---|
| Tokyo | https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837 | Trevor Dobson; obtain the original 56718x21450 version and follow the source page terms |
| London | https://upload.wikimedia.org/wikipedia/commons/9/96/Kings_reach_panorama_2.jpg | William Hall; CC BY-SA 3.0 |
| Pearl | https://upload.wikimedia.org/wikipedia/commons/3/30/The_most_complete_and_accurate_reconstruction_of_Girl_with_a_Pearl_Earring-21_Gigapixel-Digital_Profoundism-Demo.jpg | Koorosh Orooj; CC BY-SA 4.0; digital reconstruction |
| Vela | https://storage.noirlab.edu/media/archives/images/original/noirlab2406a.tif | CTIO/NOIRLab/DOE/NSF/AURA; Image Processing: T.A. Rector, M. Zamani, D. de Martin; CC BY 4.0 |
| Dobeln | https://upload.wikimedia.org/wikipedia/commons/e/e8/Stadt_D%C3%B6beln_%28Landesamt_f%C3%BCr_Geobasisinformation_Sachsen%29.tif?download=1 | GeoSN; stitched/downscaled by PantheraLeo1359531; dl-de/by-2-0 |
| Mars | https://drive.usercontent.google.com/download?id=1Ro1lWxRsl97Jbzm9EA2k9nUEyyVUwxEu&export=download&confirm=t | NASA/JPL-Caltech/MSSS; ACORN README: public domain |

For Mars, the recorded file is the ACORN-distributed TIFF version; the NASA source page is https://science.nasa.gov/photojournal/curiositys-18-billion-pixel-panorama/ . The supplied source record identifies the exact downloaded variant.

London, Vela and Tokyo use full source images. Mars, Pearl and Dobeln use RGB crops, without resizing, with the following half-open boxes:

| Image | Box [left, top, right, bottom] | Output WxH |
|---|---|---|
| Mars | [0, 3200, 60416, 21500] | 60416x18300 |
| Pearl | [15500, 7500, 49500, 43000] | 34000x35500 |
| Dobeln | [13000, 6500, 57500, 33300] | 44500x26800 |

The script saves the prepared inputs under `data/div2k/` and `data/gigapixel/`. The crop operation is equivalent to:

```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
with Image.open('mars-acorn.tif') as source:
    source.crop((0, 3200, 60416, 21500)).save(
        'data/gigapixel/mars.png', compress_level=1, optimize=False)
```

PNG encoding versions/options can affect file-byte SHA256 while leaving RGB pixels unchanged. The recorded original and cropped input hashes identify the precise experimental files; use the recorded pixel dimensions and crop without rescaling. Input hashes bind each capacity reference to its exact input file. Do not mix capacity records from a differently encoded input.

## Neural Experts pretraining

The pinned official repository includes `data/rgb_img/kodim19.png`. The portable run lists use that image once for random-assignment pretraining and share the resulting initialization. It is obtained with the official dependency checkout, not bundled again here. Its recorded SHA256 is `b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc`.

## Included figure excerpts

Only small source excerpts/overviews needed to inspect the paper figures are included under `figures/`. They inherit the corresponding source-image terms; the MIT license for ODIE's original code does not replace those terms.
