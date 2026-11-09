# ODIE — On-Demand INR Experts

Code and experimental records for **Implicit Neural Representations for Gigapixel Images with On-Demand Parameter Allocation and Expert Routing**. ODIE combines multiresolution grid experts, position-dependent routing and a shared RGB decoder, with residual-driven allocation of sparse supports.

## Installation

Training requires Linux, an NVIDIA GPU, a compatible CUDA-enabled PyTorch/toolkit and C++ compiler, CMake >= 3.24, and Git. The validated build environment used Python 3.10, PyTorch 2.5.1+cu118, CUDA 11.8 and GCC 11.

Run from the repository root after installing PyTorch:

```bash
python -m pip install -e '.[plots]'
python scripts/prepare_vendor.py --methods ingp siren finer miner neural_experts
python scripts/build_native.py
python -m pip install --no-build-isolation vendor/ingp/dependencies/tiny-cuda-nn/bindings/torch
python -m pip install -r requirements-baseline-entry.txt -r requirements-ingp-entry.txt
python scripts/prepare_vendor.py --verify --methods ingp siren finer miner neural_experts
```

Official revisions and patches are pinned in [upstreams.lock.json](upstreams.lock.json). `build_native.py` builds INGP's `pyngp`; ODIE compiles its CUDA extension on first use. Optional `libjbig-dev` enables JBIG support compression; otherwise export uses lossless bitmap/DEFLATE encoding. To prepare only ODIE's dependency, select `--methods ingp`.

## Data

```bash
python data/prepare_data.py --plan
python data/prepare_data.py --only div2k
# Omit --only to prepare DIV2K and all six gigapixel inputs.
python data/prepare_data.py --verify-only --only div2k
```

Sources, crop boxes and hashes are embedded in the script. Existing downloads can be supplied with `--source-dir` or `--div2k-archive`; see [data/README.md](data/README.md) for details. Preserve original DIV2K PNG bytes because file sizes determine capacity targets. Downloads are ignored by Git.

## Paper experiments

The five run lists cover 125 distinct image fits. Run the main lists before their ablations:

```bash
python scripts/run_baselines.py --manifest configs/experiments/div2k_50.json --vendor-root vendor
python scripts/run_baselines.py --manifest configs/experiments/div2k_500.json --vendor-root vendor
python scripts/run_baselines.py --manifest configs/experiments/gigapixel_50.json --vendor-root vendor
python scripts/run_baselines.py --manifest configs/experiments/tokyo_components.json --vendor-root vendor
python scripts/run_baselines.py --manifest configs/experiments/expert_count.json --vendor-root vendor
```

Manifest paths resolve relative to that file. Jobs run serially, INGP precedes ODIE for each image, and outputs go to new or empty directories under `reruns/`. `--dry-run` checks a list without training but still requires inputs and applicable capacity references. Input-conversion caches are released between jobs.

The paper uses 50 MiB gigapixel configs. ODIE matches INGP's native-verified parameter count; activation information is measured separately. DIV2K uses 50 epochs, with additional 500-epoch ODIE/INGP runs. Settings are in [configs/](configs/), with resolved experiment configs in `results/`.

Neural Experts shares one 3,000-update random-assignment pretraining on official `kodim19.png` across ten fits, using a width matched to 50% of the mean DIV2K PNG size. Pretraining is recorded in `reruns/shared-pretraining/` and excluded from fitting time. MINER retains its official best-image return and a 1.5-times allocation allowance; CR uses actual stored size.

## Reconstruction and evaluation

Training exports `reconstruction.png`, model files, logs and `metrics.json`. To decode an ODIE model and evaluate the exported RGB8 image:

```bash
python scripts/reconstruct.py \
  --checkpoint reruns/gigapixel_50/tokyo/odie/model_latest_fp16.pt \
  --output reruns/gigapixel_50/tokyo/odie/reconstructed.png
python scripts/evaluate_reconstruction.py \
  --input data/gigapixel/tokyo.jpg \
  --prediction reruns/gigapixel_50/tokyo/odie/reconstruction.png \
  --model-files reruns/gigapixel_50/tokyo/odie/model_latest_fp16.pt \
  --checkpoint latest --output reruns/gigapixel_50/tokyo/odie/evaluation.json
```

Use `--quality-only` instead of `--model-files` to check only an image. For CR, list all decoding files, including sidecars such as INGP's `image_metadata.json`. Quality evaluation uses full-resolution RGB8 without resizing and SSIM with a uniform 7×7 window/sample covariance. Use final `evaluation` fields, not training-time diagnostic PSNR.

Paper CR is `H × W × 3 / complete inference bytes` (`evaluation.compression.cr_raw_rgb8`). ODIE's Table II parameter-only CR uses `parameters_only.cr_raw_rgb8`, counting saved FP16 grid/router/decoder weights. `parameters_and_activation` excludes container overhead and is not total CR. INGP uses complete-file CR in both rows. Per-image metrics are averaged arithmetically.

`training_loop_seconds` measures the ODIE and Neural Experts training loops, excluding setup and final export. Other elapsed-time fields may include export; their scope is recorded with the results.

## Results and figures

| Paper item | Retained records |
|---|---|
| Table I | `results/div2k_50/` and `results/div2k_500/` |
| Table II | `results/gigapixel_50/` |
| Table III | `results/tokyo_components/`; Full reuses Tokyo ODIE in Table II |
| Table IV | `results/expert_count/`; 16 experts reuse Table I's ODIE results |
| Figs. 1–5 | [architecture](figures/architecture/), [DIV2K](figures/div2k/), [Tokyo](figures/tokyo/), [routing](figures/routing/), [allocation](figures/allocation/); each contains its final PDF and source inputs |

The repository includes experiment records and small plotting inputs. Full datasets, reconstructions, model weights and caches are not included. Full-image plotting requires external files; supplied crops/arrays support offline ablation plotting.

Each figure directory groups its scripts, config, data and final PDF. Architecture and Tokyo comparison have SVG sources. Run these commands from the repository root to regenerate the other figures (DIV2K requires full reference/reconstruction files at the paths in its config):

```bash
python figures/div2k/make_reconstruction_error_figure.py --config figures/div2k/config.json
python figures/routing/plot_router_ablation.py \
  --with-router figures/routing/with-router.npz --without-router figures/routing/without-router.npz \
  --reference figures/routing/reference.png \
  --with-metadata figures/routing/with-router.json --without-metadata figures/routing/without-router.json \
  --groups Middle+Fine=7-16 --layout errors --mean-denominator group --routing-statistic normalized-sum \
  --font-family 'DejaVu Serif' --font-size 8.4 --width 3.5 --output figures/routing/generated
python figures/allocation/make_region_ablation_error_figure.py --config figures/allocation/config.json
python figures/allocation/make_support_error_print_figure.py --config figures/allocation/config.json --output figures/allocation/print.svg
```

Config paths resolve relative to the config file; crop boxes use half-open `[left, top, right, bottom]` original-image coordinates. Error maps show per-pixel RGB RMSE in RGB8 units on a shared 0–20 scale. The routing map sums visible weights of experts 7–16 and divides by the maximum sum over the crop. Allocation comparisons use the same union of Full's active cells for experts 14–16. Generated SVGs embed their image panels; use Times New Roman, if available, to match the paper's routing typography.

## License

ODIE's original code is licensed under the [MIT License](LICENSE). Third-party code, upstream-derived portions of patches, and source images retain their applicable terms; see [third-party notices](patches/licenses/README.md) and [data attribution](data/README.md).
