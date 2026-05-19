# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PromptIR (NeurIPS'23) — a single all-in-one blind image restoration network handling **low-light enhancement, deraining, and dehazing** via learnable prompt blocks injected into a Restormer-style transformer UNet. This fork swaps the original Gaussian-denoise task for paired low-light enhancement on **LOL-v2** (Real_captured + Synthetic subsets); all training/eval/dataset code reflects that swap. Based on AirNet and Restormer.

## Environment

Single conda env defined in `env.yml` (PyTorch 1.8.1, CUDA 11.6, Python 3.8, PyTorch Lightning 2.0.1):

```bash
conda env create -f env.yml
conda activate promptir
```

Key runtime deps: `lightning`, `einops`, `wandb`, `opencv-python`, `scikit-image`. There is no lint/test harness — evaluation runs via `test.py` producing PSNR/SSIM.

## Common commands

| Task | Command |
|---|---|
| Train (defaults: 120 epochs, 4 GPUs DDP, lowlight + derain + dehaze) | `python train.py` |
| Train a subset | `python train.py --de_type derain dehaze` |
| Resume from `train_ckpt/last.ckpt` if present | `python train.py --resume auto` |
| Resume from explicit checkpoint | `python train.py --resume path/to/ckpt.ckpt` |
| Evaluate low-light only | `python test.py --mode 0` |
| Evaluate deraining only | `python test.py --mode 1` |
| Evaluate dehazing only | `python test.py --mode 2` |
| Evaluate all-in-one | `python test.py --mode 3` |
| Inference on a folder | `python demo.py --test_path ./test/demo/ --output_path ./output/demo/` |
| Inference with tiling (for large images) | `python demo.py --test_path X --output_path Y --tile True --tile_size 128 --tile_overlap 32` |

`train.py` hard-codes `accelerator="gpu"` + `strategy="ddp_find_unused_parameters_true"`. For single-GPU runs override with `--num_gpus 1`. Training defaults to W&B logging under project `promptir`; pass `--wblogger None` to fall back to TensorBoard under `logs/`.

`--resume` accepts the literal string `"auto"` (picks up `train_ckpt/last.ckpt` if it exists, otherwise starts fresh) or an explicit path. Restores optimizer, scheduler, epoch, and global_step.

`test.py` and `demo.py` call `torch.cuda.set_device(...)` / `.cuda()` unconditionally — CPU-only runs will fail in `test.py` (only `demo.py` respects `torch.cuda.is_available()`).

Checkpoint location is hard-coded: `test.py` and `demo.py` load `ckpt/<--ckpt_name>` (default `model.ckpt`). For evaluation-only flows, place weights at `ckpt/model.ckpt`. Training writes to `--ckpt_dir` (default `train_ckpt/`) — the `ModelCheckpoint` callback saves `last.ckpt` plus a `best-{epoch}-{val_psnr}.ckpt` selected by `val_psnr` when validation is active.

## Architecture (big picture)

Pipeline: `PromptTrainDataset` → `PromptIR` (wrapped in a `pl.LightningModule` called `PromptIRModel` — **the exact same class is redeclared in `train.py`, `test.py`, and `demo.py`**; any interface change must be made in all three). Loss is plain L1 against the clean patch.

### `net/model.py` — `PromptIR`
A 4-level encoder/decoder of Restormer `TransformerBlock`s (MDTA self-attention + GDFN feed-forward). Base `dim=48`, `num_blocks=[4,6,6,8]`, `heads=[1,2,4,8]`, `ffn_expansion_factor=2.66`. Downsampling uses `PixelUnshuffle`, upsampling uses `PixelShuffle`. Output is residual (`output(...) + inp_img`).

The prompt mechanism activates only when `decoder=True` (always True in this repo). Three `PromptGenBlock`s sit between the latent and each decoder level. Each block:
1. Holds a learnable tensor `prompt_param` of shape `(1, prompt_len=5, prompt_dim, prompt_size, prompt_size)`.
2. Pools incoming features to a global vector, projects to `prompt_len` weights via softmax, and takes a weighted sum over the 5 prompts.
3. Bilinearly resizes the selected prompt to the feature map's HxW and applies a 3×3 conv.
4. The prompt is concatenated channel-wise, passed through a `noise_level*` TransformerBlock, then a 1×1 conv (`reduce_noise_level*`) squeezes back to the decoder channel width.

Prompt dimensions per level: `prompt1` (dim=64, size=64), `prompt2` (128, 32), `prompt3` (320, 16). Mismatching these with the `reduce_noise_level*` / `noise_level*` Conv/Transformer widths will silently break shape math — channel arithmetic in `__init__` is fragile.

### `utils/dataset_utils.py`
- `PromptTrainDataset`: de-types are now integer-coded `{lowlight: 0, derain: 3, dehaze: 4}` (the three legacy denoise IDs `{15:0, 25:1, 50:2}` are gone). The constant `LOWLIGHT_DE_ID = 0` lives at module top. Each sample loads its degraded image directly from disk (no on-the-fly noise synthesis):
  - **lowlight** → reads from `data_dir/lowlight/lowlight_train.txt` under `--lowlight_dir`. GT is resolved by `_lowlight_gt_path`, which handles both LOL-v2 conventions: `Real_captured` (`Low/lowNNNNN.png` ↔ `Normal/normalNNNNN.png`) and `Synthetic` (identical basenames in `Low/` and `Normal/`).
  - **derain** → manifest `rainy/rainTrain.txt` under `--derain_dir`; GT via `_get_gt_name`. IDs replicated **120×**.
  - **dehaze** → manifest `hazy/hazy_outside.txt` under `--dehaze_dir`; GT via `_get_nonhazy_name`.
  - Lowlight IDs are **not** replicated — the comment in `_init_lowlight_ids` mentions ~50× balancing but the actual line `self.ll_ids = self.ll_ids` is a no-op. With ~1.6k LOL pairs vs. ~120k rain and ~72k haze samples per epoch, lowlight is heavily under-represented; if balance matters, multiply explicitly there.
- `LowLightTestDataset`: paired evaluator. Expects `<root>/Low/*.{png,jpg,jpeg,bmp}` inputs and corresponding `<root>/Normal/...` GTs. Used at both train-time validation and `test.py`.
- `DerainDehazeDataset`: expects paired `input/` + `target/` subdirs per task. `set_dataset(task)` switches between `derain` and `dehaze`.
- `TestSpecificDataset`: for `demo.py`, walks an arbitrary folder or single file.

The legacy `DenoiseTestDataset` and the `Degradation.single_degrade` noise-synthesis path are gone.

All images are cropped to multiples of 16 (`crop_img(..., base=16)`); the transformer strides require HxW divisible by 8. `demo.py`'s `pad_input` enforces this with reflect padding.

### `data_dir/` vs `data/`
`data_dir/` (checked in) holds **text manifests** that list which image filenames belong to each split: `lowlight/lowlight_train.txt`, `rainy/rainTrain.txt`, `hazy/hazy_outside.txt`. (`noisy/denoise.txt` is leftover from the original repo and unused.) Actual pixel data must live under `data/Train/{LowLight,Derain,Dehaze}/` (not checked in — see `INSTALL.md` for dataset download links and expected subdirectory layout). Don't confuse the two.

### Validation pipeline (`train.py`)
`PromptIRModel` now defines `validation_step` and `on_validation_epoch_end`. `train.py:main` builds a list of per-task val loaders gated by **both** `(a)` the task being in `--de_type` and `(b)` the test data existing on disk under `--lowlight_test_path`/`--derain_test_path`/`--dehaze_test_path`. Each loader runs at `batch_size=4, num_workers=0`. Per-task PSNR/SSIM are logged as `val_psnr_<task>` / `val_ssim_<task>`; their mean is logged as `val_psnr` / `val_ssim`, and `ModelCheckpoint(monitor="val_psnr", mode="max")` selects the best checkpoint. If no val data is found, training runs without validation and `save_top_k=-1` (keep all). Validation cadence is set by `--val_every_n_epochs` (default 1).

For low-light validation, `train.py` iterates over `Real_captured/` and `Synthetic/` subdirs of `--lowlight_test_path` and concatenates them into one loader.

### Paths configured in `options.py`
- Manifests: `--data_file_dir` = `data_dir/`
- Train pixel roots: `--lowlight_dir`, `--derain_dir`, `--dehaze_dir` (under `data/Train/...`)
- Val/test pixel roots: `--lowlight_test_path`, `--derain_test_path`, `--dehaze_test_path` (under `test/...`)
- Outputs: `--output_path` (test/demo PNG dumps), `--ckpt_dir` (training checkpoints, default `train_ckpt`)
- `--ckpt_path` exists in `options.py` (default `ckpt/Denoise/`) but is unused by `train.py`/`test.py` — vestigial.

## Gotchas

- The three copies of `PromptIRModel` (in `train.py`, `test.py`, `demo.py`) must stay in sync. `test.py` and `demo.py` define stripped-down copies; both reference `optim` and `LinearWarmupCosineAnnealingLR` inside `configure_optimizers` without importing them, but it's harmless because eval never calls that method. Only `train.py`'s copy implements `validation_step` / `on_validation_epoch_end` — don't expect those hooks elsewhere.
- `utils/val_utils.py` keeps `compute_niqe` commented out (the `skvideo` dep is not installed); do not call it. Only `compute_psnr_ssim` is wired up.
- `train.py` and `test.py` use `subprocess.check_output(['mkdir', '-p', ...])` — Linux/macOS only; on Windows it will fail.
- `test.py --mode 0` evaluates **both** LOL-v2 subsets (`Real_captured` then `Synthetic`) in sequence, printing per-subset PSNR/SSIM and dumping outputs to `output/lowlight/<split>/`. Mode 3 runs lowlight (both splits) → derain (Rain100L) → dehaze (SOTS).
- Default `--de_type` is `['lowlight', 'derain', 'dehaze']` — passing `--de_type denoise ...` will raise `ValueError("Unknown de_type id: ...")` in `PromptTrainDataset.__getitem__`.
