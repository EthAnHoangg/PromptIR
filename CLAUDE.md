# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PromptIR (NeurIPS'23) — a single all-in-one blind image restoration network that handles denoising, deraining, and dehazing using learnable prompt blocks injected into a Restormer-style transformer UNet. Based on AirNet and Restormer.

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
| Train (defaults: 120 epochs, 4 GPUs DDP, all degradations) | `python train.py` |
| Train a subset | `python train.py --de_type derain dehaze` |
| Evaluate denoising only | `python test.py --mode 0` |
| Evaluate deraining only | `python test.py --mode 1` |
| Evaluate dehazing only | `python test.py --mode 2` |
| Evaluate all-in-one | `python test.py --mode 3` |
| Inference on a folder | `python demo.py --test_path ./test/demo/ --output_path ./output/demo/` |
| Inference with tiling (for large images) | `python demo.py --test_path X --output_path Y --tile True --tile_size 128 --tile_overlap 32` |

`train.py` hard-codes `accelerator="gpu"` + `strategy="ddp_find_unused_parameters_true"`. For single-GPU runs override with `--num_gpus 1`. Training defaults to W&B logging under project `promptir`; pass `--wblogger None` to fall back to TensorBoard under `logs/`.

`test.py` and `demo.py` call `torch.cuda.set_device(...)` / `.cuda()` unconditionally — CPU-only runs will fail in `test.py` (only `demo.py` respects `torch.cuda.is_available()`).

Checkpoint location is hard-coded: `test.py` and `demo.py` load `ckpt/<--ckpt_name>` (default `model.ckpt`). Download the pretrained weights from the repo's releases tab (see `ckpt/README.md`) and place at `ckpt/model.ckpt`.

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
- `PromptTrainDataset`: de-types are integer-coded `{denoise_15:0, denoise_25:1, denoise_50:2, derain:3, dehaze:4}`. For denoise tasks it loads clean images listed in `data_dir/noisy/denoise_airnet.txt` and adds Gaussian noise at training time via `Degradation.single_degrade`. For derain/dehaze it loads the degraded image directly and derives the GT filename (`_get_gt_name`, `_get_nonhazy_name`). Denoise IDs are replicated 3× and rain IDs 120× so the sampler sees a balanced mix across the three supervisions — changing this ratio changes training dynamics significantly.
- `DenoiseTestDataset`: synthesizes noise at eval time via `set_sigma(...)`, tested at σ ∈ {15, 25, 50}.
- `DerainDehazeDataset`: expects paired `input/` + `target/` subdirs per task.
- `TestSpecificDataset`: for `demo.py`, walks an arbitrary folder or single file.

All images are cropped to multiples of 16 (`crop_img(..., base=16)`); the transformer strides require HxW divisible by 8. `demo.py`'s `pad_input` enforces this with reflect padding.

### `data_dir/` vs `data/`
`data_dir/` (checked in) holds **text manifests** that list which image filenames belong to each split: `noisy/denoise_airnet.txt`, `rainy/rainTrain.txt`, `hazy/hazy_outside.txt`. Actual pixel data must live under `data/Train/{Denoise,Derain,Dehaze}/` (not checked in — see `INSTALL.md` for dataset download links and expected subdirectory layout). Don't confuse the two.

### Paths configured in `options.py`
`--data_file_dir` = `data_dir/` (manifests), `--denoise_dir` / `--derain_dir` / `--dehaze_dir` = the pixel data roots, `--ckpt_dir` = where training writes checkpoints (default `train_ckpt`), `--output_path` = where test/demo dumps restored PNGs.

## Gotchas

- The three copies of `PromptIRModel` (in `train.py`, `test.py`, `demo.py`) must stay in sync. `test.py`'s and `demo.py`'s copies reference `optim` and `LinearWarmupCosineAnnealingLR` that are not imported there — they only work because `configure_optimizers` is never called during eval.
- `test.py` mode 1 uses an undefined `opt` (line 160, inside `DerainDehazeDataset(opt, ...)` should be `testopt`) — known bug.
- `DenoiseTestDataset.tile_degrad` is dead code with an undefined `restored` reference.
- `train.py` uses `subprocess.check_output(['mkdir', '-p', ...])` — Linux-only; on macOS `/bin/mkdir` supports `-p` so it works, but on Windows it will fail.
