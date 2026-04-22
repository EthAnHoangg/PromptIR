# Low-Light Migration — Change Log

Replaces the three denoise degradations (σ=15/25/50) with **Low-Light Enhancement** while keeping Derain and Dehaze. Task set is now `{lowlight, derain, dehaze}`.

The PromptIR architecture (`net/model.py`) was **not modified** — prompt blocks are degradation-agnostic and learn their specialization from the data mix. All changes are in config, data pipeline, and eval.

---

## 1. Dataset: LOL-v2

Source: `~/Desktop/devzone/UTS/CNN/data/LOL-v2/` (not in repo).

Copied into the project as:

```
data/Train/LowLight/
    Real_captured/Low/        689 images  (lowNNNNN.png)
    Real_captured/Normal/     689 images  (normalNNNNN.png)
    Synthetic/Low/            900 images  (rXXXt.png)
    Synthetic/Normal/         900 images  (rXXXt.png — same basename)
test/lowlight/
    Real_captured/{Low,Normal}/   100 pairs
    Synthetic/{Low,Normal}/       100 pairs
data_dir/lowlight/lowlight_train.txt     1,589-line manifest
```

**Training totals**: 1,589 paired images (Real 689 + Synthetic 900). **Test totals**: 200 pairs.

Manifest entries are paths **relative to `--lowlight_dir`** (which defaults to `data/Train/LowLight/`), e.g. `Real_captured/Low/low00001.png`. Only the `Low/` inputs are listed; GT paths are derived.

### Two naming conventions, one helper

`utils/dataset_utils.py` defines `_lowlight_gt_path(low_path)` which handles both:

| Subset | Low filename | Normal (GT) filename |
|---|---|---|
| Real_captured | `lowNNNNN.png` | `normalNNNNN.png` |
| Synthetic | `rXXXt.png` | `rXXXt.png` (same) |

Rule: replace `/Low/` with `/Normal/`; if the basename starts with `low`, swap the prefix to `normal`.

### `.gitignore`

The 1 GB copy is excluded from git. Added at the top of `.gitignore`:

```
data/Train/
test/
output/
train_ckpt/
logs/
wandb/
ckpt/*.ckpt
```

---

## 2. Code changes

### `options.py`

- Removed `--denoise_dir`.
- Added `--lowlight_dir` (default `data/Train/LowLight/`).
- Changed `--de_type` default from `['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze']` to `['lowlight', 'derain', 'dehaze']`.

### `utils/dataset_utils.py`

- Dropped `from utils.degradation_utils import Degradation` and the `self.D` field — no synthetic degradation needed anymore (low-light is paired supervised).
- Dropped the `import torch` at module top (now unused).
- **`de_dict`** is now `{'lowlight': 0, 'derain': 3, 'dehaze': 4}`. Codes 3 and 4 are preserved so the derain/dehaze branches in `__getitem__` keep working unchanged.
- Added module-level constant `LOWLIGHT_DE_ID = 0` and helper `_lowlight_gt_path(low_path)`.
- Replaced `_init_clean_ids` with `_init_lowlight_ids`:
  - Reads `data_dir/lowlight/lowlight_train.txt`.
  - Builds `self.ll_ids = [{"clean_id": ..., "de_type": 0}, ...]`.
  - **Replicates ×50** → ~79k samples/epoch. This balances against rain ×120 (~86k) and haze ×1 (~72k). If low-light underperforms, raise this multiplier.
- Updated `_init_ids()` and `_merge_ids()` to branch on `'lowlight'` instead of the three denoise codes.
- Rewrote `__getitem__`:
  - Removed the `de_id < 3` synthetic-noise branch entirely.
  - Added a `de_id == LOWLIGHT_DE_ID` branch that loads paired Low/Normal images, crops, augments — mirrors the derain/dehaze flow.
- **Removed `DenoiseTestDataset` and its dead `tile_degrad` method** (the latter had an undefined `restored` reference).
- **Added `LowLightTestDataset`** for evaluation:
  - Takes `args.lowlight_path` (a root with `Low/` and `Normal/` subdirs).
  - Iterates `Low/*.png`, resolves GT via `_lowlight_gt_path`.
  - Returns `([name], degraded_img, clean_img)` — same tuple shape as `DerainDehazeDataset`.

### `test.py`

Full rewrite. Key changes:

- **`--mode` remapping**: `0=lowlight, 1=derain, 2=dehaze, 3=all`.
- New `--lowlight_path` arg (default `test/lowlight/`). Removed `--denoise_path`.
- New `test_LowLight(net, dataset, split_name)` function.
- Removed the three `test_Denoise(..., sigma=15/25/50)` calls from modes 0 and 3.
- Evaluates **both LOL-v2 subsets separately** (Real_captured, Synthetic) — walks `lowlight_splits = ["Real_captured", "Synthetic"]` and reports per-split PSNR/SSIM.
- Outputs go to `output/lowlight/<split>/<name>.png`.
- **Pre-existing bug fixed**: line 160 used undefined `opt` — changed to `testopt` inside `DerainDehazeDataset(...)`.
- Slimmed the Lightning wrapper: the old `PromptIRModel` copy referenced unimported `optim` / `LinearWarmupCosineAnnealingLR` inside `configure_optimizers`; since `configure_optimizers` is never called during eval, I kept only `__init__` and `forward` in the test-side wrapper.

### Not modified

- `net/model.py` — architecture unchanged; prompt params re-specialize via gradient descent.
- `train.py` — works as-is with the new defaults. The `PromptIRModel` class there still holds the real `configure_optimizers` with the AdamW + LinearWarmupCosineAnnealingLR setup.
- `demo.py` — task-agnostic inference, works unchanged.
- `utils/degradation_utils.py` — still present but no longer imported; safe to delete later.
- `utils/image_utils.py`, `utils/val_utils.py`, `utils/image_io.py`, `utils/schedulers.py`, `utils/loss_utils.py`, `utils/pytorch_ssim/` — untouched.

---

## 3. How to use

```bash
conda activate promptir
```

### Train from scratch (single GPU)

```bash
python train.py --num_gpus 1
# or explicitly: python train.py --de_type lowlight derain dehaze --num_gpus 1
```

### Train with **warm-start from the official pretrained checkpoint**

The checkpoint at `ckpt/model.ckpt` is structurally compatible — same architecture, identical tensor names and shapes. What's obsolete is the *semantic* content of the prompt params (encoded noise/rain/haze signatures). It still makes a good warm start because the encoder/decoder transformer blocks carry general restoration priors.

Add to `train.py` right after `model = PromptIRModel()`:

```python
import torch
ckpt = torch.load("ckpt/model.ckpt", map_location="cpu")
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Warm-start loaded. missing={len(missing)} unexpected={len(unexpected)}")
```

Do **not** pass `ckpt_path=` to `trainer.fit(...)` — that would resume optimizer + scheduler state (i.e., from epoch 120 with LR=0), not what we want.

### Train only low-light (debug run)

```bash
python train.py --de_type lowlight --num_gpus 1
```

This skips derain/dehaze dataset loading — useful when Derain/Dehaze data isn't on disk yet.

### Evaluate

```bash
python test.py --mode 0                  # lowlight (both Real + Synthetic splits)
python test.py --mode 1                  # derain
python test.py --mode 2                  # dehaze
python test.py --mode 3                  # all three
```

### Demo / single-image inference

Unchanged:
```bash
python demo.py --test_path ./test/demo/ --output_path ./output/demo/
```

---

## 4. Design decisions & invariants

### Why replication factor ×50 for low-light

Mixed-task training needs roughly balanced per-task sample counts per epoch or the prompt softmax under-fits the rare task. Target ~80k samples per task.

| Task | Unique pairs | Multiplier | Samples / epoch |
|---|---|---|---|
| lowlight | 1,589 | ×50 | 79,450 |
| derain   | ~720 | ×120 | ~86,000 |
| dehaze   | 72,135 | ×1 | 72,135 |

If low-light PSNR plateaus low, raise the ×50. If it overfits (train PSNR ≫ val PSNR), lower it.

### Why codes 0, 3, 4 (not 0, 1, 2)

`DerainDehazeDataset.task_dict` and the derain/dehaze branches in `PromptTrainDataset.__getitem__` key off explicit integer IDs. Keeping `derain=3, dehaze=4` means those code paths work unchanged. Low-light gets the freed slot 0.

### Patch size vs LOL-v2 image size

LOL-v2 images are ~400×600. With `--patch_size 128` you get ~3×4 possible crop origins per image, which is tight. If you see low-light convergence struggling, try `--patch_size 192` — still divides by 8 (transformer stride requirement), still fits in memory for a 48-base-dim model.

### Only the test-side `PromptIRModel` was slimmed

The Lightning wrapper is **duplicated in three files** (`train.py`, `test.py`, `demo.py`) — historical code smell from the upstream repo. Any change to the wrapper must be made in all three. I only touched `test.py`'s copy (removed the unused `training_step` and `configure_optimizers` that referenced unimported names); the train-side copy must keep `configure_optimizers` intact because Lightning actually calls it.

---

## 5. Known limitations / TODOs

- `train.py` uses `subprocess.check_output(['mkdir', '-p', ...])` — POSIX-only. Fine on macOS/Linux, breaks on Windows. Not fixed here.
- `utils/degradation_utils.py` is now dead code. Safe to delete in a follow-up.
- No SSIM / perceptual loss added — plain L1. Low-light often benefits from `L1 + λ·(1-SSIM)` or VGG perceptual. Add to `PromptIRModel.training_step` in `train.py` if first training run plateaus.
- The `ckpt/` directory still contains the original `model.ckpt`. After first successful run from scratch (or warm-started fine-tune), replace with your own or load from `train_ckpt/`.
- `test.py` hard-codes `torch.cuda.set_device(...)` and `.cuda()` — CPU-only eval will fail. Only `demo.py` respects `torch.cuda.is_available()`.

---

## 6. Files touched in this migration

| File | Change |
|---|---|
| `options.py` | `--de_type` default, `--lowlight_dir` added, `--denoise_dir` removed |
| `utils/dataset_utils.py` | `PromptTrainDataset` rewritten for lowlight; `LowLightTestDataset` added; `DenoiseTestDataset` removed |
| `test.py` | Full rewrite: mode remap, lowlight eval path, bug fix |
| `.gitignore` | Exclude data/test/output/ckpt artifacts |
| `data_dir/lowlight/lowlight_train.txt` | New — 1,589-line manifest |
| `data/Train/LowLight/**` | New — 1,589 train pairs (1 GB) |
| `test/lowlight/**` | New — 200 test pairs |
| `docs/lowlight_migration.md` | This file |

No changes to: `net/model.py`, `train.py`, `demo.py`, `env.yml`, anything else under `utils/`.
