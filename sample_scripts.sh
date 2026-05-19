#!/usr/bin/env bash
# Sample training commands for PromptIR (lowlight + derain + dehaze).
# Pick one of the blocks below and run it; do NOT execute the file as-is.

set -euo pipefail


# ---------------------------------------------------------------------------
# 1. All-in-one training, single GPU (matches the current single-GPU setup).
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 8 \
    --epochs 120 \
    --de_type lowlight derain dehaze


# ---------------------------------------------------------------------------
# 2. Resume from train_ckpt/last.ckpt if it exists, else start fresh.
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 8 \
    --resume auto


# ---------------------------------------------------------------------------
# 3. Resume from an explicit checkpoint (restores optimizer/scheduler/epoch).
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 8 \
    --resume train_ckpt/best-099-25.123.ckpt


# ---------------------------------------------------------------------------
# 4. Single-task training: lowlight only.
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 8 \
    --de_type lowlight \
    --ckpt_dir train_ckpt_lowlight


# ---------------------------------------------------------------------------
# 5. Two-task training: derain + dehaze (skip lowlight).
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 8 \
    --de_type derain dehaze \
    --ckpt_dir train_ckpt_rain_haze


# ---------------------------------------------------------------------------
# 6. Disable Weights & Biases, log to TensorBoard under logs/ instead.
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 8 \
    --wblogger None


# ---------------------------------------------------------------------------
# 7. Fast smoke test: tiny patch, few epochs, no W&B. Good for verifying the
#    pipeline runs end-to-end after dataset / sampler / model changes.
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --batch_size 2 \
    --epochs 2 \
    --patch_size 64 \
    --num_workers 2 \
    --wblogger None \
    --val_every_n_epochs 1 \
    --ckpt_dir train_ckpt_smoke


# ---------------------------------------------------------------------------
# 8. Custom data paths (if your data does not live in the defaults).
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 1 \
    --lowlight_dir data/Train/LowLight/ \
    --derain_dir   data/Train/Derain/ \
    --dehaze_dir   data/Train/Dehaze/ \
    --lowlight_test_path test/lowlight/ \
    --derain_test_path   test/derain/ \
    --dehaze_test_path   test/dehaze/


# ---------------------------------------------------------------------------
# 9. Multi-GPU (DDP) — use only when 4 GPUs are actually available.
# ---------------------------------------------------------------------------
python train.py \
    --num_gpus 4 \
    --batch_size 8 \
    --epochs 120 \
    --de_type lowlight derain dehaze
