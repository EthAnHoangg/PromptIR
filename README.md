# PromptIR: Prompting for All-in-One Blind Image Restoration (NeurIPS'23)

[Vaishnav Potlapalli](https://www.vaishnavrao.com/), [Syed Waqas Zamir](https://scholar.google.ae/citations?hl=en&user=POoai-QAAAAJ), [Salman Khan](https://salman-h-khan.github.io/) and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.13090)


<hr />

> **Abstract:** *Image restoration involves recovering a high-quality clean image from its degraded
version. Deep learning-based methods have significantly improved image restora-
tion performance, however, they have limited generalization ability to different
degradation types and levels. This restricts their real-world application since it
requires training individual models for each specific degradation and knowing the
input degradation type to apply the relevant model. We present a prompt-based
learning approach, PromptIR, for All-In-One image restoration that can effectively
restore images from various types and levels of degradation. In particular, our
method uses prompts to encode degradation-specific information, which is then
used to dynamically guide the restoration network. This allows our method to
generalize to different degradation types and levels, while still achieving state-of-
the-art results on image denoising, deraining, and dehazing. Overall, PromptIR
offers a generic and efficient plugin module with few lightweight prompts that can
be used to restore images of various types and levels of degradation with no prior
information of corruptions.* 
<hr />

## Network Architecture

<img src = "mainfig.png"> 

> **Fork note (this repository).** The original PromptIR is trained on three degradations: **Gaussian denoising, deraining, and dehazing**. This fork swaps the Gaussian-denoise task for paired **low-light enhancement** on LOL-v2 (Real_captured + Synthetic), so the task set is `{lowlight, derain, dehaze}`. All training/eval/dataset code reflects that swap. See [`docs/lowlight_migration.md`](docs/lowlight_migration.md) for the full change log.

## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

## Training

After preparing the training data in `data/Train/{LowLight,Derain,Dehaze}/`, start training with:

```
python train.py
```

By default this trains on all three degradation types (`--de_type lowlight derain dehaze`) for 120 epochs across 4 GPUs (DDP). Use `--de_type` to select a subset:

```
python train.py --de_type derain dehaze        # skip lowlight
python train.py --de_type lowlight             # lowlight only
python train.py --num_gpus 1                   # single-GPU run
```

### Validation and checkpointing

`train.py` builds per-task validation loaders for each task that is **both** in `--de_type` **and** has test data present under `test/{lowlight,derain,dehaze}/`. Each task contributes `val_psnr_<task>` and `val_ssim_<task>`; their mean is logged as `val_psnr` / `val_ssim` and monitored by `ModelCheckpoint` (`monitor="val_psnr", mode="max", save_top_k=1`). Outputs:

- `train_ckpt/best-{epoch}-{val_psnr}.ckpt` — best epoch by mean val PSNR
- `train_ckpt/last.ckpt` — most recent epoch (always saved)

If no test data is found, training runs without validation and keeps every per-epoch checkpoint (`save_top_k=-1`). Validation cadence: `--val_every_n_epochs` (default `1`).

### Resuming

```
python train.py --resume auto                  # picks up train_ckpt/last.ckpt if present
python train.py --resume path/to/ckpt.ckpt     # explicit checkpoint
```

`auto` falls back to a fresh start if `last.ckpt` is absent. Resuming restores optimizer, scheduler, epoch, and global_step.

### Logging

Default logger is Weights & Biases under project `promptir`. Pass `--wblogger None` to fall back to TensorBoard under `logs/`.

## Testing

Place the checkpoint at `ckpt/<--ckpt_name>` (default `ckpt/model.ckpt`). Pretrained weights from the upstream paper are available [here](https://drive.google.com/file/d/1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl/view?usp=sharingg), but note they were trained on Gaussian denoising rather than lowlight — for this fork, use a checkpoint trained with the swapped task set.

```
python test.py --mode {n}
```

| `--mode` | Task |
|---|---|
| `0` | Low-light enhancement (iterates both LOL-v2 `Real_captured` and `Synthetic` subsets, reports per-subset PSNR/SSIM) |
| `1` | Deraining (Rain100L) |
| `2` | Dehazing (SOTS) |
| `3` | All three, in sequence |

Restored images are written to `output/<task>/...`.

## Demo
To obtain visual results from the model ```demo.py``` can be used. After placing the saved model file in ```ckpt``` directory, run:
```
python demo.py --test_path {path_to_degraded_images} --output_path {save_images_here}
```
Example usage to run inference on a directory of images:
```
python demo.py --test_path './test/demo/' --output_path './output/demo/'
```
Example usage to run inference on an image directly:
```
python demo.py --test_path './test/demo/image.png' --output_path './output/demo/'
```
To use tiling option while running ```demo.py``` set ```--tile``` option to ```True```. The Tile size and Tile overlap parameters can be adjusted using ```--tile_size``` and ```--tile_overlap``` options respectively.




## Results
Performance results of the PromptIR framework trained under the all-in-one setting

<summary><strong>Table</strong> </summary>

<img src = "prompt-ir-results.png"> 

<summary><strong>Visual Results</strong></summary>

The visual results of the PromptIR model evaluated under the all-in-one setting can be downloaded [here](https://drive.google.com/drive/folders/1Sm-mCL-i4OKZN7lKuCUrlMP1msYx3F6t?usp=sharing)



## Citation
If you use our work, please consider citing:

    @inproceedings{potlapalli2023promptir,
      title={PromptIR: Prompting for All-in-One Image Restoration},
      author={Potlapalli, Vaishnav and Zamir, Syed Waqas and Khan, Salman and Khan, Fahad},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023}
    }


## Contact
Should you have any questions, please contact pvaishnav2718@gmail.com


**Acknowledgment:** This code is based on the [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet) and [Restormer](https://github.com/swz30/Restormer) repositories. 

