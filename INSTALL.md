# Installation

### Dependencies Installation

This repository is built in PyTorch 1.8.1 (Python 3.8, CUDA 11.6, cuDNN 8.5, PyTorch Lightning 2.0.1).
Follow these instructions:

1. Clone the repository
```
git clone https://github.com/va1shn9v/PromptIR.git
cd PromptIR
```

2. Create the conda environment from `env.yml`
```
conda env create -f env.yml
conda activate promptir
```


### Dataset Download and Preparation

This fork trains and evaluates on three paired restoration tasks: **low-light enhancement, deraining, and dehazing**. The original Gaussian-denoise task has been removed.

| Task | Dataset | Download |
|---|---|---|
| Low-light | LOL-v2 (Real_captured + Synthetic) | [LOL-v2](https://daooshee.github.io/BMVC2018website/) |
| Deraining | Train100L / Rain100L | [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing) |
| Dehazing | RESIDE OTS (train) / SOTS (test) | [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) |

#### Training data layout

Place training images under `data/Train/{LowLight,Derain,Dehaze}/`. The pixel data is **not** checked in — only the filename manifests in `data_dir/` are. The default paths (`--lowlight_dir`, `--derain_dir`, `--dehaze_dir` in `options.py`) expect:

```
data/Train
├── LowLight
│   ├── Real_captured
│   │   ├── Low      # lowNNNNN.png
│   │   └── Normal   # normalNNNNN.png
│   └── Synthetic
│       ├── Low      # rXXXt.png
│       └── Normal   # rXXXt.png  (same basename as Low)
├── Derain
│   ├── rainy        # rain-*.png (listed in data_dir/rainy/rainTrain.txt)
│   └── gt           # norain-*.png
└── Dehaze
    ├── synthetic    # hazy crops (listed in data_dir/hazy/hazy_outside.txt)
    └── original     # haze-free reference images
```

The manifests already shipped under `data_dir/` are the source of truth for which filenames are loaded:
- `data_dir/lowlight/lowlight_train.txt`
- `data_dir/rainy/rainTrain.txt`
- `data_dir/hazy/hazy_outside.txt`

(`data_dir/noisy/denoise.txt` is leftover from the original repo and unused.)

#### Test / validation data layout

Place evaluation images under `test/`. The defaults (`--lowlight_test_path`, `--derain_test_path`, `--dehaze_test_path`) expect:

```
test
├── lowlight
│   ├── Real_captured
│   │   ├── Low
│   │   └── Normal
│   └── Synthetic
│       ├── Low
│       └── Normal
├── derain
│   └── Rain100L
│       ├── input
│       └── target
└── dehaze
    ├── input
    └── target
```

`test.py --mode 0` iterates over both `Real_captured/` and `Synthetic/` and reports PSNR/SSIM per subset. Mode 1 evaluates `derain/Rain100L`, mode 2 evaluates `dehaze/`, mode 3 runs all three sequentially.
