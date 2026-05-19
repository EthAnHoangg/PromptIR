import os
import argparse
import subprocess
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from utils.dataset_utils import PromptTrainDataset, LowLightTestDataset, DerainDehazeDataset
from utils.val_utils import compute_psnr_ssim
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        [_], degrad, clean = batch
        restored = self.net(degrad)
        psnr, ssim, _ = compute_psnr_ssim(restored, clean)
        # val batch_size=1 so per-batch value == per-image value; Lightning averages
        # across batches via on_epoch=True to yield the true mean over each val loader.
        task = self.val_task_names[dataloader_idx]
        self.log(f"val_psnr_{task}", psnr, on_epoch=True, on_step=False,
                 add_dataloader_idx=False, sync_dist=True)
        self.log(f"val_ssim_{task}", ssim, on_epoch=True, on_step=False,
                 add_dataloader_idx=False, sync_dist=True)

    def on_validation_epoch_end(self):
        # Aggregate per-task metrics into a single mean `val_psnr`/`val_ssim` that the
        # ModelCheckpoint callback monitors. Single-task case: mean-of-one == that value.
        m = self.trainer.callback_metrics
        psnr_vals = [float(m[k]) for k in m if k.startswith("val_psnr_") and k != "val_psnr"]
        ssim_vals = [float(m[k]) for k in m if k.startswith("val_ssim_") and k != "val_ssim"]
        if psnr_vals:
            self.log("val_psnr", sum(psnr_vals) / len(psnr_vals), prog_bar=True)
            self.log("val_ssim", sum(ssim_vals) / len(ssim_vals), prog_bar=True)

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)

    # Task-balanced sampling. Weight each sample inversely to its task's count so
    # every batch is drawn ~uniformly across tasks regardless of raw dataset sizes.
    # Default epoch length = n_tasks * max(per-task count); override via
    # --samples_per_epoch to shorten an epoch without dropping any training data
    # (the sampler is stochastic, so each epoch draws a different random subset).
    task_counts = Counter(s["de_type"] for s in trainset.sample_ids)
    sample_weights = [1.0 / task_counts[s["de_type"]] for s in trainset.sample_ids]
    if opt.samples_per_epoch > 0:
        samples_per_epoch = opt.samples_per_epoch
    else:
        samples_per_epoch = len(task_counts) * max(task_counts.values())
    sampler = WeightedRandomSampler(sample_weights, num_samples=samples_per_epoch,
                                    replacement=True)
    print("Train task counts: {}".format(dict(task_counts)))
    print("Samples per epoch (weighted): {}".format(samples_per_epoch))

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True,
                             sampler=sampler, drop_last=True, num_workers=opt.num_workers)

    # Build per-task val loaders. A loader is built only if BOTH (a) the task is in
    # --de_type (so the model was actually trained on it) AND (b) the test data is on
    # disk. Evaluating on untrained tasks would produce garbage PSNR that drags down
    # the mean `val_psnr` used for best-checkpoint selection.
    val_dataloaders, val_task_names = [], []

    if 'lowlight' in opt.de_type:
        lowlight_subsets = []
        for split in ['Real_captured', 'Synthetic']:
            split_root = os.path.join(opt.lowlight_test_path, split)
            if os.path.isdir(os.path.join(split_root, 'Low')):
                lowlight_subsets.append(LowLightTestDataset(
                    argparse.Namespace(lowlight_path=split_root)
                ))
        if lowlight_subsets:
            ll_set = ConcatDataset(lowlight_subsets) if len(lowlight_subsets) > 1 else lowlight_subsets[0]
            # batch_size=1: val images come at native resolution with mixed shapes
            # (LOL-v2 Real 600x400 vs Synthetic 384x384, plus orientation mismatches),
            # which break the default stack-based collate at any larger batch size.
            val_dataloaders.append(DataLoader(ll_set, batch_size=1, num_workers=0,
                                              pin_memory=True, shuffle=False))
            val_task_names.append('lowlight')
            print("LowLight val: {} pairs".format(len(ll_set)))

    if 'derain' in opt.de_type:
        derain_split_root = os.path.join(opt.derain_test_path, 'Rain100L/')
        if os.path.isdir(os.path.join(derain_split_root, 'input')):
            derain_val_args = argparse.Namespace(
                derain_path=derain_split_root,
                dehaze_path=opt.dehaze_test_path,
            )
            derain_set = DerainDehazeDataset(derain_val_args, task='derain',
                                             addnoise=False, sigma=15)
            # batch_size=1: Rain100L mixes landscape/portrait orientations.
            val_dataloaders.append(DataLoader(derain_set, batch_size=1, num_workers=0,
                                              pin_memory=True, shuffle=False))
            val_task_names.append('derain')
            print("Derain val: {} pairs".format(len(derain_set)))

    if 'dehaze' in opt.de_type:
        if os.path.isdir(os.path.join(opt.dehaze_test_path, 'input')):
            dehaze_val_args = argparse.Namespace(
                derain_path=opt.derain_test_path,
                dehaze_path=opt.dehaze_test_path,
            )
            dehaze_set = DerainDehazeDataset(dehaze_val_args, task='dehaze',
                                             addnoise=False, sigma=15)
            # batch_size=1: SOTS dehaze test set has mixed image dimensions.
            val_dataloaders.append(DataLoader(dehaze_set, batch_size=1, num_workers=0,
                                              pin_memory=True, shuffle=False))
            val_task_names.append('dehaze')
            print("Dehaze val: {} pairs".format(len(dehaze_set)))

    if not val_dataloaders:
        val_dataloaders = None
        print("No validation data found - training without validation.")
    else:
        print("Validation tasks: {}".format(val_task_names))

    has_val = val_dataloaders is not None
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        monitor="val_psnr" if has_val else None,
        mode="max",
        save_top_k=1 if has_val else -1,
        save_last=True,
        filename="best-{epoch:03d}-{val_psnr:.3f}" if has_val else None,
    )

    model = PromptIRModel()
    model.val_task_names = val_task_names

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=opt.val_every_n_epochs,
    )

    # Resolve --resume. "auto" -> train_ckpt/last.ckpt if present; else treat as an explicit path.
    resume_path = opt.resume
    if resume_path == "auto":
        auto_path = os.path.join(opt.ckpt_dir, "last.ckpt")
        resume_path = auto_path if os.path.exists(auto_path) else None
        print("--resume auto: {}".format(
            "resuming from " + auto_path if resume_path else "no last.ckpt found, starting fresh"))
    elif resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError("--resume path does not exist: {}".format(resume_path))
        print("Resuming from checkpoint: {}".format(resume_path))

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=val_dataloaders,
                ckpt_path=resume_path)


if __name__ == '__main__':
    main()



