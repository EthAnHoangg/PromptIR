import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from utils.dataset_utils import LowLightTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)


def test_LowLight(net, dataset, split_name=""):
    output_path = testopt.output_path + 'lowlight/' + split_name + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("LowLight [{}]: psnr: {:.2f}, ssim: {:.4f}".format(split_name, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for lowlight, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--lowlight_path', type=str, default="test/lowlight/",
                        help='root of low-light test data; expects <split>/Low and <split>/Normal subdirs')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    # LOL-v2 ships two subsets; evaluate each separately and report per-subset metrics.
    lowlight_splits = ["Real_captured", "Synthetic"]
    derain_splits = ["Rain100L/"]

    lowlight_tests = []
    base_ll_path = testopt.lowlight_path
    for split in lowlight_splits:
        testopt.lowlight_path = os.path.join(base_ll_path, split)
        if os.path.isdir(os.path.join(testopt.lowlight_path, "Low")):
            lowlight_tests.append((split, LowLightTestDataset(testopt)))
    testopt.lowlight_path = base_ll_path

    print("CKPT name : {}".format(ckpt_path))

    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.mode == 0:
        for split, testset in lowlight_tests:
            print('Start testing LowLight [{}]...'.format(split))
            test_LowLight(net, testset, split_name=split)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="dehaze")
    elif testopt.mode == 3:
        for split, testset in lowlight_tests:
            print('Start testing LowLight [{}]...'.format(split))
            test_LowLight(net, testset, split_name=split)

        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")
