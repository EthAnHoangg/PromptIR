import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['lowlight', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='directory holding per-task filename manifests.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/LowLight/',
                    help='root of low-light training data (expects Low/ and Normal/ subtrees).')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--lowlight_test_path', type=str, default='test/lowlight/',
                    help='root of low-light validation data; contains Real_captured/ and Synthetic/ subdirs')
parser.add_argument('--derain_test_path', type=str, default='test/derain/',
                    help='root of derain validation data; expects Rain100L/{input,target} underneath')
parser.add_argument('--dehaze_test_path', type=str, default='test/dehaze/',
                    help='root of dehaze validation data; expects input/ and target/ underneath')
parser.add_argument('--val_every_n_epochs', type=int, default=1,
                    help='run validation every N epochs')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint to resume training from. Use "auto" to pick up '
                         'train_ckpt/last.ckpt if it exists, or pass an explicit path. '
                         'Restores optimizer, scheduler, epoch, and global_step.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")

options = parser.parse_args()

