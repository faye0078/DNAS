import argparse
import numpy as np

def obtain_test_args():
    parser = argparse.ArgumentParser(description="ReTrain the nas model")

    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--use_default', type=bool, default=False, help='if use the default arch')
    parser.add_argument('--use_low', type=bool, default=True, help='if use the low level features')
    parser.add_argument('--model_name', type=str, default='hrnet', choices=['one_path', 'multi', 'hrnet', 'flexinet', 'deeplabv3plus', 'pspnet', 'unet', 'refinenet', 'fast-nas', 'MACUNet', 'MAResUNet', 'MSFCN'],  help='the model name')
    parser.add_argument('--checkname', type=str, default='test', help='set the checkpoint name')
    parser.add_argument('--nclass', type=int, default=4, help='number of class')

    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    parser.add_argument('--initial-fm', default=None, type=int)


    parser.add_argument('--net_arch', type=str, default='/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_forward/path.npy')
    parser.add_argument('--cell_arch', type=str, default='/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_forward/cell.npy')

    parser.add_argument('--opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'], help='opt level for half percision training (default: O0)')

    parser.add_argument('--dataset', type=str, default='GID', choices=['cityscapes', 'GID', 'hps-GID', 'uadataset', 'uadataset_dfc'], help='dataset name (default: pascal)')
    parser.add_argument('--nas', type=str, default='train', choices=['search', 'train'])

    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=320, help='base image size')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--use_ABN', type=bool, default=False, help='whether to use abn (default: False)')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
    NORMALISE_PARAMS = [
                        1.0 / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
                        np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
                        ]
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")

    parser.add_argument("--dist", type=bool, default=False)

    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--filter_multiplier', type=int, default=32)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--alpha_epoch', type=int, default=20,metavar='N', help='epoch to start training alphas')

    parser.add_argument('--num_worker', type=int, default=4,metavar='N', help='numer workers')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: auto)')

    parser.add_argument('--lr', type=float, default=0.025, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR', help='learning rate for alpha and beta in architect searching process')
    parser.add_argument('--lr-scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--val', action='store_true', default=True, help='skip validation during training')

    args = parser.parse_args()
    return args
