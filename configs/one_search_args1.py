import argparse
import numpy as np

def obtain_search_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--forward', type=bool, default=True, help='if use the low level features')
    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='test', help='set the checkpoint name')
    parser.add_argument('--model_name', type=str, default='FlexiNet', choices=['AutoDeeplab', 'DCNAS', 'FlexiNet'], help='set the model name')
    parser.add_argument('--model_encode_path', type=str, default='/media/dell/DATA/wy/Seg_NAS/model/model_encode/uadataset/one_loop_14layers_mixedcell/first_connect_4.npy')
    parser.add_argument('--model_cell_arch', type=str, default='/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_1/cell_arch/1_cell_arch_epoch13.npy', help='the initial cell arch')

    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'], help='backbone name (default: resnet)')
    parser.add_argument('--opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'], help='opt level for half percision training (default: O0)')
    parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='uadataset', choices=['pascal', 'coco', 'cityscapes', 'kd', 'GID', 'hps-GID', 'GID-15', 'uadataset'], help='dataset name (default: pascal)')
    parser.add_argument('--nas', type=str, default='search', choices=['search', 'train'])
    parser.add_argument('--use-sbd', action='store_true', default=False, help='whether to use SBD dataset (default: False)')
    parser.add_argument('--load-parallel', type=int, default=0)
    parser.add_argument('--clean-module', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0,metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=320, help='base image size')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')
    parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
    NORMALISE_PARAMS = [
                        1.0 / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
                        np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # STD
                        ]

    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")
    parser.add_argument('--nclass', type=int, default=15, help='number of class')

    parser.add_argument("--dist", type=bool, default=False)
    # training hyper params
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--alpha_epoch', type=int, default=40, metavar='N', help='epoch to start training alphas')

    parser.add_argument('--num_worker', type=int, default=0, metavar='N', help='numer workers')

    parser.add_argument('--train_batch_size', type=int, default=5, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=5, metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=5, metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False, help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR', help='learning rate for alpha and beta in architect searching process')

    parser.add_argument('--lr-scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true',default=False, help='disables CUDA training')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gpu-ids', type=str, default='0',help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')


    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--val', action='store_true', default=True, help='skip validation during training')
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    parser.add_argument('--multi_scale', default=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), type=bool, help='whether use multi_scale in train')
    args = parser.parse_args()
    return args
