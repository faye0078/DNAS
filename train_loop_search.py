import os
import torch
import numpy as np
import random
# from configs.one_search_args import obtain_search_args
from configs.one_search_args1 import obtain_search_args
from engine.one_search_trainer import Trainer
from engine.one_search_trainer_1 import Trainer as Trainer_1

# 设置所使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 为每个卷积层搜索最适合它的卷积实现算法
# torch.backends.cudnn.benchmark=True
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # print(args)
    setup_seed(args.seed)
    trainer = Trainer_1(args)
    # trainer.training_stage1(40)
    #
    trainer.core_path = np.load('/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_1/path/2_core_path_epoch29.npy')
    trainer.cell_arch_1 = np.load('/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_1/cell_arch/1_cell_arch_epoch18.npy')
    trainer.connections_3 = np.load('/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_1/connections/2_connections_epoch37.npy')
    #
    # trainer.change_batchsize(6)
    # trainer.training_stage3(20)
    #
    # trainer.change_batchsize(12)
    # trainer.training_stage1(40)
    # trainer.change_batchsize(8)
    # trainer.training_stage2(40)
    trainer.change_batchsize(5)
    trainer.training_stage3(40)


if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
