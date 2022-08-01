import os
import torch
import numpy as np
import random
from configs.search_args import obtain_search_args
from engine.search_trainer import Trainer

# 设置所使用的GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
    trainer = Trainer(args)

    print('Total Epoches:', trainer.args.epochs)
    start_epoch = 0 #暂时先设置为0，需要读取保存过的模型
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
