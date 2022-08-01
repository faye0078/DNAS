import os
import torch
from ptflops import get_model_complexity_info
from thop import profile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from configs.test_model_args import obtain_test_args
from configs.retrain_args import obtain_retrain_args
from configs.search_args import obtain_search_args
from engine.retrainer import Trainer as retrainer
from engine.search_trainer import Trainer as search_trainer
from torchsummary import summary
from torchstat import stat


# 为每个卷积层搜索最适合它的卷积实现算法
# torch.backends.cudnn.benchmark=True

def main():
    args = obtain_test_args()
    args.cuda = torch.cuda.is_available()
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

    print(args)
    torch.manual_seed(args.seed)
    trainer = retrainer(args)

    device = torch.device("cpu")
    model = trainer.model.to(device)

    stat(model, (5, 512, 512))

    macs, params = get_model_complexity_info(trainer.model, (4, 512, 512), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    print("this model macs: " + macs)
    print("this model params: " + params)

    trainer.validation(0)



if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
