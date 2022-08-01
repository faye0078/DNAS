import os
import torch
import time
from ptflops import get_model_complexity_info
from thop import profile
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    print(args)
    torch.manual_seed(args.seed)
    trainer = retrainer(args)
    # trainer.training(0)
    torch.cuda.synchronize()
    start = time.time()
    trainer.predict()
    torch.cuda.synchronize()
    end = time.time()
    print((end-start)/2100)



if __name__ == "__main__":
    main()

    # args: gpu_id seed epoch dataset nas(阶段：搜索、再训练) use_amp(使用apex)
