import os
import torch
import sys
sys.path.append("../")
from configs.test_model_args import obtain_test_args
from engine.retrainer import Trainer as retrainer
from torchstat import stat
# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def main():
    args = obtain_test_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    # init model
    trainer = retrainer(args)
    
    # test model inference
    trainer.test_model(0)
    
    # test model para, flops and memory
    device = torch.device("cpu")
    model = trainer.model.to(device)
    stat(model, (4, 512, 512))
    

if __name__ == "__main__":
    main()
