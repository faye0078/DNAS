import torch
import time
from configs.test_model_args import obtain_test_args
from engine.retrainer import Trainer as retrainer

def main():
    args = obtain_test_args()
    args.cuda = torch.cuda.is_available()

    print(args)
    torch.manual_seed(args.seed)
    trainer = retrainer(args)
    torch.cuda.synchronize()
    start = time.time()
    trainer.predict()
    torch.cuda.synchronize()
    end = time.time()
    print((end-start)/2100)

if __name__ == "__main__":
    main()