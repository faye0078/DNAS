import torch
import numpy as np
import random
import sys
sys.path.append("../")
from configs.search_args import obtain_search_args
from engine.search_trainer import Trainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = obtain_search_args()
    print(args)
    args.cuda = torch.cuda.is_available()
    setup_seed(args.seed)
    trainer = Trainer(args)

    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

if __name__ == "__main__":
    main()
