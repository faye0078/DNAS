from dataloaders.datasets import GID
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("../")
from configs.Path import get_data_path

from dataloaders.datasets.GID import (
    Normalise,
    RandomCrop,
    RandomMirror,
    ToTensor,
)
from torchvision import transforms
def make_data_loader(args, **kwargs):
    if 'GID' in args.dataset:
        data_path = get_data_path(args.dataset)
        num_class = 5
        if args.dataset == 'GID-15':
            num_class = 15

        composed_trn = transforms.Compose(
            [
                RandomMirror(),
                RandomCrop(args.crop_size),
                Normalise(*args.normalise_params),
                ToTensor(),
            ]
        )
        composed_val = transforms.Compose(
            [

                RandomCrop(args.crop_size),
                Normalise(*args.normalise_params),
                ToTensor(),
            ]
        )
        composed_test = transforms.Compose(
            [
                RandomCrop(args.crop_size),
                Normalise(*args.normalise_params),
                ToTensor(),
            ])
        if args.nas == 'search':
            if args.model_name == 'DNAS':
                train_set = GID.GIDDataset(stage="train",
                                           data_file=data_path['train_list'],
                                           data_dir=data_path['dir'],
                                           transform_trn=composed_trn, )
                val_set = GID.GIDDataset(stage="val",
                                        data_file=data_path['val_list'],
                                         data_dir=data_path['dir'],
                                         transform_val=composed_val, )
                test_set = GID.GIDDataset(stage="test",
                                          data_file=data_path['test_list'],
                                          data_dir=data_path['dir'],
                                          transform_test=composed_test, )
        elif args.nas == 'train':
            train_set = GID.GIDDataset(stage="train",
                                        data_file=data_path['train_list'],
                                        data_dir=data_path['dir'],
                                        transform_trn=composed_trn,)
            val_set = GID.GIDDataset(stage="val",
                                        data_file=data_path['val_list'],
                                        data_dir=data_path['dir'],
                                        transform_val=composed_val,)
            test_set = GID.GIDDataset(stage="test",
                                        data_file=data_path['test_list'],
                                        data_dir=data_path['dir'],
                                        transform_test=composed_test,)
        else:
            raise Exception('nas param not set properly')

        n_examples = len(train_set)
        n_train = int(n_examples/2)
        train_set1, train_set2 = random_split(train_set, [n_train, n_examples - n_train])
        
        if args.nas == 'nas':
            print(" Created train setB = {} examples, train setB = {}, val set = {} examples".format(len(train_set1), len(train_set2), len(val_set)))
        elif args.nas == 'train':
            print(" Created train set = {} examples, val set = {} examples, test set = {} examples".format(len(train_set), len(val_set), len(test_set)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        if args.nas == 'train':
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        if args.nas == 'search':
            return train_loader1, train_loader2, val_loader, num_class
        elif args.nas == 'train':
            return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
