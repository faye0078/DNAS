from collections import OrderedDict
def get_data_path(dataset):
    if dataset == 'GID':
        Path = OrderedDict()
        Path['dir'] = "../data/512"
        Path['train_list'] = "./data/lists/GID/rs_train.lst"
        Path['val_list'] = "./data/lists/GID/rs_val.lst"
        Path['test_list'] = "./data/lists/GID/rs_test.lst"
    elif dataset == 'uadataset' or dataset == 'uadataset_dfc':
        Path = OrderedDict()
        Path['dir'] = "../data/"
        Path['train_list'] = "./data/lists/uadataset/map_uad_512_train.lst"
        Path['val_list'] = "./data/lists/uadataset/map_uad_512_val.lst"
        Path['test_list'] = "./data/lists/uadataset/map_uad_512_test.lst"
        Path['mini_train_list'] = "./data/lists/uadataset/mini_uad_512_train.lst"
        Path['mini_val_list'] = "./data/lists/uadataset/mini_uad_512_val.lst"
        Path['mini_test_list'] = "./data/lists/uadataset/mini_map_uad_512_test.lst"
    return Path