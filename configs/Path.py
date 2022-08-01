from collections import OrderedDict
def get_data_path(dataset):
    if dataset == 'hps-GID':
        Path = OrderedDict()
        Path['dir'] = "/media/dell/DATA/wy/data/GID-15/512/"
        Path['train_list'] = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_train.lst"
        Path['val_list'] = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_val.lst"
        Path['test_list'] = "/media/dell/DATA/wy/LightRS/data/list/gid15_vege_val.lst"
        Path['mini_train_list'] = "./data/lists/mini_hps_train.lst"
        Path['mini_val_list'] = "./data/lists/mini_hps_val.lst"
    elif dataset == 'GID':
        Path = OrderedDict()
        # Path['dir'] = "../data/GID-5/1024"
        # Path['train_list'] = "./data/lists/gid_1024_train.lst"
        # Path['val_list'] = "./data/lists/gid_1024_val.lst"
        # Path['test_list'] = "./data/lists/gid_1024_test.lst"
        # Path['mini_train_list'] = "./data/lists/mini_rs_train.lst"
        # Path['mini_val_list'] = "./data/lists/mini_rs_val.lst"
        Path['dir'] = "../data/512"
        Path['train_list'] = "./data/lists/GID/rs_train.lst"
        Path['val_list'] = "./data/lists/GID/rs_val.lst"
        Path['test_list'] = "./data/lists/GID/rs_test.lst"
        Path['mini_train_list'] = "./data/lists/GID/mini_rs_train.lst"
        Path['mini_val_list'] = "./data/lists/GID/mini_rs_val.lst"
        Path['mini_test_list'] = "./data/lists/GID/mini_rs_test.lst"
    elif dataset == 'GID-15':
        Path = OrderedDict()
        Path['dir'] = "../data/GID-15/512/"
        Path['train_list'] = "./data/lists/gid15_train.lst"
        Path['val_list'] = "./data/lists/gid15_val.lst"
        Path['test_list'] = "./data/lists/gid15_val.lst"
        Path['mini_train_list'] = "./data/lists/mini_gid15_train.lst"
        Path['mini_val_list'] = "./data/lists/mini_gid15_val.lst"
    elif dataset == 'cityscapes':
        Path = '/media/dell/DATA/wy/data/cityscapes'

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