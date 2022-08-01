import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, betas, core_path, core_path_num):
        self.core_path = core_path
        self._betas = betas
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(self._num_layers, 4, 3)

        used_betas = []
        for i in range(self._num_layers):
            used_betas.append([])
            for j in range(self.core_path[i] + 1):
                used_betas[i].append([])
                if j < self.core_path[i]:
                    if j == 0:
                        if core_path[i-1] == 0:
                            used_betas[i][j] = self._betas[i][j][:1]
                        else:
                            used_betas[i][j] = self._betas[i][j][:2]
                    elif j == 1:
                        if core_path[i-1] == 1:
                            used_betas[i][j] = self._betas[i][j][:2]
                        else:
                            used_betas[i][j] = self._betas[i][j][:3]
                    elif j == 2 :
                        if core_path[i-1] == 2:
                            used_betas[i][j] = self._betas[i][j][:2]
                        else:
                            used_betas[i][j] = self._betas[i][j][:3]
                else:
                    used_betas[i][j] = self._betas[i][j][:int(core_path_num[i])]

        for i in range(len(used_betas)):
            for j in range(len(used_betas[i])):
                for k in range(len(used_betas[i][j])):
                    if used_betas[i][j][k] > 0:
                        used_betas[i][j][k] = 1
                    else:
                        used_betas[i][j][k] = 0

        self.used_betas = used_betas

def get_second_space(betas_path, core_path):
    # betas_path = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_second_batch24/experiment_1/betas/'
    betas_list = OrderedDict()
    used_betas_list = OrderedDict()
    # core_path = [0, 0, 1, 1, 1, 0, 1, 0, 1, 2, 2, 2]
    core_path_num = np.zeros(len(core_path))
    for i in range(len(core_path)):
        if i == 0:
            continue
        core_path_num[i] = core_path_num[i-1] + core_path[i-1] + 1

    for dirpath, dirnames, filenames in os.walk(betas_path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'betas':
                betas_list[filename] = np.load(dirpath + filename)
                decoder = Decoder(betas_list[filename], core_path, core_path_num)
                used_betas_list[filename] = decoder.used_betas

    order_path_list = []
    for i in range(len(used_betas_list)):
        idx = 'betas_{}.npy'.format(str(i))
        order_path_list.append(used_betas_list[idx])

    return order_path_list
    # print(path_list)
    # print(used_betas_list)
    # b = np.array(path_list['betas_52.npy'])

    # np.save(path + 'path.npy', b)
    # print(b)


