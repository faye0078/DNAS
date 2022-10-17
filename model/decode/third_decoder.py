import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, alphas):
        self.alphas = torch.from_numpy(alphas)
        self._num_layers = self.alphas.shape[0]
        self.cell_space = torch.zeros(self._num_layers, 4, 10)

        for i in range(self._num_layers):
            for j in range(4):
                self.cell_space[i][j][self.alphas[i][j].sort()[1][-3:]] = 1

def get_retrain_space(alpha_path):
    path = alpha_path
    alphas_list = OrderedDict()
    cell_list = OrderedDict()

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'alphas':
                alphas_list[filename] = np.load(dirpath + filename)
                decoder = Decoder(alphas_list[filename])
                cell_list[filename] = decoder.cell_space

    order_cell_list = []
    for i in range(len(cell_list)):
        # idx = 'alphas_{}.npy'.format(str(i))
        idx = 'alphas_{}.npy'.format(str(i))
        order_cell_list.append(cell_list[idx])
    # print(path_list)
    print(order_cell_list)
    return order_cell_list


