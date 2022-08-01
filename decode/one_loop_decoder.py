import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict

def get_connections_2(core_path):
    connections = []
    connections.append([[-1, 0], [0, 0]])
    for i in range(len(core_path) - 1):
        connections.append([[i, core_path[i]], [i + 1, core_path[i + 1]]])
    return connections


class Decoder_2(object):
    def __init__(self, alphas, core_path, cell_arch):
        self.alphas = torch.from_numpy(alphas)
        self._num_layers = self.alphas.shape[0]
        self.cell_space = cell_arch.copy()
        self.core_path = core_path

    def get_n_arch(self):
        for i in range(self._num_layers):
            for j in range(4):
                if self.core_path[i] == j:
                    num = int(self.cell_space[i][j].sum() + 3)
                    if num > 13:
                        num = 13
                    self.cell_space[i][j] = 0
                    self.cell_space[i][j][self.alphas[i][j].sort()[1][-num:]] = 1
        return self.cell_space

def make_rs_ops_disable(cell_arch):
    cell_arch[:, :, -2:] = 0
    return cell_arch

if __name__ == "__main__":
    # alphas = np.random.randn(14, 4, 11)
    # core_path = np.load('/media/dell/DATA/wy/Seg_NAS/run/GID/one_loop/search/experiment_2/path/1_core_path_epoch1.npy')
    # cell_arch = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/one_loop_14layers_mixedcell1_3operation/init_cell_arch.npy')
    # decoder = Decoder_2(alphas, core_path, cell_arch)
    # a = decoder.get_n_arch()
    cell_arch = np.load('/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_0/cell_arch/2_cell_arch_epoch24.npy')
    no_rs_cell_arch = make_rs_ops_disable(cell_arch)
    np.save('/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_0/cell_arch/2_cell_arch_epoch_nors24.npy', no_rs_cell_arch)