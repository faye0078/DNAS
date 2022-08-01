# 这个文件用于对betas进行多分支的解码

import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, alphas, betas, steps):
        self._alphas = torch.from_numpy(alphas)
        self._betas = torch.from_numpy(betas)
        self._num_layers = self._betas.shape[0]
        self._steps = steps
        self.network_space = torch.zeros(self._num_layers, 4, 3)
        self.arch_space = torch.zeros(self._num_layers, 4, 3)

        for layer in range(self._num_layers):
            if layer == 0:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)  * (2/3)
            elif layer == 1:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)

            elif layer == 2:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)


            else:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)
                self.network_space[layer][3][:2] = F.softmax(self._betas[layer][3][:2], dim=-1) * (2/3)

    def multi_decode(self):
        for i in range(self._betas.shape[0]):
            for j in range(self._betas.shape[1]):
                for k in range(self._betas.shape[2]):
                    if self._betas[i][j][k] > 0:
                        self.arch_space[i][j][k] = self._betas[i][j][k]

        return self.arch_space

    def genotype_decode(self):

        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_alphas = F.softmax(self._alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(normalized_alphas, self._steps)

        return gene_cell

if __name__ == '__main__':
    path = '/media/dell/DATA/wy/Seg_NAS/run/cityscapes/12layers_forward/'
    alphas_list = OrderedDict()
    betas_list = OrderedDict()

    cell_list = OrderedDict()
    path_list = OrderedDict()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'alphas':
                alphas_list[filename] = np.load(dirpath + filename)

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'betas':
                betas_list[filename] = np.load(dirpath + filename)
                alphas_name = filename.replace('betas', 'alphas')
                decoder = Decoder(alphas_list[alphas_name], betas_list[filename], 5)
                cell_list[alphas_name] = decoder.genotype_decode()
                path_list[filename] = decoder.multi_decode()

    print(cell_list)
    print(path_list)

    a = np.array(cell_list['alphas_56.npy'])
    b = np.array(path_list['betas_56.npy'])

    print(a)
    print(b)

