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

    def viterbi_decode(self):
        prob_space = np.zeros((self.network_space.shape[:2]))
        path_space = np.zeros((self.network_space.shape[:2])).astype('int8')

        for layer in range(self.network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = self.network_space[layer][0][1]
                prob_space[layer][1] = self.network_space[layer][0][2]
                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(self.network_space.shape[1]):
                    if layer - sample < - 1:
                        continue
                    local_prob = []
                    for rate in range(self.network_space.shape[2]):  # k[0 : ➚, 1: ➙, 2 : ➘]
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            continue
                        else:
                            local_prob.append(prob_space[layer - 1][sample + 1 - rate] *
                                              self.network_space[layer][sample + 1 - rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path  # path[1 : ➚, 0: ➙, -1 : ➘]

        output_sample = prob_space[-1, :].argmax(axis=-1)
        actual_path = np.zeros(self._num_layers).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self._num_layers):
            actual_path[-i - 1] = actual_path[-i] + path_space[self._num_layers - i, actual_path[-i]]

        return actual_path

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


def trans_betas(betas):
    after_trans = np.zeros([12, 4, 3])
    shape = betas.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if j == 0:
                after_trans[i][0][1] = betas[i][j][1]
                after_trans[i][1][0] = betas[i][j][2]
            if j == 1:
                after_trans[i][0][2] = betas[i][j][0]
                after_trans[i][1][1] = betas[i][j][1]
                after_trans[i][2][0] = betas[i][j][2]
            if j == 2:
                after_trans[i][1][2] = betas[i][j][0]
                after_trans[i][2][1] = betas[i][j][1]
                after_trans[i][3][0] = betas[i][j][2]
            if j == 3:
                after_trans[i][2][2] = betas[i][j][0]
                after_trans[i][3][1] = betas[i][j][1]

    return after_trans


if __name__ == '__main__':
    path = '/media/dell/DATA/wy/Seg_NAS/run/cityscapes/12layers_forward/'
    alphas_list = OrderedDict()
    betas_list = OrderedDict()
    trans = False

    if path.find('forward') != -1:
        trans = True

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

                if trans:
                    betas_list[filename] = trans_betas(betas_list[filename])
                alphas_name = filename.replace('betas', 'alphas')

                decoder = Decoder(alphas_list[alphas_name], betas_list[filename], 5)
                cell_list[alphas_name] = decoder.genotype_decode()
                path_list[filename] = decoder.viterbi_decode()

    print(cell_list)
    print(path_list)

    a = np.array(cell_list['alphas_56.npy'])
    b = np.array(path_list['betas_56.npy'])

    np.save(path + 'cell.npy', a)
    np.save(path + 'path.npy', b)
    print(a, b)


