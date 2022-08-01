import numpy as np
import torch
from torch.nn import functional as F
import os
from collections import OrderedDict
class Decoder(object):
    def __init__(self, betas):
        self._betas = torch.from_numpy(betas)
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(self._num_layers, 4, 3)
        self.path = ['', '', '', '']

        for layer in range(self._num_layers):
            if layer == 0:
                self.network_space[layer][0][1] = F.softmax(self._betas[layer][0][1], dim=-1)  * (1/3)
            elif layer == 1:
                self.network_space[layer][0][1] = F.softmax(self._betas[layer][0][1], dim=-1)  * (1/3)
                self.network_space[layer][1][0] = F.softmax(self._betas[layer][1][0], dim=-1) * (1 / 3)
            elif layer == 2:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1][:2] = F.softmax(self._betas[layer][1][:2], dim=-1) * (2/3)
                self.network_space[layer][2][0] = F.softmax(self._betas[layer][2][0], dim=-1) * (1/3)
            elif layer == 3:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2][:2] = F.softmax(self._betas[layer][2][:2], dim=-1) * (2 / 3)
                self.network_space[layer][3][0] = F.softmax(self._betas[layer][3][0], dim=-1) * (1 / 3)

            else:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)
                self.network_space[layer][3][:2] = F.softmax(self._betas[layer][3][:2], dim=-1) * (2/3)

        self.trans_betas()

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
    # TODO: def max_path(self):



    def trans_betas(self):
        betas = self.network_space
        after_trans = np.zeros([16, 4, 3])
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

        self.network_space = after_trans


def get_first_space(path):
    # path = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_flexinet_alldata_first_batch24_relu/experiment_0/betas/'
    betas_list = OrderedDict()
    const_network_list = OrderedDict()
    trans = True

    path_list = OrderedDict()

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[0].split('_')[0] == 'betas':
                betas_list[filename] = np.load(dirpath + filename)
                decoder = Decoder(betas_list[filename])
                const_network_list[filename] = decoder.network_space
                path_list[filename] = decoder.viterbi_decode()
    print(path_list)
    order_path_list = []
    for i in range(len(path_list)):
        idx = 'betas_{}.npy'.format(str(i))
        order_path_list.append(path_list[idx])

    print(order_path_list)
    return order_path_list
# path = '/media/dell/DATA/wy/Seg_NAS/run/GID/12layers_first_batch24_relu/experiment_0/betas/'
# a = get_first_space(path)
# b = 0


