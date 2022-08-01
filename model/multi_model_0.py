# this model is derived from GID(40 epoches) autodeeplab supernet using multi decode

import torch
import torch.nn as nn
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

import torch.nn.functional as F
import numpy as np
from retrain.operations import *



class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, args.affine, args.use_ABN)
        self.preprocess = ReLUConvBN(self.C_prev, self.C_out, 1, 1, 0, args.affine, args.use_ABN)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1, affine=args.affine, use_ABN=args.use_ABN)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)

        return concat_feature


class newModel(nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, filter_multiplier=32, block_multiplier=5, step=5, cell=Cell,
                 BatchNorm=NaiveBN, args=None):
        super(newModel, self).__init__()
        self.args = args
        self._step = step
        self.cells = nn.ModuleList()
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._block_multiplier = args.block_multiplier
        self._filter_multiplier = args.filter_multiplier
        self.use_ABN = args.use_ABN
        initial_fm = 128 if args.initial_fm is None else args.initial_fm
        half_initial_fm = initial_fm // 2
        self.stem0 = nn.Sequential(
            nn.Conv2d(4, half_initial_fm, 3, stride=2, padding=1),
            BatchNorm(half_initial_fm)
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_initial_fm, half_initial_fm, 3, padding=1),
            BatchNorm(half_initial_fm)
        )
        ini_initial_fm = half_initial_fm
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_initial_fm, initial_fm, 3, stride=2, padding=1),
            BatchNorm(initial_fm)
        )
        # C_prev_prev = 64
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}

        self.cell_0_0_1 = cell(self._step, self._block_multiplier, ini_initial_fm / args.block_multiplier,
                               initial_fm / args.block_multiplier, self.cell_arch,self._filter_multiplier *filter_param_dict[0], 0, self.args)
        self.cell_0_1_0 = cell(self._step, self._block_multiplier, ini_initial_fm / args.block_multiplier,
                               initial_fm / args.block_multiplier, self.cell_arch,self._filter_multiplier *filter_param_dict[1], -1, self.args)

        self.cell_1_0_2 = cell(self._step, self._block_multiplier, initial_fm / args.block_multiplier,
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[0], 1, self.args)
        self.cell_1_1_0 = cell(self._step, self._block_multiplier, initial_fm / args.block_multiplier,
                               self._filter_multiplier * filter_param_dict[0], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], -1, self.args)

        self.cell_2_1_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[0], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], -1, self.args)
        self.cell_2_2_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[0],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], -1, self.args)

        self.cell_3_2_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[0],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], -1, self.args)
        self.cell_3_2_1 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 0, self.args)
        self.combine_3 = nn.Sequential(
            nn.Conv2d(self._filter_multiplier * filter_param_dict[2] * 5 * 2,
                      self._filter_multiplier * filter_param_dict[2] * 5, 3, stride=1, padding=1),
            BatchNorm(self._filter_multiplier * filter_param_dict[2] * 5))

        self.cell_4_1_2 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], 1, self.args)
        self.cell_4_2_1 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 0, self.args)

        self.cell_5_1_1 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], 0, self.args)
        self.cell_5_2_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], -1, self.args)
        self.cell_5_2_1 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 0, self.args)
        self.combine_5 = nn.Sequential(
            nn.Conv2d(self._filter_multiplier * filter_param_dict[2] * 5 * 2,
                      self._filter_multiplier * filter_param_dict[2] * 5, 3, stride=1, padding=1),
            BatchNorm(self._filter_multiplier * filter_param_dict[2] * 5))

        self.cell_6_1_2 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], 1, self.args)
        self.cell_6_2_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], -1, self.args)
        self.cell_6_2_1 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 0, self.args)
        self.cell_6_3_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[3], -1, self.args)
        self.combine_6 = nn.Sequential(
            nn.Conv2d(self._filter_multiplier * filter_param_dict[2] * 5 * 2,
                      self._filter_multiplier * filter_param_dict[2] * 5, 3, stride=1, padding=1),
            BatchNorm(self._filter_multiplier * filter_param_dict[2] * 5))

        self.cell_7_2_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], -1, self.args)
        self.cell_7_2_2 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[3], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 1, self.args)
        self.cell_7_3_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[3], -1, self.args)
        self.combine_7 = nn.Sequential(
            nn.Conv2d(self._filter_multiplier * filter_param_dict[2] * 5 * 2,
                      self._filter_multiplier * filter_param_dict[2] * 5, 3, stride=1, padding=1),
            BatchNorm(self._filter_multiplier * filter_param_dict[2] * 5))

        self.cell_8_1_2 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], 1, self.args)
        self.cell_8_2_2 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[3], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 1, self.args)

        self.cell_9_0_2 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[0], 1, self.args)
        self.cell_9_2_1 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[3],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[2], 0, self.args)

        self.cell_10_1_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[1],
                               self._filter_multiplier * filter_param_dict[0], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[1], -1, self.args)
        self.cell_10_3_0 = cell(self._step, self._block_multiplier,
                               self._filter_multiplier * filter_param_dict[2],
                               self._filter_multiplier * filter_param_dict[2], self.cell_arch,
                               self._filter_multiplier * filter_param_dict[3], -1, self.args)

        self.cell_11_0_2 = cell(self._step, self._block_multiplier,
                                self._filter_multiplier * filter_param_dict[0],
                                self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                                self._filter_multiplier * filter_param_dict[0], 1, self.args)
        self.cell_11_1_1 = cell(self._step, self._block_multiplier,
                                self._filter_multiplier * filter_param_dict[0],
                                self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                                self._filter_multiplier * filter_param_dict[1], 0, self.args)
        self.cell_11_2_0 = cell(self._step, self._block_multiplier,
                                self._filter_multiplier * filter_param_dict[0],
                                self._filter_multiplier * filter_param_dict[1], self.cell_arch,
                                self._filter_multiplier * filter_param_dict[2], -1, self.args)
        self.cell_11_3_1 = cell(self._step, self._block_multiplier,
                                self._filter_multiplier * filter_param_dict[2],
                                self._filter_multiplier * filter_param_dict[3], self.cell_arch,
                                self._filter_multiplier * filter_param_dict[3], 0, self.args)



    def forward(self, x):
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        x_0_0 = self.cell_0_0_1(stem0, stem1)
        x_0_1 = self.cell_0_1_0(stem, stem1)

        x_1_0 = self.cell_1_0_2(stem1, x_0_1)
        x_1_1 = self.cell_1_1_0(stem1, x_0_0)

        x_2_1 = self.cell_2_1_0(x_0_1, x_1_0)
        x_2_2 = self.cell_2_2_0(x_0_0, x_1_1)

        x_3_2_0 = self.cell_3_2_0(x_1_0, x_2_1)
        x_3_2_1 = self.cell_3_2_1(x_1_1, x_2_2)
        x_3_2 = torch.cat([x_3_2_0, x_3_2_1], dim=1)
        x_3_2 = self.combine_3(x_3_2)

        x_4_1 = self.cell_4_1_2(x_2_1, x_3_2)
        x_4_2 = self.cell_4_2_1(x_2_1, x_3_2)

        x_5_1 = self.cell_5_1_1(x_3_2, x_4_1)
        x_5_2_0 = self.cell_5_2_0(x_3_2, x_4_1)
        x_5_2_1 = self.cell_5_2_1(x_3_2, x_4_2)
        x_5_2 = torch.cat([x_5_2_0, x_5_2_1], dim=1)
        x_5_2 = self.combine_5(x_5_2)

        x_6_1 = self.cell_6_1_2(x_4_1, x_5_2)
        x_6_2_0 = self.cell_6_2_0(x_4_1, x_5_1)
        x_6_2_1 = self.cell_6_2_1(x_4_1, x_5_2)
        x_6_2 = torch.cat([x_6_2_0, x_6_2_1], dim=1)
        x_6_2 = self.combine_6(x_6_2)
        x_6_3 = self.cell_6_3_0(x_4_1, x_5_2)

        x_7_2_0 = self.cell_7_2_0(x_5_2, x_6_1)
        x_7_2_1 = self.cell_7_2_2(x_5_2, x_6_3)
        x_7_2 = torch.cat([x_7_2_0, x_7_2_1], dim=1)
        x_7_2 = self.combine_7(x_7_2)
        x_7_3 = self.cell_7_3_0(x_5_1, x_6_2)

        x_8_1 = self.cell_8_1_2(x_6_1, x_7_2)
        x_8_2 = self.cell_8_2_2(x_6_2, x_7_3)

        x_9_0 = self.cell_9_0_2(x_7_2, x_8_1)
        x_9_2 = self.cell_9_2_1(x_7_3, x_8_2)

        x_10_1 = self.cell_10_1_0(x_8_1, x_9_0)
        x_10_3 = self.cell_10_3_0(x_8_2, x_9_2)

        x_11_0 = self.cell_11_0_2(x_9_0, x_10_1)
        x_11_1 = self.cell_11_1_1(x_9_0, x_10_1)
        x_11_2 = self.cell_11_2_0(x_9_0, x_10_1)
        x_11_3 = self.cell_11_3_1(x_9_2, x_10_3)

        return [x_11_0, x_11_1, x_11_2, x_11_3]

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params