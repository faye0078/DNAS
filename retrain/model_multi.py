import numpy as np
from torch._C import ErrorReport
import torch.nn as nn
import sys
sys.path.append("..")

from retrain.operations import NaiveBN
from retrain.aspp import ASPP
import torch
from model.multi_model_0 import newModel
from retrain.decoder import network_layer_to_space

class Retrain_Autodeeplab(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab, self).__init__()
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        BatchNorm2d = NaiveBN
        self.args = args

        cell_arch = np.load(args.cell_arch)

        self.encoder = newModel(None, cell_arch, args.nclass, 12, args.filter_multiplier, BatchNorm=BatchNorm2d, args=args)

        self.aspp0 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[0],
                          64, args.nclass, conv=nn.Conv2d, norm=BatchNorm2d)
        self.aspp1 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[1],
                          64, args.nclass, conv=nn.Conv2d, norm=BatchNorm2d)
        self.aspp2 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[2],
                          64, args.nclass, conv=nn.Conv2d, norm=BatchNorm2d)
        self.aspp3 = ASPP(args.filter_multiplier * args.block_multiplier * filter_param_dict[3],
                          64, args.nclass, conv=nn.Conv2d, norm=BatchNorm2d)

        self.decoder = nn.Sequential(nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, args.nclass, kernel_size=1, stride=1))


    def forward(self, x):
        features = self.encoder(x)
        aspp_result_0 = self.aspp0(features[0])
        aspp_result_1 = self.aspp1(features[1])
        aspp_result_2 = self.aspp2(features[2])
        aspp_result_3 = self.aspp3(features[3])
        upsample = nn.Upsample(size=[64, 64], mode='bilinear', align_corners=True)
        aspp_result_0 = upsample(aspp_result_0)
        aspp_result_1 = upsample(aspp_result_1)
        aspp_result_2 = upsample(aspp_result_2)
        aspp_result_3 = upsample(aspp_result_3)
        aspp_result = torch.cat([aspp_result_0, aspp_result_1, aspp_result_2, aspp_result_3], dim=1)
        decoder_output = self.decoder(aspp_result)
        upsample = nn.Upsample(size=[512, 512], mode='bilinear', align_corners=True)(decoder_output)
        return upsample

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params