"""MobileNetv2 Encoder"""

import torch
import torch.nn as nn
from .layer_factory import InvertedResidual, conv_bn_relu6, conv_1x1_bn_relu6


__all__ = ["mbv2"]


# model_paths = {"mbv2_voc": "E:/wangyu_file/nas-segm-pytorch-master/src/ckpt/test.pth"}
# model_paths = {"mbv2_voc": "E:/wangyu_file/model/pretrained_hrnet_encoder.pth"}
model_paths = {"mbv2_voc": "./data/weights/mbv2_voc_rflw.ckpt"}

class MobileNetV2(nn.Module):
    """MobileNetV2 definition"""

    # expansion rate, output channels, number of repeats, stride
    mobilenet_config = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    in_planes = 32  # number of input channels
    num_layers = len(mobilenet_config)

    def __init__(self, width_mult=1.0, return_layers=[1, 2, 4, 6]):
        super(MobileNetV2, self).__init__()
        self.return_layers = return_layers
        self.max_layer = max(self.return_layers)
        self.out_sizes = [
            self.mobilenet_config[layer_idx][1] for layer_idx in self.return_layers
        ]
        input_channel = int(self.in_planes * width_mult)
        self.init = conv_1x1_bn_relu6(3, 3)
        self.layer1 = conv_bn_relu6(3, input_channel, 2)
        for layer_idx, (t, c, n, s) in enumerate(
            self.mobilenet_config[: self.max_layer + 1]
        ):
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if i == 0:
                    features.append(
                        InvertedResidual(input_channel, output_channel, s, t)
                    )
                else:
                    features.append(
                        InvertedResidual(input_channel, output_channel, 1, t)
                    )
                input_channel = output_channel
            setattr(self, "layer{}".format(layer_idx + 2), nn.Sequential(*features))
        for m in self.init.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        outs = []
        x = self.init(x)
        x = self.layer1(x)
        for layer_idx in range(self.max_layer + 1):
            x = getattr(self, "layer{}".format(layer_idx + 2))(x)
            outs.append(x)
        return [outs[layer_idx] for layer_idx in self.return_layers]

    def init_weight(self):
        a = 0
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.constant_(m.weight, 1)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()

    def forward(self, x):
        x = x

def mbv2(pretrained=False, **kwargs):
    """Constructs a MobileNet-v2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(**kwargs)
    model.init_weight()
    if pretrained:
        # model.load_state_dict(
        #     torch.load(model_paths["mbv2_{}".format(str(pretrained))]), strict=False
        # )
          model.load_state_dict(
               torch.load('../../model/pretrained_mobile_encoder.pth'), strict=False
           )
    return model

def hrnet(pretrained=False, **kwargs):
    """Constructs a HRNet model.

    :param pretrained: If True, return a model pre-trained on GID
    :param kwargs:
    :return: HRNet model
    """
    model = HRNet(**kwargs)
    hrnet_path = ""
    if pretrained:
        model.load_state_dict(
            torch.load(hrnet_path), strict=False
        )
    return model

def create_encoder(pretrained="voc", ctrl_version="cvpr", **kwargs):
    """Create Encoder"""
    return_layers = [1, 2, 4, 6] if ctrl_version == "cvpr" else [1, 2]
    if (ctrl_version == "cvpr"):
        return mbv2(pretrained=False, return_layers=return_layers, **kwargs)
        # return get_seg_model(pretrained=True)

