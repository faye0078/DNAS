import torch
from torch import nn
import torch.nn.functional as F
from model.ops import OPS, OPS_mini
from model.ops import conv3x3
class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConvBN, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.scale = C_in/C_out
        self._initialize_weights()

    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out):
        super(ConvBNReLU, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=False)
        )

        self.scale = C_in/C_out
        self._initialize_weights()

    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class ReLUConv5BN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConv5BN, self).__init__()
        kernel_size = 5
        padding = 2
        self.scale = 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.scale = C_in/C_out
        self._initialize_weights()



    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class MixedCell(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedCell, self).__init__()
        kernel_size = 5
        padding = 2
        self.scale = 1
        self._ops = nn.ModuleDict()
        for op_name in OPS:
            op = OPS[op_name](C_in, C_out, 1, True)
            self._ops[op_name] = op
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out
        self._initialize_weights()

    def forward(self, x, cell_alphas):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return sum(w * self._ops[op](x) for w, op in zip(cell_alphas, self._ops))

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'sobel_operator.filter' not in name:
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))
        
class MixedRetrainCell(nn.Module):
    
    def __init__(self, C_in, C_out, arch):
        super(MixedRetrainCell, self).__init__()
        self.scale = 1
        self._ops = nn.ModuleList()
        for i, op_name in enumerate(OPS):
            if arch[i] == 1:
                op = OPS[op_name](C_in, C_out, 1, True)
                self._ops.append(op)
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out
        self._initialize_weights()

    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return sum(op(x) for op in self._ops)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class MixedCellMini(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedCellMini, self).__init__()
        kernel_size = 5
        padding = 2
        self.scale = 1
        self._ops = nn.ModuleDict()
        for op_name in OPS_mini:
            op = OPS_mini[op_name](C_in, C_out, 1, True)
            self._ops[op_name] = op
        self.ops_num = len(self._ops)
        self.scale = C_in / C_out
        self._initialize_weights()

    def forward(self, x, cell_alphas):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return sum(w * self._ops[op](x) for w, op in zip(cell_alphas, self._ops))

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'sobel_operator.filter' not in name:
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class DCNAS_cell(nn.Module):
    def __init__(self):
        super(DCNAS_cell, self).__init__()

class Fusion(nn.Module):

    def __init__(self, C_in, C_out):
        super(Fusion, self).__init__()

        self.scale = 1

        self.conv = nn.Sequential(
        conv3x3(C_in, C_out, 1),
        nn.BatchNorm2d(C_out, 1),
        nn.ReLU(inplace=False),)
        self.scale = C_in / C_out
        self._initialize_weights()

    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return self.conv(x)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'sobel_operator.filter' not in name:
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))