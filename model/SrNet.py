import torch
from collections import OrderedDict
from math import pow
import torch
from torch import nn

class imgTree:
    def __init__(self, input_img, max_depth):
        self.org_img = input_img
        self.img = OrderedDict()
        self.tem_img = OrderedDict()
        self.img[''] = self.org_img
        self.max_depth = max_depth

        shape = self.org_img.shape
        if len(shape) != 4 \
            or not float(self.max_depth).is_integer() \
            or not (shape[2] / pow(2, self.max_depth)).is_integer() \
            or not (shape[3] / pow(2, self.max_depth)).is_integer():
            print('Image shape does not match the max depth')
            exit()
        else:
            self.down_size = 256
            # self.down_size = [int(shape[2] / pow(2, self.max_depth)), int(shape[3] / pow(2, self.max_depth))]
    def update(self):
        self.img.update(self.tem_img)
        indexes = [key[:-1] for key in self.tem_img.keys()]
        indexes = list(set(indexes))
        for index in indexes:
            del self.img[index]
        self.tem_img.clear()

    def grow(self, beta, encode):
        beta = torch.squeeze(beta)
        # if beta > 0 or encode == '':
        self.tem_img[encode + '0'], self.tem_img[encode + '1'], self.tem_img[encode + '2'], self.tem_img[encode + '3'] = self.quarter(self.img[encode], beta)

    def quarter(self, feature, beta):

        width, height = feature.shape[2:]
        upper_left = feature[:, :, :int(width//2), int(height//2):]
        upper_right = feature[:, :, int(width//2):, int(height//2):]
        lower_left = feature[:, :, :int(width//2), :int(height//2)]
        lower_right = feature[:, :, int(width//2):, :int(height//2)]

        return upper_left, upper_right, lower_left, lower_right

    def resize(self):
        img_list = [nn.Upsample(size=self.down_size, \
                           mode='bilinear', align_corners=True)(self.img[index]) \
                           for index in self.img]
        self.img_tensor = torch.cat(img_list, dim=0)
        self.encode = [index for index in self.img]

    def encoding(self, map):
        if self.encode:
            img_list = []
            for index in self.encode:
                # Get each clipped img locations
                location = [0, 0]
                for j, num in enumerate(index):
                    if num == '0':
                        location[0] += 0
                        location[1] += map.shape[2] // pow(2, j + 1)
                    elif num == '1':
                        location[0] += map.shape[1] // pow(2, j + 1)
                        location[1] += map.shape[2] // pow(2, j + 1)
                    elif num == '2':
                        location[0] += 0
                        location[1] += 0
                    elif num == '3':
                        location[0] += map.shape[1] // pow(2, j + 1)
                        location[1] += 0
                location[0] = int(location[0])
                location[1] = int(location[1])
                add_x = int(map.shape[1] // pow(2, len(index)))
                add_y = int(map.shape[2] // pow(2, len(index)))

                clip_image = map[:, location[0]:location[0] + add_x, location[1]:location[1] + add_y]
                img_list.append(nn.Upsample(size=self.down_size, mode='nearest')(clip_image.unsqueeze(0)).squeeze().unsqueeze(0))

        map_tensor = torch.cat(img_list, dim=0)
        return map_tensor


class clipModule(nn.Module):
    def __init__(self):
        super(clipModule, self).__init__()
        self.operations = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        return self.operations(x)


class SrNet(nn.Module):
    def __init__(self, max_depth, backbone):
        super(SrNet, self).__init__()
        self.max_depth = max_depth
        self.backbone = backbone
        
        for depth in range(self.max_depth):
            setattr(self, "clipModules_{}".format(depth), nn.ModuleList())
            module_num = int(pow(4, depth))
            for i in range(module_num):
                getattr(self, "clipModules_{}".format(depth)).append(clipModule())

    def forward(self, x, chioce=None):
        self.shape = x.shape
        img_tree = imgTree(x, self.max_depth)

        for depth in range(self.max_depth):
            for index in img_tree.img:
                if index == '' and len(index) == depth:
                    f = getattr(self, "clipModules_{}".format(depth))[0](img_tree.img[index])
                    img_tree.grow(torch.tanh(f), index)
                elif len(index) == depth:
                    f = getattr(self, "clipModules_{}".format(depth))[int(index, 4)](img_tree.img[index])
                    img_tree.grow(torch.tanh(f), index)
            img_tree.update()
        img_tree.resize()
        encode = img_tree.encode
        image = img_tree.img_tensor
        predict = self.backbone(image)
        if chioce == 'val' or chioce == 'test':
            return self.collect(predict, encode)
        else:
            return predict, img_tree

    def collect(self, image, encode):
        '''
        Splicing to get the full map
        '''
        classes = 5
        map = torch.zeros(1, classes, self.shape[2], self.shape[3]).cuda()
        if len(image) != len(encode):
            print("resize:Image size does not match the encode size!")
            exit()
        for i in range(len(image)):
            location = [0, 0]
            # Get each clipped img locations
            for j, num in enumerate(encode[i]):
                if num == '0':
                    location[0] += 0
                    location[1] += self.shape[3] // pow(2, j + 1)
                elif num == '1':
                    location[0] += self.shape[2] // pow(2, j + 1)
                    location[1] += self.shape[3] // pow(2, j + 1)
                elif num == '2':
                    location[0] += 0
                    location[1] += 0
                elif num == '3':
                    location[0] += self.shape[2] // pow(2, j + 1)
                    location[1] += 0
            # resize and collect
            location[0] = int(location[0])
            location[1] = int(location[1])
            add_x = int(self.shape[2] // pow(2, len(encode[i])))
            add_y = int(self.shape[3] // pow(2, len(encode[i])))

            map[:, :, location[0]:location[0] + add_x, location[1]:location[1] + add_y] \
                = nn.Upsample(size=(add_x, add_y), mode='bilinear', align_corners=True)(image[i].unsqueeze(0))

        return map