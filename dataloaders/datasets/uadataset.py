"""Data Transformations and pre-processing."""

from __future__ import print_function, division

import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def make_even(x):
    """Make number divisible by 2"""
    if x % 2 != 0:
        x -= 1
    return x


class Pad(object):
    """Pad image and mask to the desired size

    Args:
      size (int) : minimum length/width
      img_val (array) : image padding value
      msk_val (int) : mask padding value

    """

    def __init__(self, size, img_val, msk_val):
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        image = np.stack(
            [
                np.pad(
                    image[:, :, c],
                    pad,
                    mode="constant",
                    constant_values=self.img_val[c],
                )
                for c in range(3)
            ],
            axis=2,
        )
        mask = np.pad(mask, pad, mode="constant", constant_values=self.msk_val)
        return {"image": image, "mask": mask}


class CentralCrop(object):
    """Crop centrally the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        h, w = image.shape[:2]

        h_margin = (h - self.crop_size) // 2
        w_margin = (w - self.crop_size) // 2

        image = image[
            h_margin : h_margin + self.crop_size, w_margin : w_margin + self.crop_size
        ]
        mask = mask[
            h_margin : h_margin + self.crop_size, w_margin : w_margin + self.crop_size
        ]
        return {"image": image, "mask": mask}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top : top + new_h, left : left + new_w]
        mask = mask[top : top + new_h, left : left + new_w]
        return {"image": image, "mask": mask}


class ResizeShorter(object):
    """Resize shorter side to a given value."""

    def __init__(self, shorter_side):
        assert isinstance(shorter_side, int)
        self.shorter_side = shorter_side

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        min_side = min(image.shape[:2])
        scale = 1.0
        if min_side < self.shorter_side:
            scale *= self.shorter_side * 1.0 / min_side
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            mask = cv2.resize(
                mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        return {"image": image, "mask": mask}


class ResizeScale(object):
    """Resize (shorter or longer) side to a given value and randomly scale."""

    def __init__(self, resize_side, low_scale, high_scale, longer=False):
        assert isinstance(resize_side, int)
        self.resize_side = resize_side
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.longer = longer

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        scale = np.random.uniform(self.low_scale, self.high_scale) ###
        if self.longer:
            mside = max(image.shape[:2])
            if mside * scale > self.resize_side:
                scale = self.resize_side * 1.0 / mside
        else:
            mside = min(image.shape[:2])
            if mside * scale < self.resize_side:
                scale = self.resize_side * 1.0 / mside
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        mask = cv2.resize(
            mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        return {"image": image, "mask": mask}


class RandomMirror(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return {"image": image, "mask": mask}


class Normalise(object):
    """Normalise an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image = sample["image"]
        return {
            "image": (self.scale * image - self.mean) / self.std,
            "mask": sample["mask"],
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask)}


class UADataset(Dataset):
    """Custom Pascal VOC"""

    def __init__(self, stage, data_file, data_dir, transform_trn=None, transform_val=None, transform_test=None):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        try:
            # self.datalist = [
            #     (k, k.replace('image/', 'label/').replace('img.tif', 'label.png'))
            #     for k in map(
            #         lambda x: x.decode("utf-8").strip("\n").strip("\r"), datalist
            #     )
            # ]
            self.datalist = [
                (k[0], k[1])
                for k in map(
                    lambda x: x.decode("utf-8").strip("\n").strip("\r").split("\t"), datalist
                )
            ]
        except ValueError:  # Adhoc for test.
            self.datalist = [
                (k, k) for k in map(lambda x: x.decode("utf-8").strip("\n"), datalist)
            ]

        self.all_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.valid_classes = [255, 0, 1, 2, 3, 4, 5, 6, 255, 255, 7, 8, 9, 10, 11, 255]
        self.class_map = dict(zip(self.all_classes, self.valid_classes))
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.stage = stage



    def set_config(self, crop_size, resize_side):
        self.transform_trn.transforms[0].resize_side = resize_side
        self.transform_trn.transforms[2].crop_size = crop_size

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])

        def read_image(x):
            img_arr = np.asarray(Image.open(x), dtype=np.uint8)
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr

        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        # mask = self.encode_segmap(mask).astype(np.float32)
        # if img_name != msk_name:
        #     assert len(mask.shape) == 2, "Masks must be encoded without colourmap"
        sample = {"image": image, "mask": mask}
        if self.stage == "train":
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == "val":
            if self.transform_val:
                sample = self.transform_val(sample)
        elif self.stage == 'test':
            if self.transform_test:
                sample = self.transform_test(sample)
        return sample

    def encode_segmap(self, mask):
        for _validc in self.all_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
