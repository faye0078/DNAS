import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class RandomCrop(object):
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
        return {"image": image, "mask": mask, "name": sample["name"]}

class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return {"image": image, "mask": mask, "name": sample["name"]}


class Normalise(object):
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        return {
            "image": (self.scale * image - self.mean) / self.std,
            "mask": sample["mask"],
            "name": sample["name"]
        }


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask), "name": sample["name"]}


class GIDDataset(Dataset):

    def __init__(self, stage, data_file, data_dir, transform_trn=None, transform_val=None, transform_test=None):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        try:
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

        sample = {"image": image, "mask": mask, "name": self.datalist[idx][1]}
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
