# Adapted from: https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py

import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            size = [size, size]
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        if len(img_group) == 0:
            print(img_group)
        if isinstance(img_group[0], Image.Image):
            w, h = img_group[0].size
        elif isinstance(img_group[0], torch.Tensor):
            _, w,h = img_group[0].size()
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            # assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                if isinstance(img, Image.Image):
                    out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
                else:
                    out_images.append(img[: , x1: x1 + tw,  y1: y1 + th])

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomRotation(object): 
    def __init__(self, angle=30):
        self.worker = torchvision.transforms.RandomRotation(angle)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            if isinstance(img_group[0], Image.Image):
                img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            else:
                horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
                img_group = [horizontal_flip(img) for img in img_group]
        return img_group

class GroupRandomColorJitter(object): 
    def __init__(self, brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5,1.5), hue=(-0.5, 0.5)):
        self.worker = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, img_group):
        # Choose how many and to which images to apply the transform to randomly
        N = random.randint(0, len(img_group))
        idxs = random.sample(range(len(img_group), N))
        trans_imgs = []
        for i in range(len(img_group)):
            if i in idxs:
                trans_imgs.append(self.worker(img_group[i]))
            else:
                trans_imgs.append(img_group[i])
        return trans_imgs

class ToTensor(object):
    def __init__(self):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        if isinstance(img_group[0], torch.Tensor):
            return torch.stack(img_group, 0)
        else:
            img_group = [self.worker(img) for img in img_group]
            return torch.stack(img_group, 0)

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):  # (T, 3, 224, 224)
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor


class ZeroPad(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):
        if tensor.size(0) == self.max_len:
            return tensor

        n_pad = self.max_len - tensor.size(0)
        pad = torch.zeros(n_pad, tensor.size(1), tensor.size(2), tensor.size(3))
        tensor = torch.cat([tensor, pad], 0)  # (T, 3, 224, 224)
        return tensor

