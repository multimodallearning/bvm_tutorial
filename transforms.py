import torch
from torch.nn import ConstantPad3d as Pad
import bvm_tutorial.utils
import warnings
warnings.filterwarnings("ignore")


class ZeroPad(object):
    """Pad tensors with zeros in z-direction for MSD02Heart data"""

    def __call__(self, sample):
        pad_front = int(sample['pad0']) + 15
        pad_back = int(sample['pad1']) + 15
        pad = Pad((0, 0, 0, 0, pad_front, pad_back), 0)
        sample['image'] = pad(sample['image'])
        sample['label'] = pad(sample['label'])
        return sample


class Crop(object):
    """Crop at absolute position"""

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x0 = x_min
        self.x1 = x_max
        self.y0 = y_min
        self.y1 = y_max
        self.z0 = z_min
        self.z1 = z_max

    def __call__(self, sample):
        sample['image'] = sample['image'][:, :, self.z0:self.z1,
                                                self.y0:self.y1,
                                                self.x0:self.x1]
        sample['label'] = sample['label'][:, :, self.z0:self.z1,
                                                self.y0:self.y1,
                                                self.x0:self.x1]
        return sample


class Scale(object):
    """Scale tensors spatially."""

    def __init__(self, width=1, height=1, depth=1):
        assert width or height or depth
        self.width = width
        self.height = height
        self.depth = depth

    def __call__(self, sample):
        sample['image'] = F.interpolate(
            sample['image'],
            scale_factor=(self.depth, self.height, self.width)
        )

        sample['label'] = F.interpolate(
            sample['label'].float(),
            scale_factor=(self.depth, self.height, self.width),
            mode='nearest'
        ).long()

        return sample


class AugmentAffine(object):
    """Author: Mattias P Heinrich"""

    def __init__(self, strength=0.05):
        self.strength = strength

    def __call__(self, sample):
        img, seg = augmentAffine(sample['image'], sample['label'], self.strength)
        sample['image'] = img
        sample['label'] = seg
        return sample


class ToCuda(object):
    def __call__(self, sample):
        sample['image'] = sample['image'].cuda()
        sample['label'] = sample['label'].cuda()
        return sample
