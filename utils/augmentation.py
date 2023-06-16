import numpy as np
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def get_cyto_transform(crop_size, padding, cutout_size, dtype):
    if dtype == 'BF':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif dtype == 'MIP':
        mean = 0.445
        std = 0.269

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            elastic_transform(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return {'train': transform_train, 'test': transform_test}

class elastic_transform:
    def __init__(self, p = 0.5, params = ((1, 1), (5, 2), (1, 0.5), (1, 3))):
        assert all([isinstance(i, tuple) for i in params])
        assert all(len(i) == 2 for i in params)
        self.params = params
        self.p = p

    def __call__(self, img):
        _img = np.array(img)

        if torch.rand(1) < self.p:

            shape = _img.shape
            alpha, sigma = random.choice(self.params)

            random_state = np.random.RandomState(None)

            dx = gaussian_filter((random_state.rand(shape[0], shape[1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(shape[0], shape[1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            # indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            return map_coordinates(_img, indices, order=1).reshape(shape)

        else:
            return _img
