from batchgenerators.transforms import AbstractTransform
from skimage.transform import resize, rotate
import numpy as np
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.filters import rank
from copy import deepcopy
from scipy.ndimage.morphology import binary_erosion


class ContourTransform(AbstractTransform):
    def __init__(self, contour_width=10, keys=("label",)):
        self.keys = keys
        self.contour_width = contour_width

    def __call__(self, **data_dict):
        for key in self.keys:
            data_dict[key] = (
                binary_erosion(
                    data_dict[key], iterations=self.contour_width, brute_force=True
                )
                - data_dict[key]
            )

        return data_dict


class ResizeTransform(AbstractTransform):
    def __init__(self, size: tuple, data_key="data", seg_key="label"):

        super().__init__()
        self._size = size
        self._data_key = data_key
        self._seg_key = seg_key

    @staticmethod
    def _resize(image_batch, size):

        image_batch = np.moveaxis(image_batch, 1, -1)

        new_image_batch = []

        for img in image_batch:
            new_image_batch.append(resize(img, size))

        new_image_batch = np.array(new_image_batch)
        return np.moveaxis(new_image_batch, -1, 1)

    def __call__(self, **data_dict):
        for key in (self._data_key, self._seg_key):
            data_dict[key] = self._resize(data_dict[key], self._size)

        return data_dict


class HistogramEqualization(AbstractTransform):
    def __init__(self, data_key="data", selem: np.ndarray = None, per_channel=True):

        self.selem = selem
        self.data_key = data_key
        self.per_channel = per_channel

    def _equalize(self, sample):

        sample_ubyte = img_as_ubyte(sample)

        if self.selem is None:
            sample_eq = exposure.equalize_hist(sample_ubyte)
        else:
            sample_eq = rank.equalize(sample_ubyte, selem=self.selem)

        return sample_eq

    def __call__(self, **data):
        new_data = []

        for sample in data[self.data_key]:
            if self.per_channel:
                sample_eq = np.array([self._equalize(_sample) for _sample in sample])
            else:
                sample = np.moveaxis(sample, 0, -1)
                sample_eq = self._equalize(sample)
                sample_eq = np.moveaxis(sample_eq, -1, 0)

            new_data.append(sample_eq)

        data[self.data_key] = np.array(new_data)
        return data


class CopyTransform(AbstractTransform):
    def __init__(self, src_key, dst_key):
        self.src_key = src_key
        self.dst_key = dst_key

    def __call__(self, **data):
        data[self.dst_key] = deepcopy(data[self.src_key])
        return data


class AddGridTransform(AbstractTransform):
    def __init__(self, keys=("data",)):
        self._keys = keys

    def __call__(self, **data):
        for key in self._keys:
            new_batch = []
            for img in data[key]:
                y = np.tile(np.arange(img.shape[1]), (img.shape[2], 1)).T
                x = np.tile(np.arange(img.shape[2]), (img.shape[1], 1))
                grid = np.array([y, x])
                new_batch.append(np.concatenate([img, grid]))

            data[key] = np.array(new_batch)

        return data
