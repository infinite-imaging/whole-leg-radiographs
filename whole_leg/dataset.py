from delira.data_loading import AbstractDataset
import os
from delira.utils import subdirs
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from delira import get_current_debug_mode
from functools import partial
from multiprocessing import Pool
from scipy.ndimage.morphology import (
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
)
from skimage.morphology import disk


def load_sample_psinet(path, img_size, flip=False, crop_percentage=0.6, contourwidth=5):

    img = imread(os.path.join(path, "image.png"))
    mask_femur = imread(os.path.join(path, "mask_femur.png"))
    mask_tibia = imread(os.path.join(path, "mask_tibia.png"))

    # total_mask = np.array([mask_femur.astype(np.uint8),
    #                        mask_tibia.astype(np.uint8)]
    is_left = "left" in path
    width = int(img.shape[1] * crop_percentage)

    if is_left:
        img = img[:, :width]
        mask_femur = mask_femur[:, :width]
        mask_tibia = mask_tibia[:, :width]
    else:
        img = img[:, -width:]
        mask_femur = mask_femur[:, -width:]
        mask_tibia = mask_tibia[:, -width:]

    old_size = img.shape
    if img_size is not None:
        img = resize(img, img_size)
        mask_femur = resize(mask_femur, img_size)
        mask_tibia = resize(mask_tibia, img_size)

    total_mask = np.zeros_like(mask_femur)
    total_mask[mask_femur.astype(np.bool)] = 1
    total_mask[mask_tibia.astype(np.bool)] = 2
    total_mask = total_mask.astype(np.uint8)

    contour_femur = binary_dilation(
        binary_erosion(mask_femur) - mask_femur, disk(contourwidth)
    )
    contour_tibia = binary_dilation(
        binary_erosion(mask_tibia) - mask_tibia, disk(contourwidth)
    )

    total_contour = np.zeros_like(contour_femur)
    total_contour[contour_femur.astype(np.bool)] = 1
    total_contour[contour_tibia.astype(np.bool)] = 2
    total_contour = total_contour.astype(np.uint8)

    distance_background = distance_transform_edt(total_mask == 0)
    distance_femur = distance_transform_edt(mask_femur)
    distance_tibia = distance_transform_edt(mask_tibia)
    total_distance_map = np.array([distance_background, distance_femur, distance_tibia])

    if flip:
        img = np.fliplr(img)
        total_mask = np.fliplr(total_mask)
        total_contour = np.fliplr(total_contour)
        total_distance_map = np.moveaxis(
            np.fliplr(np.moveaxis(total_distance_map, 0, -1)), -1, 0
        )

        is_left = not is_left

    return {
        "data": img[None, ...],
        "label": total_mask[None, ...],
        "contour": total_contour[None, ...],
        "distance": total_distance_map,
        "is_left": np.array([is_left]),
        "img_size": old_size,
    }


def load_sample(path, img_size, flip=False, crop_percentage=0.6, contourwidth=None):

    img = imread(os.path.join(path, "image.png"))
    mask_femur = imread(os.path.join(path, "mask_femur.png"))
    mask_tibia = imread(os.path.join(path, "mask_tibia.png"))

    # total_mask = np.array([mask_femur.astype(np.uint8),
    #                        mask_tibia.astype(np.uint8)]
    is_left = "left" in path
    width = int(img.shape[1] * crop_percentage)

    if is_left:
        img = img[:, :width]
        mask_femur = mask_femur[:, :width]
        mask_tibia = mask_tibia[:, :width]
    else:
        img = img[:, -width:]
        mask_femur = mask_femur[:, -width:]
        mask_tibia = mask_tibia[:, -width:]

    old_size = img.shape
    if img_size is not None:
        img = resize(img, img_size)
        mask_femur = resize(mask_femur, img_size)
        mask_tibia = resize(mask_tibia, img_size)

    for i in range(10):
        mask_femur = binary_dilation(binary_erosion(mask_femur))
        mask_tibia = binary_dilation(binary_erosion(mask_tibia))

    if contourwidth is not None:
        mask_femur = (
            binary_erosion(mask_femur, iterations=contourwidth, brute_force=True)
            - mask_femur
        )
        mask_tibia = (
            binary_erosion(mask_tibia, iterations=contourwidth, brute_force=True)
            - mask_tibia
        )

    total_mask = np.zeros_like(mask_femur, dtype=np.uint8)
    total_mask[mask_femur.astype(np.bool)] = 1
    total_mask[mask_tibia.astype(np.bool)] = 2
    total_mask = total_mask.astype(np.uint8)

    if flip:
        img = np.fliplr(img)
        total_mask = np.fliplr(total_mask)
        is_left = not is_left

    return {
        "data": img[None, ...],
        "label": total_mask[None, ...],
        "is_left": np.array([is_left]),
        "img_size": old_size,
    }


def load_single_bone(
    path,
    img_size,
    flip=False,
    crop_percentage=0.6,
    bone_label=1,
    crop_percentage_height=0.1,
    size_after_cropping=None,
):

    if crop_percentage_height is not None and size_after_cropping is None:
        size_after_cropping = (int(img_size[0] / 2), *img_size[1:])

    sample = load_sample(
        path=path, img_size=img_size, flip=flip, crop_percentage=crop_percentage
    )

    bone_mask = sample["label"] == bone_label
    img = sample["data"]
    y_indices = []
    for idx in range(bone_mask.shape[1]):
        if bone_mask[:, idx, ...].any():
            y_indices.append(idx)

    if crop_percentage_height is not None:
        num_crop_idxs = int(len(y_indices) * crop_percentage_height)

        min_idx = max(0, min(y_indices) - num_crop_idxs)
        max_idx = min(bone_mask.shape[1] - 1, max(y_indices) + num_crop_idxs)

        bone_mask = bone_mask[:, min_idx : max_idx + 1, ...]
        img = img[:, min_idx : max_idx + 1, ...]

        bone_mask = resize(bone_mask[0], size_after_cropping)[None, ...]
        img = resize(img[0], size_after_cropping)[None, ...]

    sample["data"] = img
    sample["label"] = bone_mask

    return sample


def multiproc_fn(args, img_size, crop_percentage=0.6, func=load_sample, **kwargs):
    return func(
        path=args[0],
        img_size=img_size,
        flip=args[1],
        crop_percentage=crop_percentage,
        **kwargs
    )


class WholeLegDataset(AbstractDataset):
    def __init__(
        self,
        root_path,
        include_flipped=True,
        lazy=False,
        img_size=None,
        load_fn=None,
        **kwargs
    ):

        if load_fn is None:
            load_fn = load_sample
        super().__init__(root_path, load_fn)
        self._include_flipped = include_flipped
        self.lazy = lazy
        self._img_size = img_size
        self._add_kwargs = kwargs
        self.data = self._make_dataset(self.data_path)

    def _make_dataset(self, path: str):

        patients = subdirs(path)

        sub_dirs = []

        for pat in patients:
            sub_dirs += [x for x in subdirs(pat)]

        sub_dirs = sorted(sub_dirs)
        sub_dirs_not_flipped = [(x, False) for x in sub_dirs]
        if self._include_flipped:
            sub_dirs_flipped = [(x, True) for x in sub_dirs]
        else:
            sub_dirs_flipped = []

        sub_dirs = sub_dirs_not_flipped + sub_dirs_flipped

        if not self.lazy:
            if get_current_debug_mode():
                return [
                    self._load_fn(
                        tmp[0], self._img_size, flip=tmp[1], **self._add_kwargs
                    )
                    for tmp in sub_dirs
                ]
            else:
                func = partial(
                    multiproc_fn,
                    img_size=self._img_size,
                    func=self._load_fn,
                    **self._add_kwargs
                )

                with Pool() as p:
                    return p.map(func, sub_dirs)

        return sub_dirs

    def __getitem__(self, item):
        sample = self.data[item]

        if self.lazy:
            sample = self._load_fn(
                sample[0], self._img_size, flip=sample[1], **self._add_kwargs
            )

        return sample

    def __len__(self):
        return len(self.data)


class SingleBoneDataset(WholeLegDataset):
    def __init__(
        self, root_path, include_flipped=True, lazy=False, img_size=None, bone_label=1
    ):
        super().__init__(
            root_path,
            include_flipped=include_flipped,
            lazy=lazy,
            img_size=img_size,
            load_fn=partial(load_single_bone, bone_label=bone_label),
        )


class MultiObjectiveDataset(WholeLegDataset):
    def __init__(
        self, root_path, include_flipped=True, lazy=False, img_size=None, **kwargs
    ):
        super().__init__(
            root_path,
            include_flipped=include_flipped,
            lazy=lazy,
            img_size=img_size,
            load_fn=load_sample_psinet,
            **kwargs
        )

