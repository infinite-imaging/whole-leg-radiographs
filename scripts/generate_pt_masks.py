import torch
from delira_unet import UNetTorch
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import regionprops, label
from skimage.transform import resize
from batchgenerators.augmentations.normalizations import range_normalization
from whole_leg.streamlined_mask_processor import WholeLeg
import numpy as np
from PIL import Image
from skimage.io import imsave

import itertools
import numpy as np


def pts_exporter(pts, file_handle, **kwargs):

    if pts.shape[-1] == 2:
        pts = pts[:, [1, 0]] + 1
    else:
        pts = pts[:, [2, 1, 0]] + 1

    header = "version: 1\nn_points: {}\n{{".format(pts.shape[0])
    np.savetxt(
        file_handle,
        pts,
        delimiter=" ",
        header=header,
        footer="}",
        fmt="%.3f",
        comments="",
    )


def angle_with_start(coord, start):
    vec = coord - start
    return np.angle(np.complex(vec[0], vec[1]))


def sort_clockwise(points):

    mean = np.array(points).mean(axis=0)

    points = sorted(
        points, key=lambda coord: angle_with_start(coord, mean), reverse=True
    )

    return points[::-1]


def get_closest_contour_pt(contour_pts, pt, return_idx=False):
    contour_pts = [tuple(_pt) for _pt in contour_pts]
    pt = tuple(pt)
    if pt in contour_pts:
        if return_idx:
            idx = list(contour_pts).index(pt)
            return pt, idx
        return pt

    closest_pt, min_dist, min_idx = None, float("inf"), None

    for idx, cpt in enumerate(contour_pts):
        distance = np.sqrt((cpt[0] - pt[0]) ** 2 + (cpt[1] - pt[1]) ** 2)

        if distance < min_dist:
            closest_pt = cpt
            min_dist = distance
            min_idx = idx

    assert closest_pt is not None

    if return_idx:
        return closest_pt, min_idx

    return closest_pt


def crop_img(img, segs=None, contains_left=True, contains_right=True, crop=0.6):

    result = {}
    offsets = {}
    seg_results = {}

    if contains_left and contains_right:

        num_pixels = int(img.shape[1] * img.shape)
        result["right"] = img[:, :num_pixels]
        result["left"] = img[:, -num_pixels:]

        if segs is not None:
            seg_results["left"] = segs[:, -num_pixels:]
            seg_results["right"] = segs[:, num_pixels:]

        offsets["left"] = num_pixels
        offsets["right"] = 0

    elif contains_left and not contains_right:
        result["left"] = img
        offsets["left"] = 0
        if segs is not None:
            seg_results["left"] = segs

    elif contains_right and not contains_left:
        result["right"] = img
        offsets["right"] = 0
        if segs is not None:
            seg_results["right"] = segs

    if segs is None:
        return result, offsets
    else:
        return result, offsets, seg_results


def process_patientv2(
    img, segs, is_left, num_pts=250, fill_holes=True, return_img=False
):
    mask_background, mask_femur, mask_tibia = segs

    if fill_holes:
        mask_femur = binary_fill_holes(mask_femur)
        mask_tibia = binary_fill_holes(mask_tibia)

    if not is_left:
        mask_femur = mask_femur[..., ::-1]
        mask_tibia = mask_tibia[..., ::-1]
        mask_background = mask_background[..., ::-1]
        img = img[..., ::-1]

    processor = WholeLeg(
        img, mask=np.array([mask_background, mask_femur, mask_tibia]), is_left=True
    )
    points_femur, points_tibia = [], []

    points_femur = [
        processor.femur.mechanical_axis[0],
        processor._convert_separate_coords_to_pt_tuples(processor.femur.circle_points)[
            0
        ],
        processor._convert_separate_coords_to_pt_tuples(processor.femur.circle_points)[
            -1
        ],
    ]

    # determine order of knee level points
    if processor.femur.apex_of_femural_notch[1] > processor.femur.shrinked_level[0][1]:
        points_femur += [
            processor.femur.shrinked_level[1],
            processor.femur.apex_of_femural_notch,
            processor.femur.shrinked_level[0],
        ]
    else:
        points_femur += [
            processor.femur.shrinked_level[0],
            processor.femur.apex_of_femural_notch,
            processor.femur.shrinked_level[1],
        ]

    points_tibia = [processor.tibia.mid_1]

    if processor.tibia.mid_1[1] > processor.tibia.level1[0][1]:
        order1 = [1, 0]
    else:
        order1 = [0, 1]

    if processor.tibia.mid_2[1] > processor.tibia.level2[0][1]:
        order2 = [1, 0]
    else:
        order2 = [0, 1]

    points_tibia += [
        processor.tibia.level1[order1[0]],
        processor.tibia.level2[order2[0]],
        processor.tibia.mid_2,
        processor.tibia.level2[order2[1]],
        processor.tibia.leve1[order1[1]],
    ]

    contour_femur = np.array(list(zip(*processor.femur.cont_pts)))
    contour_tibia = np.array(list(zip(*processor.tibia.cont_pts)))
    contour_femur = sort_clockwise(contour_femur)
    contour_tibia = sort_clockwise(contour_tibia)

    points_tibia = [get_closest_contour_pt(contour_tibia, pt) for pt in points_tibia]
    points_femur = [get_closest_contour_pt(contour_femur, pt) for pt in points_femur]

    total_points_femur, total_points_tibia = [*points_femur], [*points_tibia]

    idxs_femur = np.round(np.linspace(0, len(contour_femur) - 1, num_pts)).astype(int)
    idxs_tibia = np.round(np.linspace(0, len(contour_tibia) - 1, num_pts)).astype(int)

    total_points_femur += np.array(contour_femur)[idxs_femur].tolist()
    total_points_tibia += np.array(contour_tibia)[idxs_tibia].tolist()

    if return_img:
        raise NotImplementedError
    else:
        return total_points_femur, total_points_tibia


def parallel_wrapper(args, save_path_femur, save_path_tibia, num_pts):
    idx, sample = args
    mask = sample["label"]

    mask = np.concatenate([mask == 0, mask == 1, mask == 2])
    img = (sample["data"] * 255).astype(np.uint8)

    is_left = not sample["is_left"]

    try:
        pts_femur, pts_tibia = process_patientv2(
            img, segs=mask, is_left=not is_left, num_pts=num_pts, return_img=False
        )

        pts_exporter(
            np.array(pts_femur), os.path.join(save_path_femur, "sample_%03d.pts" % idx)
        )
        pts_exporter(
            np.array(pts_tibia), os.path.join(save_path_tibia, "sample_%03d.pts" % idx)
        )

        if is_left:
            img = sample["data"][0, ..., ::-1]
        else:
            img = sample["data"][0]

        imsave(os.path.join(save_path_femur, "sample_%03d.png" % idx), img)
        imsave(os.path.join(save_path_tibia, "sample_%03d.png" % idx), img)
    except Exception as e:
        print(e)

    finally:
        print(idx)


if __name__ == "__main__":
    from delira_unet import UNetTorch
    import os
    from tqdm import tqdm
    from delira import set_debug_mode, get_current_debug_mode
    from whole_leg import (
        WholeLegDataset,
        draw_mask_overlay,
        combine_gt_pred_masks,
        HistogramEqualization,
        CopyTransform,
        AddGridTransform,
        SingleBoneDataset,
    )
    from delira.training import Predictor
    import numpy as np
    from batchgenerators.transforms import RangeTransform, Compose
    from delira.training.backends import convert_torch_to_numpy
    from functools import partial
    from delira.data_loading import DataManager, SequentialSampler
    from functools import partial
    from multiprocessing import Pool

    set_debug_mode(False)

    data_path = ""
    save_path = ""
    split = "Train"
    num_pts = 250

    data_path = os.path.join(data_path, split)
    save_path_femur = os.path.join(save_path, "Femur")
    save_path_tibia = os.path.join(save_path, "Tibia")
    save_path_femur = os.path.join(save_path_femur, "%03d_pts" % num_pts, split)
    save_path_tibia = os.path.join(save_path_tibia, "%03d_pts" % num_pts, split)
    os.makedirs(save_path_femur, exist_ok=True)
    os.makedirs(save_path_tibia, exist_ok=True)

    print("Load Data")
    dset = WholeLegDataset(root_path=data_path, include_flipped=False, img_size=None)

    func = partial(
        parallel_wrapper,
        save_path_femur=save_path_femur,
        save_path_tibia=save_path_tibia,
        num_pts=num_pts,
    )

    if get_current_debug_mode():
        for args in enumerate(dset):
            func(args)
    else:
        with Pool() as p:
            p.map(func, enumerate(dset))

