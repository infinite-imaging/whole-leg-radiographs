import numpy as np
import math
from whole_leg.mask_utils import (
    get_contpt,
    get_lowest_mask_pt,
    calc_disc,
    shrink_points,
)
from whole_leg.bresenham_slope import bres
from whole_leg.least_squares import lsf
from skimage.transform import rotate as rot_img
from ..utils import angle_between


def rotate(origin, point, angle, deg=True):

    if deg:
        angle = np.deg2rad(angle)
    oy, ox = origin
    py, px = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qy, qx


def transform(pt, origin, angle, offset=None):
    new_pt = rotate(origin, pt, angle)

    if offset is not None:
        new_pt = np.array(new_pt) + np.array(offset)

    return new_pt


def find_notch(mask, min_y=None, percentage=0.1):
    if min_y is None:
        min_y = determine_min_y(mask, percentage)
    y, _ = get_lowest_mask_pt(mask)

    y = int(y)

    allowed = False
    notch = None, None

    while y >= min_y:
        disc = calc_disc(mask, y)
        print(disc)

        if isinstance(disc, np.ndarray) and len(disc):
            return_allowed = True

        else:
            if allowed:
                new_disc = calc_disc(mask, y + 1)
                notch = (y + 1, new_disc[int(len(new_disc) / 2)])

                allowed = False

        y -= 1

    return notch


def rotate_vertical(mask, return_angle=True, **kwargs):
    ls_fit = lsf(mask, **kwargs)

    ls_vec = [_lsfit[0] - _lsfit[-1] for _lsfit in ls_fit]

    vertical_dir = (-1, 0)

    angle = np.rad2deg(angle_between(ls_vec / np.linalg.norm(ls_vec), vertical_dir))

    rotated_image = rot_img(mask, angle, resize=True, preserve_range=True) > 0
    rotated_image = rotated_image.astype(np.uint8)

    if return_angle:
        return rotated_image, angle
    else:
        return rotated_image


def transform(pt, origin, angle, offset=None):
    new_pt = rotate(origin, pt, angle)

    if offset is not None:
        new_pt = np.array(new_pt) + np.array(offset)

    return new_pt


def determine_min_y(mask, percentage=0.1):
    contour_pts = get_contpt(mask)

    num_pts = int(len(contour_pts[0]) * percentage)

    sorted_y = np.sort(contour_pts[0])
    return sorted_y[-num_pts]


def calc_level_v1(mask, rot=False, return_notch=True):
    notch = find_notch(mask, determine_min_y(mask))

    if rot:
        rotated_mask, angle = rotate_vertical(mask)
    else:
        rotated_mask, angle = mask, 0

    rot_offset = np.array(
        [
            _rot_dim - _orig_dim
            for _rot_dim, _orig_dim in zip(rotated_mask.shape, mask.shape)
        ]
    )
    rot_center = (
        int((rotated_mask.shape[0] - 1) / 2),
        int((rotated_mask.shape[1] - 1) / 2),
    )

    notch_rot = transform(
        notch,
        (int((mask.shape[0] - 1) / 2), int((mask.shape[1] - 1) / 2)),
        angle,
        offset=rot_offset / 2,
    )

    offset_y = np.array([notch_rot[0], 0])
    offset_x = np.array([0, notch_rot[1]])

    notch_rot_int = notch_rot.astype(np.uint16)

    right_pt = (
        np.array(
            get_lowest_mask_pt(rotated_mask[notch_rot_int[0]:, : notch_rot_int[1]])
        )
        + offset_y
    )
    left_pt = (
        np.array(
            get_lowest_mask_pt(rotated_mask[notch_rot_int[0]:, notch_rot_int[1]:])
        )
        + offset_y
        + offset_x
    )

    rot_left = transform(left_pt, rot_center, -angle, offset=-rot_offset)
    rot_right = transform(right_pt, rot_center, -angle, offset=-rot_offset)

    if return_notch:
        return rot_left, rot_right, notch
    else:
        return rot_left, rot_right


def num_mask_points_on_line(mask, start, end, thresh_pt=None):
    pts_on_line = bres([start], end, -1).astype(np.uint16)

    if thresh_pt is not None:
        if start[1] < thresh_pt[1]:
            pts_on_line = np.array([pt for pt in pts_on_line if pt[1] >= thresh_pt[1]])
        else:
            pts_on_line = np.array([pt for pt in pts_on_line if pt[1] <= thresh_pt[1]])

    sum_val = 0
    for pt in pts_on_line:
        sum_val += mask[pt[0], pt[1]]

    return sum_val


def calc_level_v2(
    mask, rot=True, start=None, thresh=2, step=1, shrink=True, return_notch=True
):
    notch = find_notch(mask, determine_min_y(mask))

    if rot_img:
        rotated_mask, angle = rotate_vertical(mask)
    else:
        rotated_mask, angle = mask, 0

    rot_offset = np.array(
        [
            _rot_dim - _orig_dim
            for _rot_dim, _orig_dim in zip(rotated_mask.shape, mask.shape)
        ]
    )
    center = (
        int((rotated_mask.shape[0] - 1) / 2),
        int((rotated_mask.shape[1] - 1) / 2),
    )

    _rot = transform(
        notch,
        (int((mask.shape[0] - 1) / 2), int((mask.shape[1] - 1) / 2)),
        angle,
        offset=rot_offset / 2,
    )

    if start is None:
        start = get_lowest_mask_pt(rotated_mask)
    else:
        start = transform(
            start,
            (int((mask.shape[0] - 1) / 2), int((mask.shape[1] - 1) / 2)),
            angle,
            offset=rot_offset / 2,
        )

    if start[1] < _rot[1]:
        x_end = rotated_mask.shape[1] - 1
        rot_dir = -1

    else:
        x_end = 0
        rot_dir = 1

    end = (start[0], x_end)

    total_rot = 0
    while (
        num_mask_points_on_line(
            rotated_mask, start, rotate(start, end, total_rot), _rot
        )
        < thresh
    ):

        print(total_rot)

        total_rot += step * rot_dir

    final_end_pt = rotate(start, end, total_rot)

    final_end_orig = transform(final_end_pt, center, -angle, offset=-rot_offset / 2)
    start_orig = transform(start, center, -angle, offset=-rot_offset / 2)

    if shrink:
        _, new_end = shrink_points(mask, start_orig, final_end_orig)
        new_start = start_orig
    else:
        new_start = start_orig
        new_end = final_end_orig

    if return_notch:
        return new_start, new_end, notch
    else:
        return new_start, new_end
