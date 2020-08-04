import numpy as np
from skimage.morphology import binary_erosion
from .bresenham_slope import bres


def get_cont(mask):
    mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8)
    eroded_mask = binary_erosion(mask)
    diff_mask = mask - eroded_mask

    return diff_mask


def get_contpt(mask):
    diff_mask = get_cont(mask)
    return np.nonzero(diff_mask)


def get_side_contour_pts(mask, y: int):
    nonzero = np.nonzero(mask[y])[0]
    return nonzero.min(), nonzero.max()


def calc_disc(mask, y: int):
    side_pts = get_side_contour_pts(mask, y)

    return np.nonzero(1 - mask[y, side_pts[0] : side_pts[1]])[0] + side_pts[0]


def get_lowest_mask_pt(mask):
    pts = get_contpt(mask)
    indices = np.argsort(pts[0])

    return pts[0][indices[-1]], pts[1][indices[-1]]


def get_high_pt(mask):
    pts = get_contpt(mask)
    indices = np.argsort(pts[0])
    return pts[0][indices[0]], pts[1][indices[0]]


def shrink_points(mask, start_pt, end_pt):
    points_on_line = bres([start_pt], end_pt, -1)

    shrinked_start, shrinked_end = None, None

    for pt in points_on_line:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_start = pt
            break

    for pt in points_on_line[::-1]:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_end = pt
            break

    return shrinked_start, shrinked_end
