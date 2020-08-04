import numpy as np
from ..mask_utils import get_contpt


def get_min_y_per_x(mask, x):
    return np.nonzero(mask[:, int(x)])[0].min()


def get_top_width(mask):
    contour_y, _ = get_contpt(mask)

    _, top_contour_x = get_contpt(mask[: int(np.mean(contour_y))])
    return top_contour_x.min(), top_contour_x.max()


def get_bottom_width(mask):
    contour_y, _ = get_contpt(mask)

    _, bottom_contour_x = get_contpt(mask[int(np.mean(contour_y)) :])
    return bottom_contour_x.min(), bottom_contour_x.max()


def determine_mid_of_joint_x(mask, width_fn):
    min_x, max_x = width_fn(mask)
    return int(min_x + (max_x - min_x) / 2)
