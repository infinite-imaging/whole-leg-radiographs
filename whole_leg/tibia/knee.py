import numpy as np
from numpy.polynomial import Polynomial

from .utils import determine_mid_of_joint_x, get_top_width, get_min_y_per_x


def get_knee_level(mask, percentage_mid=0.5, percentage_side=0.15):

    min_x, max_x = get_top_width(mask)
    possible_indices = np.arange(np.floor(min_x), np.ceil(max_x))

    mid = int(len(possible_indices) / 2)

    num_pts_mid = int(len(possible_indices) * percentage_mid)
    num_pts_side = int(len(possible_indices) * percentage_side)
    lower_mid = int(mid - num_pts_mid / 2)
    upper_mid = int(mid + num_pts_mid / 2)

    x_vals = np.concatenate(
        [possible_indices[num_pts_side:lower_mid],
         possible_indices[upper_mid:-num_pts_side]])

    y_vals = np.array([get_min_y_per_x(mask, _x) for _x in x_vals])

    line = Polynomial.fit(
        x_vals,
        y_vals, 1,
        domain=[x_vals.min(), x_vals.max()])

    return line(possible_indices), possible_indices


def determine_mid_of_knee_x(mask):
    return determine_mid_of_joint_x(mask, get_top_width)


def determine_mid_of_knee(mask, femur_level, **kwargs):
    x = determine_mid_of_knee_x(mask)

    tibia_level = get_knee_level(mask, **kwargs)
    index_tibia = np.nonzero(tibia_level[1] == x)[0]
    index_tibia = int(np.mean(index_tibia))
    y_tibia_level = tibia_level[0][index_tibia]

    index_femur = np.nonzero(femur_level[1] == x)
    index_femur = int(np.mean(index_femur))
    y_femur_level = femur_level[0][index_femur]
    return y_femur_level + (y_tibia_level - y_femur_level) / 2, x
