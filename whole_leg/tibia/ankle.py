import numpy as np
from numpy.polynomial import Polynomial
from ..least_squares import lsf
from ..mask_utils import get_contpt
from .utils import get_bottom_width, determine_mid_of_joint_x


def calc_ankle_level(mask, vert_tol=0.025, horiz_tol=0.25):
    contour = get_contpt(mask)
    # transform relative tolerance to absolute pixel values
    vert_tol = vert_tol * (contour[0].max() - contour[0].min())

    lsf_fitting = lsf(mask)

    intersec = None

    contour_zipped = list(zip(*contour))

    for pt in zip(*lsf_fitting):

        if (int(pt[0]), int(pt[1])) in contour_zipped:
            intersec = pt

    # transform horizontal tolerance from relative to absolute
    y_level_pts = np.nonzero(mask[intersec[0]])[0]
    horiz_tol = horiz_tol * (y_level_pts.max() - y_level_pts.min())

    valid_pt = ([], [])
    min_y, max_y = intersec[0] - vert_tol, intersec[0] + vert_tol
    min_x, max_x = intersec[1] - horiz_tol, intersec[1] + horiz_tol

    for pt in zip(*contour):
        if min_y <= pt[0] <= max_y and min_x <= pt[1] <= max_x:
            valid_pt[0].append(pt[0])
            valid_pt[1].append(pt[1])

    line = Polynomial.fit(
        valid_pt[1], valid_pt[0], 1, domain=[contour[1].min(), contour[1].max()]
    )

    _, indices_level = get_contour_points(mask[int(np.mean(contour[0])):])
    indices_level = np.arange(indices_level.min(), indices_level.max())
    return line(indices_level), indices_level


def determine_mid_of_ankle_x(mask):
    return determine_mid_of_joint_x(mask, get_bottom_width)


def determine_mid_of_ankle_y(mask, leave_out_percentage=0.15):
    lsf = least_squares_fit(mask, leave_out_percentage=leave_out_percentage)
    contours = get_contour_points(mask)
    contour_zipped = list(zip(*contours))

    intersec = None
    for pt in zip(*lsf):

        if (int(pt[0]), int(pt[1])) in contour_zipped:
            intersec = pt

    return intersec[0]


def determine_mid_of_ankle(mask, leave_out_percentage=0.15):
    return (
        determine_mid_of_ankle_y(mask, leave_out_percentage=leave_out_percentage),
        determine_mid_of_ankle_x(mask),
    )
