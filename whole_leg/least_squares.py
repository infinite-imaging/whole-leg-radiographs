from numpy.polynomial import Polynomial
import numpy as np
from .mask_utils import get_contpt


def lsf(mask, leave_out_percentage=(0.15, 0.15)):

    if isinstance(leave_out_percentage, (float, int)):
        bottom_percentage = top_percentage = leave_out_percentage
    else:
        top_percentage, bottom_percentage = leave_out_percentage

    contour_points = get_contpt(mask)

    if leave_out_percentage:

        num_pts_top = int(len(contour_points[0]) * top_percentage)
        num_pts_bottom = int(len(contour_points[0]) * bottom_percentage)

        indices = np.argsort(contour_points[0])
        contour_points = (contour_points[0][indices], contour_points[1][indices])

        contour_points_filtered = (
            contour_points[0][num_pts_top:-num_pts_bottom],
            contour_points[1][num_pts_top:-num_pts_bottom],
        )
    else:
        contour_points_filtered = contour_points

    line = Polynomial.fit(
        contour_points_filtered[0],
        contour_points_filtered[1],
        1,
        domain=[contour_points[0].min(), contour_points[0].max()],
    )

    return contour_points[0], line(contour_points[0])

