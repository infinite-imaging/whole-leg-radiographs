from numpy.polynomial import Polynomial
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes
from collections import MutableMapping
from .mask_utils import (
    get_lowest_mask_pt,
    get_high_pt,
    calc_disc,
    shrink_points,
    get_contpt,
    get_cont,
)
from .least_squares import lsf
from .utils import angle_between
import math
from .bresenham_slope import bres
from skimage.transform import rotate as rot_img
from PIL import ImageDraw, ImageFont


def angle_format_decorator(func):
    def wrapper(*args, **kwargs):
        ret_val = func(*args, **kwargs)

        ret_val = np.rad2deg(ret_val)
        ret_val = ret_val % 360
        if ret_val > 180:
            ret_val = 360 - ret_val

        if ret_val > 90:
            ret_val = 180 - ret_val

        return ret_val

    return wrapper


def np_array_wrapper(func):
    def wrapper(*args, **kwargs):
        return np.array(func(*args, **kwargs))

    return wrapper


def filter_contpt(points, percentage, dim=0):
    if isinstance(percentage, (int, float)):
        percentage_top = percentage_bottom = percentage
    else:
        percentage_top, percentage_bottom = percentage

    pts_top = int(percentage_top * len(points[dim]))
    pts_bottom = int(percentage_bottom * len(points[dim]))

    indices = np.argsort(points[dim])[pts_top:-pts_bottom]

    points = np.array([_points[indices] for _points in points])
    return points


def find_notch(min_y, lowest_mask_pt, disc):
    y, _ = lowest_mask_pt
    y = int(y)

    return_allowed = False
    notch = None, None

    while y >= min_y:
        _disc = disc[y]

        if isinstance(_disc, np.ndarray) and len(_disc):
            return_allowed = True

        else:
            if return_allowed:
                new_disc = disc[y + 1]
                notch = (y + 1, new_disc[int(len(disc) / 2)])

                return_allowed = False

        y -= 1

    return notch, disc


def determine_min_y(mask, percentage=0.1):
    contour_pts = get_contpt(mask)

    num_pts = int(len(contour_pts[0]) * percentage)

    sorted_y = np.sort(contour_pts[0])
    return sorted_y[-num_pts]


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


def num_mask_line(mask, start, end, thresh_pt=None):
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


def calc_level_femur(
    mask, notch, lowest_mask, angle=0, thresh=2, step_size=1, offset=(0, 0)
):
    offset = np.array(offset)
    rot_center = (int((mask.shape[0] - 1) / 2), int((mask.shape[1] - 1) / 2))

    if angle:
        notch_rot = transform(
            notch,
            (int((mask.shape[0] - 1) / 2), int((mask.shape[1] - 1) / 2)),
            angle,
            offset=offset / 2,
        )
    else:
        notch_rot = notch

    start = lowest_mask

    if start[1] < notch_rot[1]:
        x_end = mask.shape[1] - 1
        rot_dir = -1

    else:
        x_end = 0
        rot_dir = 1

    end = (start[0], x_end)

    total_rot = 0
    while num_mask_line(mask, start, rotate(start, end, total_rot), notch_rot) < thresh:
        total_rot += step_size * rot_dir

    final_end = rotate(start, end, total_rot)

    if angle:
        final_end_orig = transform(final_end, rot_center, -angle, offset=-offset / 2)
        start_orig = transform(start, rot_center, -angle, offset=-offset / 2)
    else:
        final_end_orig = final_end
        start_orig = start

    return start_orig, final_end_orig


def determine_circ(contpts, angle, thresh_x, is_left, rot_center, percentage):
    if is_left:
        angle = abs(angle)
        indices_valid = contpts[1] <= thresh_x
    else:
        angle = -abs(angle)
        indices_valid = contpts[1] >= thresh_x

    filtered_contour_points = (contpts[0][indices_valid], contpts[1][indices_valid])

    rotated_y, rotated_x = [], []

    for _y, _x in zip(*filtered_contour_points):
        _rot_y, _rot_x = rotate(rot_center, (_y, _x), angle)
        rotated_y.append(_rot_y)
        rotated_x.append(_rot_x)

    rotated_y, rotated_x = np.array(rotated_y), np.array(rotated_x)

    indices = np.argsort(rotated_y)[: int(percentage * len(rotated_y))]
    return (filtered_contour_points[0][indices], filtered_contour_points[1][indices])


def fit_circle(y, x):
    x_m = np.mean(x)
    y_m = np.mean(y)

    u = x - x_m
    v = y - y_m

    s_uv = sum(u * v)
    s_uu = sum(u ** 2)
    s_vv = sum(v ** 2)
    s_uuv = sum(u ** 2 * v)
    s_uvv = sum(u * v ** 2)
    s_uuu = sum(u ** 3)
    s_vvv = sum(v ** 3)

    a = np.array([[s_uu, s_uv], [s_uv, s_vv]])
    b = np.array([s_uuu + s_uvv, s_vvv + s_uuv]) / 2.0
    uc, vc = np.linalg.solve(a, b)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    radius = np.mean(np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2))

    return yc_1, xc_1, radius


def calc_level_tibia2(mask, contour_pts, lsf, vert_tol=0.025, horiz_tol=0.25):
    vert_tol = vert_tol * (contour_pts[0].max() - contour_pts[0].min())

    intersec = None

    contour_zipped = list(zip(*contour_pts))

    for pt in zip(*lsf):

        if (int(pt[0]), int(pt[1])) in contour_zipped:
            intersec = pt

    # transform horizontal tolerance from relative to absolute
    y_level_pts = np.nonzero(mask[int(intersec[0])])[0]
    horiz_tol = horiz_tol * (y_level_pts.max() - y_level_pts.min())

    valid_pt = ([], [])
    min_y, max_y = intersec[0] - vert_tol, intersec[0] + vert_tol
    min_x, max_x = intersec[1] - horiz_tol, intersec[1] + horiz_tol

    for pt in zip(*contour_pts):
        if min_y <= pt[0] <= max_y and min_x <= pt[1] <= max_x:
            valid_pt[0].append(pt[0])
            valid_pt[1].append(pt[1])

    line = Polynomial.fit(
        valid_pt[1], valid_pt[0], 1, domain=[contour_pts[1].min(), contour_pts[1].max()]
    )

    _, indices_level = get_contpt(mask[int(np.mean(contour_pts[0])) :])
    indices_level = np.arange(indices_level.min(), indices_level.max())
    return line(indices_level), indices_level


def get_min_y_per_x(contour_pts, x):
    candidate_indices = contour_pts[1] == int(x)
    return contour_pts[0][candidate_indices].min()


def top_width(contour_pts):
    contour_y, contour_x = contour_pts

    indices = contour_y <= np.mean(contour_y)
    top_contour_x = contour_x[indices]

    return top_contour_x.min(), top_contour_x.max()


def bottom_width(contour_pts):
    contour_y, contour_x = contour_pts

    indices = contour_y >= np.mean(contour_y)
    bottom_contour_x = contour_x[indices]

    return bottom_contour_x.min(), bottom_contour_x.max()


def _determine_mid_x(contour_pts, width_fn):
    min_x, max_x = width_fn(contour_pts)
    return int(min_x + (max_x - min_x) / 2)


def calc_level_tibia1(contour_pts, percentage_mid=0.5, percentage_side=0.15):
    min_x, max_x = top_width(contour_pts)
    possible_indices = np.arange(np.floor(min_x), np.ceil(max_x))

    mid = int(len(possible_indices) / 2)

    num_pts_mid = int(len(possible_indices) * percentage_mid)
    num_pts_side = int(len(possible_indices) * percentage_side)
    lower_mid = int(mid - num_pts_mid / 2)
    upper_mid = int(mid + num_pts_mid / 2)

    x_vals = np.concatenate(
        [
            possible_indices[num_pts_side:lower_mid],
            possible_indices[upper_mid:-num_pts_side],
        ]
    )

    y_vals = np.array([get_min_y_per_x(contour_pts, _x) for _x in x_vals])

    # actual fitting
    line = Polynomial.fit(x_vals, y_vals, 1, domain=[x_vals.min(), x_vals.max()])

    return line(possible_indices), possible_indices


def determine_mid_x1(contour_pts):
    return _determine_mid_x(contour_pts, top_width)


def determine_mid1(contour_pts, tibia_level_line, femur_level_line=None):
    if femur_level_line is None:
        femur_level_line = tibia_level_line

    x = determine_mid_x1(contour_pts)

    index_tibia = np.nonzero(tibia_level_line[1] == x)[0]
    index_tibia = int(np.mean(index_tibia))
    y_tibia_level = tibia_level_line[0][index_tibia]

    index_femur = np.nonzero(femur_level_line[1] == x)
    index_femur = int(np.mean(index_femur))
    y_femur_level = femur_level_line[0][index_femur]
    return y_femur_level + (y_tibia_level - y_femur_level) / 2, x


def determine_mid_x2(contour_pts):
    return _determine_mid_x(contour_pts, bottom_width)


def determine_mid_y2(contour_pts, lsf):
    contour_zipped = list(zip(*contour_pts))

    intersec = None
    for pt in zip(*lsf):
        if (int(pt[0]), int(pt[1])) in contour_zipped:
            intersec = pt

    return intersec[0]


def determine_mid_tibia2(contour_pts, lsf):
    return (determine_mid_y2(contour_pts, lsf), determine_mid_x2(contour_pts))


def get_intersec(lsf, contour_pts):
    zipped_contour = list(zip(*contour_pts))
    for pt in zip(*lsf):
        if (int(pt[0]), int(pt[1])) in zipped_contour:
            return int(pt[0]), int(pt[1])


def get_trochmaj(contour_pts, lsf, femur_center):
    if femur_center[1] > lsf[1][0]:
        valid_idxs = contour_pts[1] < lsf[1][0]
    else:
        valid_idxs = contour_pts[1] > lsf[1][0]

    filtered_contour = contour_pts[0][valid_idxs], contour_pts[1][valid_idxs]
    pt_idx = np.argsort(filtered_contour[0])[0]

    return filtered_contour[0][pt_idx], filtered_contour[1][pt_idx]


class DiscMap(MutableMapping):
    def __init__(self, mask):
        self.store = dict()
        self.mask = mask

    def __getitem__(self, key):
        if key not in self.store:
            self.store[key] = calc_disc(self.mask, key)

        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class BoneProxy(object):
    def __init__(
        self, mask, is_left, lsf_leave_out_percentage=(0.25, 0.15), fill_holes=True
    ):

        if fill_holes:
            mask = binary_fill_holes(mask).squeeze().astype(np.uint8)
        self.mask = mask

        self.lsf_percentage = lsf_leave_out_percentage
        self.is_left = is_left

        self._cont = None
        self._cont = None
        self._lsf = None
        self._lsf_line = None
        self._anatomical_axis = None
        self._mechanical_axis = None
        self._mechanical_axis_line = None
        self._lowest_mask_pt = None
        self._highest_mask_pt = None
        self._filtered_cont_pts = {}
        self.discs = DiscMap(self.mask)

    @property
    @np_array_wrapper
    def cont(self):
        if self._cont is None:
            self._cont = get_cont(self.mask)

        return self._cont

    @property
    @np_array_wrapper
    def cont_pts(self):
        if self._cont_pts is None:
            self._cont_pts = get_contpt(self.cont)

        return self._cont_pts

    def filtered_contour_pts(self, dim=0):
        if dim not in self._filtered_cont_pts:
            self._filtered_contour_pts[dim] = filter_contpt(
                points=self.cont_pts, percentage=self.lsf_percentage
            )

        return self._filtered_cont_pts[dim]

    @property
    @np_array_wrapper
    def lsf(self):
        if self._lsf is None:
            self._lsf = (
                (self.lsf_line[0][0], self.lsf_line[1][0]),
                (self.lsf_line[0][-1], self.lsf_line[1][-1]),
            )

        return self._lsf

    @property
    @np_array_wrapper
    def lsf_line(self):
        if self._lsf_line is None:
            filtered_pts = self.filtered_contour_pts(0)
            tmp = lsf(
                filtered_pts[0],
                filtered_pts[1],
                self.self.cont_pts[0],
                domain=[self.cont_pts[0].min(), self.cont_pts[0].max()],
            )
            self._lsf_line = tmp[1], tmp[0]
        return self._lsf_line

    @property
    @np_array_wrapper
    def lowest_mask_pt(self):
        if self._lowest_mask_pt is None:
            self._lowest_mask_pt = get_lowest_mask_pt(self.mask)

        return self._lowest_mask_pt

    @property
    @np_array_wrapper
    def highest_mask_pt(self):
        if self._highest_mask_pt is None:
            self._highest_mask_pt = get_high_pt(self.mask)

        return self._highest_mask_pt

    @property
    @np_array_wrapper
    def anatomical_axis(self):
        return self.lsf

    @property
    @np_array_wrapper
    def anatomical_axis_line(self):
        return self.lsf_line

    @property
    def mechanical_axis(self):
        raise NotImplementedError

    @property
    @np_array_wrapper
    def mechanical_axis_line(self):
        raise NotImplementedError


class Femur(BoneProxy):
    def __init__(
        self,
        mask,
        is_left,
        lsf_leave_out_percentage=(0.25, 0.15),
        notch_percentage=0.1,
        rotate_mask=False,
        level_rot_thresh=2,
        level_rot_stepsize=1,
        circle_rot_angle=35,
        percentage_circle_pts=0.05,
        fill_holes=True,
    ):
        super().__init__(
            mask,
            is_left=is_left,
            lsf_leave_out_percentage=lsf_leave_out_percentage,
            fill_holes=fill_holes,
        )

        self._apex_of_femural_notch = None
        self._level = None
        self._shrinked_level = None
        self._level_line = None
        self._shrinked_level_line = None
        self._circle_points = None
        self._circle_params = None
        self._notch_percentage = notch_percentage
        self._level_rot_thresh = level_rot_thresh
        self._level_rot_stepsize = level_rot_stepsize
        self._circle_pt_percentage = percentage_circle_pts
        self._circle_rot_angle = circle_rot_angle
        self._trochanter_major = None

        if rotate_mask:
            self._rot_mask, self._rot_angle = rotate_vertical(
                self.mask, return_angle=True
            )
        else:
            self._rot_mask = self.mask
            self._rot_angle = 0

        self._rot_offset = [
            r - o for r, o in zip(self._rot_mask.shape, self.mask.shape)
        ]

    @property
    @np_array_wrapper
    def apex_of_femural_notch(self):
        if self._apex_of_femural_notch is None:
            num_pts = int(len(self.cont_pts[0]) * self._notch_percentage)

            sorted_y = np.sort(self.cont_pts[0])
            min_y = sorted_y[-num_pts]
            tmp = find_notch(min_y, self.lowest_mask_pt, self.discs)
            self._apex_of_femural_notch, self.discs = tmp

        return self._apex_of_femural_notch

    @property
    @np_array_wrapper
    def level(self):
        if self._level is None:
            self._level = calc_level_femur(
                self._rot_mask,
                self.apex_of_femural_notch,
                self.lowest_mask_pt,
                self._rot_angle,
                self._level_rot_thresh,
                self._level_rot_stepsize,
                self._rot_offset,
            )

        return self._level

    @property
    @np_array_wrapper
    def level_line(self):
        if self._level_line is None:
            start, end = self.level
            self._level_line = bres([start], end, -1).T

        return self._level_line

    @property
    @np_array_wrapper
    def shrinked_level(self):
        if self._shrinked_level is None:
            start, end = self.level
            _, new_end = shrink_points(self.mask, start, end)
            self._shrinked_knee_level = (start, new_end)

        return self._shrinked_knee_level

    @property
    @np_array_wrapper
    def shrinked_level_line(self):
        if self._shrinked_level_line is None:
            start, end = self.shrinked_level
            self._shrinked_level_line = bres([start], end, -1).T

        return self._shrinked_level_line

    @property
    @np_array_wrapper
    def circle_points(self):
        if self._circle_points is None:
            self._circle_points = determine_circ(
                self.cont_pts,
                self._circle_rot_angle,
                get_intersec(self.lsf_line, self.cont_pts)[1],
                self.is_left,
                [int((tmp - 1) / 2) for tmp in self.mask.shape],
                self._circle_pt_percentage,
            )

        return self._circle_points

    @property
    @np_array_wrapper
    def circle_params(self):
        if self._circle_params is None:
            self._circle_params = fit_circle(*self.circle_points)
        return self._circle_params

    @property
    @np_array_wrapper
    def mechanical_axis(self):
        if self._mechanical_axis is None:
            self._mechanical_axis = (self.circle_params[:2], self.apex_of_femural_notch)

        return self._mechanical_axis

    @property
    @np_array_wrapper
    def mechanical_axis_line(self):
        if self._mechanical_axis_line is None:
            start, end = self.mechanical_axis
            self._mechanical_axis_line = bres([start], end, -1)

        return self._mechanical_axis_line

    @property
    @np_array_wrapper
    def trochanter_major(self):
        if self._trochanter_major is None:
            self._trochanter_major = get_trochmaj(
                self.cont_pts, self.lsf, self.circle_params[:2]
            )
            return self._trochanter_major

    @property
    @np_array_wrapper
    def hip_level(self):
        return self.circle_params[:2], self.trochanter_major


class Tibia(BoneProxy):
    def __init__(
        self,
        mask,
        is_left,
        leave_out_percentage=0.15,
        y_level_femur=None,
        level_horiz_tol=0.25,
        level_vert_tol=0.025,
        level_percentage_side=0.15,
        level_percentage_mid=0.5,
        fill_holes=True,
    ):
        super().__init__(
            mask,
            is_left=is_left,
            lsf_leave_out_percentage=leave_out_percentage,
            fill_holes=fill_holes,
        )

        self._y_level_femur = y_level_femur

        self._mid_1 = None
        self._mid_2 = None
        self._level_1 = None
        self._level_line_1 = None
        self._level_2 = None
        self._level_line_2 = None

        self._horiz_tol = level_horiz_tol
        self._vert_tol = level_vert_tol
        self._level_percentage_side = level_percentage_side
        self._level_percentage_mid = level_percentage_mid

    @property
    @np_array_wrapper
    def level2(self):
        if self._ankle_level is None:
            self._ankle_level = (
                (self.ankle_level_line[0][0], self.ankle_level_line[1][0]),
                (self.ankle_level_line[0][-1], self.ankle_level_line[1][-1]),
            )

        return self._ankle_level

    @property
    @np_array_wrapper
    def level_line2(self):
        if self._level_line_2 is None:
            self._level_line_2 = calc_level_tibia2(
                self.mask, self.cont_pts, self.lsf_line, self._vert_tol, self._horiz_tol
            )

        return self._level_line_2

    @property
    @np_array_wrapper
    def level1(self):
        if self._level_1 is None:
            self._level_1 = (
                (self.level_line_1[0][0], self.level_line_1[1][0]),
                (self.level_line_1[0][-1], self.level_line_1[1][-1]),
            )

        return self._level_1

    @property
    @np_array_wrapper
    def level_line_1(self):
        if self._level_line_1 is None:
            self._level_line_1 = calc_level_tibia1(
                self.cont_pts, self._level_percentage_mid, self._level_percentage_side
            )

        return self._level_line_1

    @property
    @np_array_wrapper
    def mid_1(self):
        if self._mid_1 is None:
            self._mid_1 = determine_mid1(
                self.cont_pts, self.level_line_1, self._y_level_femur
            )

        return self._mid_1

    @property
    @np_array_wrapper
    def mid_2(self):
        if self._mid_2 is None:
            self._mid_2 = determine_mid_tibia2(self.cont_pts, self.lsf_line)

        return self._mid_2

    @property
    @np_array_wrapper
    def mechanical_axis(self):
        if self._mechanical_axis is None:
            self._mechanical_axis = (self.mid_1, self.mid_2)

        return self._mechanical_axis

    @property
    @np_array_wrapper
    def mechanical_axis_line(self):
        if self._mechanical_axis_line is None:
            start, end = self.mechanical_axis
            self._mechanical_axis_line = bres([start], end, -1)

        return self._mechanical_axis_line


class WholeLeg(object):
    def __init__(
        self,
        image,
        mask,
        is_left,
        kwargs_femur=None,
        kwargs_tibia=None,
        onehot_masks=True,
    ):

        if kwargs_femur is None:
            kwargs_femur = {}

        if kwargs_tibia is None:
            kwargs_tibia = {}

        self.image = image
        if onehot_masks:
            mask_femur = mask[1]
            mask_tibia = mask[2]
        else:
            mask_femur = mask == 1
            mask_tibia = mask == 2

        self.femur = Femur(mask_femur, is_left, **kwargs_femur)
        try:
            femur_level_line = self.femur.level_line
        except Exception as e:
            femur_level_line = None
        self.tibia = Tibia(
            mask_tibia, is_left, y_level_femur=femur_level_line, **kwargs_tibia
        )

    @property
    @np_array_wrapper
    def mechanical_axis(self):
        return self.femur.mechanical_axis[0], self.tibia.mechanical_axis[1]

    @property
    @np_array_wrapper
    def mechanical_axis_line(self):
        start, end = self.mechanical_axis
        return bres([start], end, -1)

    @staticmethod
    def _convert_separate_coords_to_pt_tuples(line):
        return list(zip(*line))

    @staticmethod
    def _convert_coord_tuples_to_flat_xy(line):
        new_line = []
        for pt in line:
            new_line += [pt[1], pt[0]]

        return new_line

    @property
    @angle_format_decorator
    def hka(self):
        return angle_between(
            self.femur.mechanical_axis[0] - self.femur.mechanical_axis[1],
            self.tibia.mechanical_axis[1] - self.tibia.mechanical_axis[0],
        )

    @property
    @angle_format_decorator
    def ama(self):
        return angle_between(
            self.femur.mechanical_axis[0] - self.femur.mechanical_axis[1],
            self.femur.anatomical_axis[0] - self.femur.anatomical_axis[1],
        )

    @staticmethod
    def line_intersection(line1, line2):
        ydiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        xdiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception("lines do not intersect")

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return np.array((y, x))
