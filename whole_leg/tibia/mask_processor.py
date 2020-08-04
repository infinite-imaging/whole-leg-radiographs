import numpy as np

from .ankle import calc_ankle_level as _calc_ankle_level, determine_mid_of_ankle
from .knee import get_knee_level, determine_mid_of_knee
from ..utils import angle_between
from ..least_squares import lsf
from collections import defaultdict
from ..mask_utils import shrink_points, get_lowest_mask_pt
from ..drawer import Drawer
from PIL import Image, ImageDraw, ImageFont


class TibiaMaskProcessor:
    def __init__(self, mask):
        self.mask = mask

    def calc_knee_level(
        self, percentage_mid=0.3, percentage_side=0.2, shrink_to_mask=False
    ):
        knee_level = get_knee_level(
            self.mask, percentage_mid=percentage_mid, percentage_side=percentage_side
        )
        start = knee_level[0][0], knee_level[1][0]
        end = knee_level[0][-1], knee_level[1][-1]

        if shrink_to_mask:
            start, end = shrink_points(self.mask, start, end)

        return start, end

    def calc_ankle_level(self, vert_tol=0.025, shrink_to_mask=False):
        level = _calc_ankle_level(self.mask, vert_tol=vert_tol)
        start = (level[0][0], level[1][0])
        end = (level[0][-1], level[1][-1])

        if shrink_to_mask:
            start, end = shrink_points(self.mask, start, end)

        return start, end

    def calc_mechanical_axis(
        self,
        knee_level_femur,
        percentage_mid=0.3,
        percentage_side=0.2,
        leave_out_percentage=0.15,
    ):
        mech_axis_top = determine_mid_of_knee(
            self.mask,
            femur_level=knee_level_femur,
            percentage_mid=percentage_mid,
            percentage_side=percentage_side,
        )
        mech_axis_bottom = determine_mid_of_ankle(
            self.mask, leave_out_percentage=leave_out_percentage
        )

        return mech_axis_top, mech_axis_bottom

    def calc_anatomical_axis(self, leave_out_percentage=0.3):
        lsf = lsf(self.mask, leave_out_percentage=leave_out_percentage)

        return (lsf[0][0], lsf[1][0]), (lsf[0][-1], lsf[1][-1])

    def calc_all(
        self,
        knee_level_femur=None,
        vert_tol=0.025,
        leave_out_percentage=0.3,
        percentage_mid=0.3,
        percentage_side=0.2,
        shrink_to_mask=False,
    ):

        output_vals = defaultdict(dict)

        ankle_level = self.calc_ankle_level(
            vert_tol=vert_tol, shrink_to_mask=shrink_to_mask
        )

        output_vals["ankle_level"] = {"start": ankle_level[0], "end": ankle_level[1]}

        knee_level = self.calc_knee_level(
            percentage_mid=percentage_mid,
            percentage_side=percentage_side,
            shrink_to_mask=shrink_to_mask,
        )
        output_vals["knee_level"] = {"start": knee_level[0], "end": knee_level[1]}

        anat_axis = self.calc_anatomical_axis(leave_out_percentage=leave_out_percentage)
        output_vals["anatomical_axis"] = {"start": anat_axis[0], "end": anat_axis[1]}

        if knee_level_femur is not None:
            mech_axis = self.calc_mechanical_axis(
                knee_level_femur=knee_level_femur,
                percentage_mid=percentage_mid,
                percentage_side=percentage_side,
                leave_out_percentage=leave_out_percentage,
            )

            output_vals["mechanical_axis"] = {
                "start": mech_axis[0],
                "end": mech_axis[1],
            }

            mech_vec = np.array(output_vals["mechanical_axis"]["start"]) - np.array(
                output_vals["mechanical_axis"]["end"]
            )
            anat_vec = np.array(output_vals["anatomical_axis"]["start"]) - np.array(
                output_vals["anatomical_axis"]["end"]
            )
            angle = np.abs(np.rad2deg(angle_between(mech_vec, anat_vec)))

            if angle > 180:
                angle = 360 - angle

            output_vals["angle"] = angle

        return output_vals

    def __call__(
        self,
        knee_level_femur=None,
        vert_tol=0.025,
        leave_out_percentage=0.3,
        percentage_mid=0.3,
        percentage_side=0.2,
        shrink_to_mask=False,
    ):
        return self.calc_all(
            knee_level_femur=knee_level_femur,
            vert_tol=vert_tol,
            leave_out_percentage=leave_out_percentage,
            percentage_mid=percentage_mid,
            percentage_side=percentage_side,
            shrink_to_mask=shrink_to_mask,
        )

    def draw(
        self,
        img: Image,
        knee_level_femur=None,
        vert_tol=0.025,
        leave_out_percentage=0.3,
        percentage_mid=0.3,
        percentage_side=0.2,
        shrink_to_mask=False,
        **kwargs
    ):

        drawer = Drawer(img, **kwargs)
        outputs = self.calc_all(
            knee_level_femur=knee_level_femur,
            vert_tol=vert_tol,
            leave_out_percentage=leave_out_percentage,
            percentage_mid=percentage_mid,
            percentage_side=percentage_side,
            shrink_to_mask=shrink_to_mask,
        )

        drawer.draw_line(
            outputs["ankle_level"]["start"], outputs["ankle_level"]["end"], color="blue"
        )
        drawer.draw_line(
            outputs["knee_level"]["start"], outputs["knee_level"]["end"], color="orange"
        )
        drawer.draw_line(
            outputs["anatomical_axis"]["start"],
            outputs["anatomical_axis"]["end"],
            color="green",
        )
        if "mechanical_axis" in outputs:
            drawer.draw_line(
                outputs["mechanical_axis"]["start"],
                outputs["mechanical_axis"]["end"],
                color="blue",
            )

        if "angle" in outputs:
            drawer.draw_text(
                (50, get_lowest_mask_pt(self.mask)[0] + 100),
                "Angle: %.3f °" % outputs["angle"],
                fill="red",
            )

        return drawer.img
