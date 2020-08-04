import numpy as np
import sys
import os

from .level import calc_level_v1, calc_level_v2, find_notch
from ..least_squares import lsf
from .head import PointDeterminer
from collections import defaultdict

from .level import angle_between
from ..drawer import Drawer


class FemurMaskProcessor:
    def __init__(self, femur_mask, knee_version=2):
        self.mask = femur_mask
        self._knee_version = knee_version

    def calc_level(self, rot=False, return_notch=False, **kwargs):
        if self._knee_version == 2:
            estim_fn = calc_level_v2
        elif self._knee_version == 1:
            estim_fn = calc_level_v1
        else:
            raise ValueError

        return estim_fn(mask=self.mask, rot=rot, return_notch=return_notch, **kwargs)

    def calc_anatomical_axis(self, leave_out_percentage=(0.25, 0.15)):
        lsf = lsf(mask=self.mask, leave_out_percentage=leave_out_percentage)

        # return only start and end points
        return (lsf[0][0], lsf[1][0]), (lsf[0][-1], lsf[1][-1])

    def calc_mechanical_axis(
        self,
        notch_pt_percentage=0.1,
        deg_femur_head=25,
        percentage_head=0.05,
        return_circ_params=False,
    ):

        notch = find_notch(self.mask, percentage=notch_pt_percentage)

        head_mid_finder = PointDeterminer(
            self.mask, degree=deg_femur_head, percentage=percentage_head
        )

        (my, mx, r), circ_pts = head_mid_finder(return_other_pts=True)

        if return_circ_params:
            return ((my, mx), notch), (r, circ_pts)
        else:
            return (my, mx), notch

    def calc_all(
        self,
        additional_outputs=False,
        rot=False,
        percentage_anat_axis=(0.25, 0.15),
        percentage_notch=0.1,
        percentage_head=0.05,
        deg_femur_head=10,
        **kwargs
    ):

        return_vals = defaultdict(dict)
        keys = ["start", "end"]
        if additional_outputs:
            keys.append("notch")

        for key, val in zip(
            keys, self.calc_level(rot=rot, return_notch=additional_outputs, **kwargs),
        ):
            return_vals["knee_level"][key] = val

        start_anat_axis, end_anat_axis = self.calc_anatomical_axis(
            leave_out_percentage=percentage_anat_axis
        )
        return_vals["anatomical_axis"] = {
            "start": start_anat_axis,
            "end": end_anat_axis,
        }

        mechanical_axis = self.calc_mechanical_axis(
            notch_pt_percentage=percentage_notch,
            return_circ_params=additional_outputs,
            deg_femur_head=deg_femur_head,
            percentage_head=percentage_head,
        )

        if additional_outputs:
            return_vals["mechanical_axis"]["start"] = mechanical_axis[0][0]
            return_vals["mechanical_axis"]["end"] = mechanical_axis[0][1]
            return_vals["mechanical_axis"]["radius"] = mechanical_axis[1][0]
            return_vals["mechanical_axis"]["circ_pts"] = mechanical_axis[1][1]

        else:
            return_vals["mechanical_axis"]["start"] = mechanical_axis[0]
            return_vals["mechanical_axis"]["end"] = mechanical_axis[1]

        mech_vec = np.array(return_vals["mechanical_axis"]["start"]) - np.array(
            return_vals["mechanical_axis"]["end"]
        )
        anat_vec = np.array(return_vals["anatomical_axis"]["start"]) - np.array(
            return_vals["anatomical_axis"]["end"]
        )
        angle = np.abs(np.rad2deg(angle_between(mech_vec, anat_vec)))

        if angle > 180:
            angle = 360 - angle

        return_vals["angle"] = angle

        return return_vals

    def draw(
        self,
        img,
        additional_outputs=True,
        rot_img=False,
        percentage_anat_axis=(0.15, 0.15),
        percentage_notch=0.1,
        percentage_head=0.05,
        deg_femur_head=10,
        **kwargs
    ):
        outputs = self.calc_all(
            additional_outputs=additional_outputs,
            rot_img=rot_img,
            percentage_anat_axis=percentage_anat_axis,
            percentage_notch=percentage_notch,
            percentage_head=percentage_head,
            deg_femur_head=deg_femur_head,
        )

        drawer = Drawer(img, **kwargs)

        draw_lines = []
        draw_pts = []

        for key, color in zip(
            ["knee_level", "mechanical_axis", "anatomical_axis"],
            ["blue", "red", "green"],
        ):
            draw_lines.append(((outputs[key]["start"], outputs[key]["end"]), color))

        if additional_outputs:
            for pt in outputs["mechanical_axis"]["circ_pts"]:
                draw_pts.append((pt, "orange"))

        for pt, color in draw_pts:
            drawer.draw_pt(pt[1], pt[0], color=color)

        for line, color in draw_lines:
            drawer.draw_line(line[0], line[1], color=color)

        if additional_outputs:
            radius = outputs["mechanical_axis"]["radius"]
            center = outputs["mechanical_axis"]["start"]

            drawer.draw_ellipse(
                (
                    center[1] - radius,
                    center[0] - radius,
                    center[1] + radius,
                    center[0] + radius,
                ),
                outline="red",
            )

        drawer.draw_text((50, 50), "Angle: %.3f °" % outputs["angle"], fill="red")

        return drawer.img
