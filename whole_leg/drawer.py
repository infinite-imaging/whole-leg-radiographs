from PIL import Image, ImageFont, ImageDraw
from .bresenham_slope import bres
import numpy as np
import os
import warnings
from typing import Iterable, Dict, Any, Union
from collections import Mapping


class Drawer:
    def __init__(
        self,
        img,
        pt_radius=25,
        line_width=10,
        text_size=150,
        font="fonts/Roboto-Black.ttf",
    ):

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        self.img = img
        self.draw = ImageDraw.Draw(img)
        self.pt_radius = pt_radius
        self.line_width = line_width
        self.text_size = text_size

        try:
            font = ImageFont.truetype(font, text_size)
        except Exception as e:
            font = ImageFont.load_default()
            warnings.warn(
                "Could not load specified font due to error "
                "%s. Switching to Default font" % str(e),
                UserWarning,
            )

        self.font = font

    def draw_pt(self, x, y, color=None, outline=None, pt_radius=None, width=0):
        if pt_radius is None:
            pt_radius = self.pt_radius

        if width is None:
            width = self.line_width
        self.draw.ellipse(
            (x - pt_radius, y - pt_radius, x + pt_radius, y + pt_radius),
            fill=color,
            outline=outline,
            width=width,
        )

        return self.img

    def draw_line(
        self,
        start_pt,
        end_pt,
        color=None,
        pt_outline=None,
        pt_radius=None,
        width=None,
        joint=None,
    ):
        if width is None:
            width = self.line_width

        self.draw_pt(
            start_pt[1],
            start_pt[0],
            color=color,
            pt_radius=pt_radius,
            outline=pt_outline,
            width=0,
        )
        self.draw_pt(
            end_pt[1],
            end_pt[0],
            color=color,
            pt_radius=pt_radius,
            outline=pt_outline,
            width=0,
        )

        line = bres([start_pt], end_pt, -1)

        line_pts = []
        for pt in line:
            line_pts += [pt[1], pt[0]]

        line_pts = []
        for pt in line:
            line_pts += [pt[1], pt[0]]

        self.draw.line(line_pts, fill=color, width=width, joint=joint)

        return self.img

    def draw_ellipse(self, xy, fill=None, outline=None, width=None):

        if width is None:
            width = self.line_width
        self.draw.ellipse(xy=xy, fill=fill, outline=outline, width=width)

        return self.img

    def draw_text(self, xy, text, fill=None, font=None, anchor=None, **kwargs):
        if font is None:
            font = self.font

        self.draw.text(xy=xy, text=text, fill=fill, font=font, anchor=anchor, **kwargs)

        return self.img

    @staticmethod
    def call_fn(function, args):
        if isinstance(args, Mapping):
            return function(**args)
        else:
            return function(*args)

    def __call__(self, points=(), lines=(), ellipses=(), texts=()):

        for pt_args in points:
            self.call_fn(self.draw_pt, pt_args)

        for line_args in lines:
            self.call_fn(self.draw_line, line_args)

        for ellipse_args in ellipses:
            self.call_fn(self.draw_ellipse, ellipse_args)

        for text_args in texts:
            self.call_fn(self.draw_text, text_args)

        return self.img
