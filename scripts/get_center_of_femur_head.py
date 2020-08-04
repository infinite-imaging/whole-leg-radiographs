from skimage.io import imread
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
from math import sqrt
from PIL import Image, ImageDraw

from whole_leg.femur.head import PointDeterminer

if __name__ == "__main__":

    print("Read Image")
    mask = imread("")
    pt_radius = 5

    mask = binary_fill_holes(mask)
    mask = mask.astype(np.uint8)

    print("Start Fitting")
    pt_finder = PointDeterminer(mask)

    (my, mx, r), pts = pt_finder()

    img = np.concatenate([mask[None], mask[None], mask[None]]) * 255
    img = np.moveaxis(img, 0, -1)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.ellipse((mx - r, my - r, mx + r, my + r), outline="blue", width=10)
    draw.ellipse(
        (mx - pt_radius, my - pt_radius, mx + pt_radius, my + pt_radius), fill="red"
    )
    for pt, color in zip(pts, ["orange", "green", "yellow"]):
        draw.ellipse(
            (
                pt[1] - pt_radius,
                pt[0] - pt_radius,
                pt[1] + pt_radius,
                pt[0] + pt_radius,
            ),
            fill=color,
        )

    img.save("/work/local/tmp.png")
