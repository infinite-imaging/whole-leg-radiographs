import numpy as np
from PIL import Image, ImageDraw
from .least_squares import lsf


def draw_mask_overlay(
    image,
    mask,
    alpha=0.25,
    least_squares=False,
    fill=255,
    width=3,
    indices_to_process=(1, 2),
    leave_out_percentage=0.25,
):
    if isinstance(image, np.ndarray):
        if image.shape[-1] == 1:
            image = np.concatenate([image, image, image], -1).astype(np.uint8)
        image = Image.fromarray(image).convert("RGBA")
    mask_img = Image.fromarray(np.moveaxis(mask, 0, -1).astype(np.uint8)).convert(
        "RGBA"
    )
    new_img = Image.blend(image, mask_img, alpha=alpha)

    if least_squares:
        draw = ImageDraw.Draw(new_img)
        for idx in indices_to_process:
            _mask = mask[idx]
            line = lsf(_mask, leave_out_percentage=leave_out_percentage)
            draw.line(
                [line[1][0], line[0][0], line[1][-1], line[0][-1]],
                fill=fill,
                width=width,
            )

    return new_img
