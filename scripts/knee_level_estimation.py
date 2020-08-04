import os
from whole_leg import WholeLegDataset
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from PIL import ImageDraw, Image
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from whole_leg.femur.level import calc_level_v1, calc_level_v2, bres


def draw_knee_level(image, start_pt, end_pt, notch=None, radius=25, line_width=10):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    line_pts = []
    for pt in bresenhamline([start_pt], end_pt, max_iter=-1):
        line_pts += [pt[1], pt[0]]
    draw.line(line_pts, fill="orange", width=line_width)
    draw.ellipse(
        (
            start_pt[1] - radius,
            start_pt[0] - radius,
            start_pt[1] + radius,
            start_pt[0] + radius,
        ),
        fill="red",
    )
    draw.ellipse(
        (
            end_pt[1] - radius,
            end_pt[0] - radius,
            end_pt[1] + radius,
            end_pt[0] + radius,
        ),
        fill="blue",
    )

    if notch is not None:
        draw.ellipse(
            (
                notch[1] - radius,
                notch[0] - radius,
                notch[1] + radius,
                notch[0] + radius,
            ),
            fill="green",
        )

    return image


def process_sample(args, save_path, with_rot):
    idx, sample = args
    image = (
        np.moveaxis(
            np.concatenate([sample["data"], sample["data"], sample["data"]]), 0, -1
        )
        * 255
    )
    failure_steps = []

    try:
        mask = (sample["label"] == 1).astype(np.uint8)
        filled_mask = binary_fill_holes(mask)
    except Exception as e:
        failure_steps += [("preprocessing", str(e))]

    try:
        knee_level_v1_no_rot = calc_knee_level_v1(
            filled_mask.squeeze().astype(np.uint8), rot_img=False
        )
        draw_knee_level(image, *knee_level_v1_no_rot).save(
            os.path.join(save_path, "v1_without_rotation", "image_%03d.png" % idx)
        )
    except Exception as e:
        failure_steps += [("v1_no_rot", str(e))]

    if with_rot:
        try:
            knee_level_v1_rot = calc_knee_level_v1(
                filled_mask.squeeze().astype(np.uint8), rot_img=True
            )
            draw_knee_level(image, *knee_level_v1_rot).save(
                os.path.join(save_path, "v1_with_rotation", "image_%03d.png" % idx)
            )
        except Exception as e:
            failure_steps += [("v1_rot", str(e))]
    try:
        knee_level_v2_no_rot = calc_knee_level_v2(
            filled_mask.squeeze().astype(np.uint8),
            rot_img=False,
            shrink_to_mask_size=True,
        )
        draw_knee_level(image, *knee_level_v2_no_rot).save(
            os.path.join(save_path, "v2_without_rotation", "image_%03d.png" % idx)
        )
    except Exception as e:
        failure_steps += [("v2_no_rot", str(e))]

    if with_rot:
        try:
            knee_level_v2_rot = calc_knee_level_v2(
                filled_mask.squeeze().astype(np.uint8),
                rot_img=True,
                shrink_to_mask_size=True,
            )
            draw_knee_level(image, *knee_level_v2_rot).save(
                os.path.join(save_path, "v2_with_rotation", "image_%03d.png" % idx)
            )
        except Exception as e:
            failure_steps += [("v2_rot", str(e))]

    if not failure_steps:
        print("Finished Sample %03d" % idx)
    else:
        for failure_step, e in failure_steps:
            print(
                "Sample %03d Failed during step %s with: %s"
                % (idx, failure_step, str(e))
            )


if __name__ == "__main__":
    parallel = True
    with_rot = False
    save_path = "/work/local/Temp/knee_level_new"
    data_path = "/work/scratch/schock/Data/WholeLegSegs/train"

    os.makedirs(os.path.join(save_path, "v1_without_rotation"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "v2_without_rotation"), exist_ok=True)
    if with_rot:
        os.makedirs(os.path.join(save_path, "v1_with_rotation"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "v2_with_rotation"), exist_ok=True)

    print("Loading Data")
    data = list(enumerate(WholeLegDataset(data_path, False, False, None)))

    estim_func = partial(process_sample, save_path=save_path, with_rot=with_rot)
    if parallel:
        print("Starting Parallel Estimation")
        with Pool() as p:
            p.map(estim_func, data)
    else:
        for args in tqdm(data):
            estim_func(args)

