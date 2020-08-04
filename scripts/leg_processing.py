# from whole_leg.femur import FemurMaskProcessor
# from whole_leg.tibia import TibiaMaskProcessor
from whole_leg.streamlined_mask_processor import WholeLeg
from whole_leg.dataset import WholeLegDataset
from whole_leg.bresenham_slope import bres
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import os
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image


def process_sample(args, save_path, fill_holes=True, error=False, **kwargs):
    idx, sample = args
    mask_femur = sample["label"] == 1
    mask_tibia = sample["label"] == 2
    mask_femur = mask_femur.squeeze().astype(np.uint8)
    mask_tibia = mask_tibia.squeeze().astype(np.uint8)

    if fill_holes:
        mask_femur = binary_fill_holes(mask_femur)
        mask_tibia = binary_fill_holes(mask_tibia)

    try:
        # processor_femur = FemurMaskProcessor(mask_femur,
        #                                      knee_version=knee_version)
        # processor_tibia = TibiaMaskProcessor(mask_tibia)

        img = (
            np.concatenate([sample["data"], sample["data"], sample["data"]]) * 255
        ).astype(np.uint8)

        img = Image.fromarray(np.moveaxis(img, 0, -1))

        # img = processor_femur.draw(img, **kwargs)
        # level_femur = processor_femur.calc_level(False, False)
        # level_femur = np.array(bresenhamline([knee_level_femur[0]],
        #

        leg = WholeLeg(img, sample["label"], bool(sample["is_left"]))

        img.save(os.path.join(save_path, "image_%03d.png" % idx))
        print("Successfully finished sample %03d" % idx)
    except Exception as e:
        if error:
            raise e
        else:
            print("Computation for sample %03d failed with: '%s'" % (idx, str(e)))


if __name__ == "__main__":
    data_path = "/work/local/Data/WholeLegSegs/train"
    save_path = "/work/local/Temp/WholeLegNew"
    parallel = True
    error = False

    # parallel = not parallel
    # error = not error

    os.makedirs(save_path, exist_ok=True)

    data = list(enumerate(WholeLegDataset(data_path)))

    func = partial(process_sample, save_path=save_path, error=error)
    os.makedirs(save_path, exist_ok=True)
    print("Start Estimation")
    if parallel:
        with Pool() as p:
            p.map(func, data)

    else:
        for args in tqdm(data):
            func(args)
