from menpo import io as mio
import joblib
from menpofit.aam import LucasKanadeAAMFitter
from menpo.image import Image
from typing import List
import os
from menpo.feature import dsift, fast_dsift, hog, ndfeature
import numpy as np
from tqdm import tqdm


@ndfeature
def float32_hog(x):
    return hog(x).astype(np.float32)


@ndfeature
def float32_fast_dsift(x):
    return fast_dsift(x).astype(np.float32)


@ndfeature
def float32_dsift(x):
    return dsift(x).astype(np.float32)


def load_images(img_path):
    imgs = []
    print("Importing Images")
    for i in mio.import_images(str(img_path), verbose=True):
        # crop image
        i = i.crop_to_landmarks_proportion(0.1)
        # convert it to greyscale if needed
        if i.n_channels == 3:
            i = i.as_greyscale(mode="luminosity")

        imgs.append(i)

    return imgs


def eval_aams(aam_file, images: List[Image]):
    aam = joblib.load(aam_file)

    img_size = int(aam_file.split("resolution_")[1].split("_", 1)[0])
    img_size = (img_size, img_size // 4)

    fitter = LucasKanadeAAMFitter(aam)

    for i in tqdm(images):
        i = i.resize(img_size)
        fitting_result = fitter.fit_from_bb(
            i,
            i.landmarks[i.landmarks.group_labels[-1]].bounding_box(),
            max_iters=50,
            gt_shape=i.landmarks[i.landmarks.group_labels[-1]],
        )
        print("")


if __name__ == "__main__":
    from copy import deepcopy

    aam_dir = ""
    data_dir = ""

    aam_files = sorted(
        [os.path.join(aam_dir, x) for x in os.listdir(aam_dir) if x.endswith(".aam")]
    )
    images = load_images(data_dir)
    for aam_file in aam_files[36:]:
        eval_aams(aam_file, deepcopy(images))
