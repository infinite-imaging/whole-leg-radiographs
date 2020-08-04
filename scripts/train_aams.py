from pathlib import Path
import menpo.io as mio
from menpofit.aam import HolisticAAM, PatchAAM
from menpo.feature import dsift, fast_dsift, hog, no_op, ndfeature
import joblib
import numpy as np
import os
from copy import deepcopy
import gc


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


def train_aam(
    imgs,
    save_path,
    n_scales=1,
    diagonal=150,
    features=dsift,
    img_size=(1024, 256),
    aam_cls=HolisticAAM,
):
    training_images = []
    # load landmarked images
    for i in deepcopy(imgs):
        if img_size is not None:
            i = i.resize(img_size)
        # append it to the list
        training_images.append(i)

    print("Training AAM")
    aam = aam_cls(
        training_images,
        group="PTS",
        verbose=True,
        holistic_features=features,
        scales=n_scales,
        diagonal=diagonal,
    )

    print("Saving AAM file")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(aam, save_path)


def parallel_wrapper(args: dict):
    id_str = args.pop("log_str")

    log_str = ""

    try:
        train_aam(**args)

    except Exception as e:
        log_str += "%s: %s" % (id_str, str(e))
        print(e)

    finally:
        print("finished %s" % id_str)
        gc.collect()
        return log_str


if __name__ == "__main__":
    import argparse
    import sys
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str)
    parser.add_argument("-s", "--savepath", type=str)
    parser.add_argument("-r", "--resolution", type=str, default="1024")
    parser.add_argument("-a", "--aamcls", type=str, default=None)
    parser.add_argument("-f", "--features", type=str, default=None)
    parser.add_argument("--diagonals", type=str, default="150")
    parser.add_argument("--scales", type=str, default="1")
    parser.add_argument("-j", "--jobs", type=int, default=None)

    if len(sys.argv) > 1:
        args = sys.argv
    else:
        args = None
    args = parser.parse_args(args)

    data_path = args.datapath
    save_path = args.savepath
    resolutions = args.resolution

    resolutions = [
        (int(_resolution), int(_resolution) // 4)
        for _resolution in resolutions.split(",")
    ]

    aam_cls = args.aamcls
    features = args.features
    diagonals = args.diagonals
    diagonals = [int(diagonal) for diagonal in diagonals.split(",")]

    scales = args.scales
    jobs = args.jobs

    aam_classes = []
    if aam_cls is None:
        aam_cls = ["all"]
    else:
        aam_cls = [tmp for tmp in aam_cls.split(",")]
    for _aam in aam_cls:
        for name, cls in [("holistic", HolisticAAM), ("patch", PatchAAM)]:
            if _aam in [name, "all"]:
                aam_classes.append(cls)

    if features is None:
        features = ["all"]
    else:
        features = [tmp for tmp in features.split(",")]

    feature_fns = []
    for _feature in features:
        for name, fn in [
            ("no_op", no_op),
            ("fast_dsift", float32_fast_dsift),
            ("dsift", float32_dsift),
            ("hog", float32_hog),
        ]:
            if _feature in [name, "all"]:
                feature_fns.append(fn)

    all_scales = []
    for _scales in scales.split(";"):
        all_scales.append([float(tmp) for tmp in _scales.split(",")])

    imgs = load_images(data_path)
    process_args = []
    for aam in aam_classes:
        for feature in feature_fns:
            for scale in all_scales:
                for resolution in resolutions:
                    for diagonal in diagonals:
                        _save_path = os.path.join(
                            save_path,
                            "%s_%s_scales_%s_resolution_%04d_diagonal_%04d.aam"
                            % (
                                aam.__name__,
                                feature.__name__,
                                "_".join([str(tmp * 100) for tmp in scale]),
                                resolution[0],
                                diagonal,
                            ),
                        )
                        process_args.append(
                            {
                                "imgs": deepcopy(imgs),
                                "save_path": _save_path,
                                "log_str": _save_path.replace(".aam", ""),
                                "n_scales": scale,
                                "diagonal": diagonal,
                                "features": feature,
                                "img_size": resolution,
                                "aam_cls": aam,
                            }
                        )

    if jobs is None or jobs > 0:
        with Pool(jobs) as p:
            log_strs = p.map(parallel_wrapper, process_args)
    else:
        log_strs = map(parallel_wrapper, process_args)

    print("\n\n".join(log_strs))
