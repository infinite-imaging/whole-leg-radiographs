import zipfile
import argparse
import tempfile
import os
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import rmtree, move
from tqdm import tqdm
from skimage.color import rgb2gray, rgba2rgb
from skimage.measure import label


def process_single_patient_dir(src_dir, dst_dir, resolution, left=True,
                               include_flipped=False):

    if not os.path.isdir(src_dir):
        return

    possible_files = [os.path.join(src_dir, x) for x in os.listdir(src_dir)
                      if (os.path.isfile(os.path.join(src_dir, x))
                          and x.endswith(".png"))]

    for file in possible_files:
        if "Hintergrund" not in file:
            continue

        imgs = {
            "original": imread(file),
            "masks": {}
        }

        if imgs["original"].shape[-1] == 4:
            imgs["original"] = rgba2rgb(imgs["original"],
                                        background=(0, 0, 0))

        for infix in ("Femur", "Tibia"):
            for mask_file in possible_files:
                if infix.lower() in mask_file.lower():
                    imgs["masks"][infix.lower()] = imread(mask_file)
                    break

        assert len(imgs["masks"]) >= 2

        patient_dst_dir = os.path.join(dst_dir, os.path.basename(src_dir))
        if left:
            patient_dst_dir = os.path.join(patient_dst_dir, "left")
        else:
            patient_dst_dir = os.path.join(patient_dst_dir, "right")

        os.makedirs(patient_dst_dir, exist_ok=True)
        if include_flipped:
            os.makedirs(patient_dst_dir + "_flipped", exist_ok=True)

        mid_pixel = int(imgs["original"].shape[1] / 2)

        if left:
            imgs["original"] = imgs["original"][:, mid_pixel:]
            for key, val in imgs["masks"].items():
                imgs["masks"][key] = val[:, mid_pixel:]
        else:
            imgs["original"] = imgs["original"][:, :mid_pixel]
            for key, val in imgs["masks"].items():
                imgs["masks"][key] = val[:, :mid_pixel]

        for key, val in imgs["masks"].items():
            imgs["masks"][key] = filter_largest_region(convert_mask_to_binary(
                val))

        if resolution is not None:
            imgs["original"] = resize(imgs["original"], resolution)
            for key, val in imgs["masks"].items():
                imgs["masks"][key] = resize(val, resolution)

        imsave(os.path.join(patient_dst_dir, "image.png"),
               imgs["original"])
        for key, val in imgs["masks"].items():
            imsave(os.path.join(patient_dst_dir, "mask_%s.png" % key), val)

        yield patient_dst_dir

        if include_flipped:
            imsave(os.path.join(patient_dst_dir + "_flipped", "image.png"),
                   np.fliplr(imgs["original"]))
            for key, val in imgs["masks"].items():
                imsave(os.path.join(patient_dst_dir + "_flipped", "mask_%s.png"
                                    % key), np.fliplr(val))

            yield patient_dst_dir + "_flipped"

        break


def process_single_patient(patient_id, patient_dict, src_dir, dst_dir,
                           resolution, include_flipped=False):

    combinations = []

    if patient_dict["left"]:
        combinations.append((os.path.join(src_dir, "right", patient_id),
                             True))
    if patient_dict["right"]:
        combinations.append((os.path.join(src_dir, "left", patient_id),
                             False))

    for _src_dir, left in combinations:
        yield from process_single_patient_dir(_src_dir, dst_dir, resolution,
                                              left, include_flipped)


def convert_mask_to_binary(img):

    img = rgba2rgb(img)
    gray = rgb2gray(img)
    binary_mask = gray != 1.

    return binary_mask


def filter_largest_region(mask):

    labels = label(mask)
    assert (labels.max() != 0)  # assume at least 1 region
    return (
        (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1) * 255
    ).astype(np.uint8)


def parse_root_dir(root_src_path):

    valid_subdirs = []

    for subdir in sorted(os.listdir(root_src_path)):
        subdir = os.path.join(root_src_path, subdir)

        if not os.path.isdir(subdir):
            continue

        if not any([os.path.isfile(os.path.join(subdir, _x))
                    and _x.endswith(".png") for _x in os.listdir(subdir)]):
            continue

        valid_subdirs.append(os.path.basename(subdir))
    return valid_subdirs


def introduce_year_scheme(root_dir):
    year_dirs = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, x))])

    print(year_dirs)

    tmp_dir = root_dir + "_copy"
    os.makedirs(os.path.join(tmp_dir, "left"))
    os.makedirs(os.path.join(tmp_dir, "right"))

    for year_dir in year_dirs:

        for prev_ext, after_ext in (("links", "left"), ("rechts", "right")):
            if os.path.isdir(os.path.join(year_dir, prev_ext)):
                sub_dirs = [os.path.join(year_dir, prev_ext, x)
                            for x in os.listdir(os.path.join(year_dir,
                                                             prev_ext))]

                for _dir in sub_dirs:
                    new_dir = str(os.path.dirname(_dir)
                                  ).replace(os.path.join(year_dir,
                                                         prev_ext),
                                            os.path.join(tmp_dir,
                                                         after_ext))

                    new_dir_name = str(os.path.basename(year_dir)).strip("PNG ") + "_" + str(os.path.basename(_dir))
                    new_dir = os.path.join(new_dir, new_dir_name)
                    move(_dir, new_dir)

    rmtree(root_dir)
    move(tmp_dir, root_dir)
    return root_dir


def process_zip_file(root_file, dst_path, include_flipped=False,
                     resolution=None, test_split=None, del_zip=False):

    tmp_dir = os.path.join(tempfile.gettempdir(), "WholeLegXRay")
    # os.makedirs(tmp_dir, exist_ok=True)

    # print("Extracting Zipfile")
    # with zipfile.ZipFile(root_file) as zfile:
    #     zfile.extractall(tmp_dir)

    if del_zip:
        print("Removing Zipfile")
        os.remove(root_file)

    if (len(os.listdir(tmp_dir)) == 1
            and os.path.isdir(os.path.join(tmp_dir,
                                           os.listdir(tmp_dir)[0]))):
        tmp_dir = os.path.join(tmp_dir,
                               os.listdir(tmp_dir)[0])

    # print("Introduce year scheme")
    # tmp_dir = introduce_year_scheme(tmp_dir)

    print("Parse for valid dirs")
    valid_dirs_left = parse_root_dir(os.path.join(tmp_dir, "left"))
    valid_dirs_right = parse_root_dir(os.path.join(tmp_dir, "right"))
    valid_dirs = {k: {"left": False, "right": False}
                  for k in valid_dirs_left + valid_dirs_right}

    for key in valid_dirs_left:
        valid_dirs[key]["left"] = True
    for key in valid_dirs_right:
        valid_dirs[key]["right"] = True

    all_patients = list(valid_dirs.keys())

    print("%03d dirs found" % len(valid_dirs))

    if test_split is None:
        test_patients = []
        train_patients = all_patients
    else:
        train_patients, test_patients = train_test_split(all_patients,
                                                         test_size=test_split / 100)

    train_set_files = []
    for patient in tqdm(train_patients):
        for result_file in process_single_patient(
                patient, valid_dirs[patient], tmp_dir,
                os.path.join(dst_path, "train"), resolution=resolution,
                include_flipped=include_flipped):
            train_set_files.append(result_file)

    test_set_files = []
    for patient in tqdm(test_patients):
        for result_file in process_single_patient(
                patient, valid_dirs[patient], tmp_dir,
                os.path.join(dst_path, "test"), resolution=resolution,
                include_flipped=False):
            test_set_files.append(result_file)

    rmtree(tmp_dir, ignore_errors=True)

    print("%03d Trainfiles and %03d Test Files" % (len(train_set_files),
                                                   len(test_set_files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", type=str,
                        help="Path to original Zip file")
    parser.add_argument("-d", "--dst_path", type=str,
                        help="Path to Preprocessing Output")
    parser.add_argument('-f', "--include_flipped", action='store_true',
                        help="Whether to include zipped files in the "
                             "trainset")
    parser.add_argument("-x", "--x_resolution", type=int, default=None,
                        help="The resolution of the x direction")
    parser.add_argument("-y", "--y_resolution", type=int, default=None,
                        help="The resolution of the y direction")
    parser.add_argument("-t", "--test_split", type=int,
                        help="The percentage of the patients to use for "
                             "testing (must be an integer)")
    parser.add_argument("-z", "--del_zip", action="store_true",
                        help="Whether to delete the zip file after "
                             "extracting it")

    args = parser.parse_args()

    if args.x_resolution is None or args.y_resolution is None:
        resolution = None
    else:
        resolution = (args.x_resolution, args.y_resolution)

    process_zip_file(root_file=args.src_path,
                     dst_path=args.dst_path,
                     include_flipped=args.include_flipped,
                     resolution=resolution,
                     test_split=args.test_split,
                     del_zip=args.del_zip)
