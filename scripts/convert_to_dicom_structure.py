import numpy as np
from skimage.io import imread
import os
import SimpleITK as sitk
import time
from skimage.color import rgb2gray, rgba2rgb
from skimage.measure import label
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def load_patient(pat_path):
    files = [os.path.join(pat_path, x) for x in os.listdir(pat_path)
             if x.endswith(".png")]

    img = None
    masks = []

    for _file in files:
        if "hintergrund" in _file.lower():
            img = imread(_file)
            break

    if img is None:
        return None
    if len(img.shape) < 3:
        img = img[None]

    markierungen = []
    keys = ["femur", "tibia", "markierungen"]

    for _key in keys:
        for _file in files:
            if _key != "markierungen":
                if _key in _file.lower():
                    masks.append(imread(_file))
            else:
                if _key in _file.lower():
                    markierungen.append(_file)
    if not len(masks) == 2:
        return None

    markierungen = sorted(markierungen)
    for _file in markierungen:
        masks.append(imread(_file))

    for idx, mask in enumerate(masks):
        masks[idx] = filter_largest_region(convert_mask_to_binary(mask))

    whole_mask = np.array(masks)
    return img, whole_mask


def convert_mask_to_binary(img):
    
    if len(img.shape) >= 3:
        if img.shape[-1] > 3:
            img = rgba2rgb(img)
        
        if img.shape[-1] == 3:
            gray = rgb2gray(img)
        else:
            gray = img.squeeze()
    else:
        gray = img
            
    binary_mask = gray != 1.

    return binary_mask


def filter_largest_region(mask):
    "
    labels = label(mask)
    assert (labels.max() != 0)
    return (
            (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)*255
    ).astype(np.uint8)


def process_patient(args, all_paths, study_id, root_path, base_out_path):
    idx, pat = args
    pat_path = os.path.join(base_out_path, "%04d" % idx)

    study_path = os.path.join(pat_path, study_id)

    for series_idx, pot_series in enumerate(["right", "left"]):
        if pat in paths[pot_series]:
            series_path = os.path.join(study_path, "%02d" % (series_idx + 1))
            os.makedirs(series_path)
            rets = load_patient(os.path.join(root_path,
                                                        pot_series,
                                                        pat))
            if rets is None:
                continue
            img, whole_mask = rets
            mask = sitk.GetImageFromArray(whole_mask)
            mask.SetMetaData("0008|0012", time.strftime("%Y%m%d")) 
            mask.SetMetaData("0008|0013", time.strftime("%H%M%S"))  
            mask.SetMetaData("0010|0020", "%04d" % idx) 
            mask.SetMetaData("0020|0010", STUDY_ID) 
            mask.SetMetaData("0020|0011", "%02d" % (series_idx+1)) 
            mask.SetMetaData("0008|103E", "Segmentation %s" % str(pot_series).capitalize())
            mask.SetMetaData("0008|1030", "Whole Leg X-Ray")
            mask.SetMetaData("0008|0060", "SEG") 
            sitk.WriteImage(mask, os.path.join(series_path, "mask.dcm"))

            if not os.path.isdir(os.path.join(study_path, "%02d" % 0)):
                img = img.astype(np.float)
                img = img - img.min()
                img = img / img.max()
                img = img * (32768 + 32767) - 32768
                img = img.astype(np.int16)
                img = sitk.GetImageFromArray(img)
                img.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
                img.SetMetaData("0008|0013", time.strftime("%H%M%S")) 
                img.SetMetaData("0010|0020", "%04d" % idx)
                img.SetMetaData("0020|0010", STUDY_ID) 
                img.SetMetaData("0020|0011", "%02d" % 0)  
                img.SetMetaData("0008|103E", "Image") 
                img.SetMetaData("0008|0060", "RG") 
                img.SetMetaData("0008|1030", "Whole Leg X-Ray")  
                os.makedirs(os.path.join(study_path, "%02d" % 0))

                sitk.WriteImage(img, os.path.join(study_path, "%02d" % 0,
                                                    "image.dcm"))

    print("Finished Patient %03d" % idx)


if __name__ == '__main__':
    root_path = ""
    base_out_path = ""

    pats_left = sorted([x for x in os.listdir(os.path.join(root_path, "left"))])
    pats_right = sorted([x for x in os.listdir(os.path.join(root_path, "right"))])
    paths = {"left": pats_left, "right": pats_right}

    STUDY_ID = "%010d" % 1

    pats = list(set(pats_left + pats_right))

    func = partial(process_patient, all_paths=paths, study_id=STUDY_ID,
                   root_path=root_path, base_out_path=base_out_path)
    with Pool() as p:
        p.map(func, list(enumerate(pats)))
        
