import gc
import json
import os
import traceback
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from batchgenerators.transforms.sample_normalization_transforms import range_normalization
from delira_unet import UNetTorch
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.transform import resize
from tqdm import tqdm

from whole_leg.io import draw_mask_overlay
from whole_leg.streamlined_mask_processor import WholeLeg

torch.multiprocessing.set_start_method('fork')
from torch.multiprocessing import Pool
import joblib


def segment_image(model, image, device,
                  thresh=0.5, min_size=300):
    image = image.squeeze().astype(np.float32)
    resized = resize(image, (1024, 256), preserve_range=True)
    resized = resized.squeeze()[None, None]
    resized = range_normalization(resized, (-1, 1))
    img_tensor = torch.from_numpy(resized).float()

    model = model.to(device)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0].detach().cpu().numpy()

    mask_bckg, mask_femur, mask_tibia = pred

    mask_femur = post_process_masks(mask_femur > thresh,
                                    min_size=min_size).astype(np.float32)
    mask_tibia = post_process_masks(mask_tibia > thresh,
                                    min_size=min_size).astype(np.float32)

    mask_femur = resize(mask_femur, image.shape, preserve_range=True) > thresh
    mask_tibia = resize(mask_tibia, image.shape, preserve_range=True) > thresh

    return mask_femur, mask_tibia


def post_process_masks(mask min_size=300):
    mask_filled = binary_fill_holes(mask)
    mask_removed_small = remove_small_objects(mask_filled, min_size)

    return mask_removed_small.astype(np.uint8)


def crop_img(img, crop=0.6):
    crop_width = int(crop * img.shape[-1])
    img_right = img[..., :crop_width]
    img_left = img[..., -crop_width:]

    return img_right, img_left, crop_width


def project_back_masks(mask_femur, mask_tibia, orig_shape, pos):
    mask_tibia_ = np.zeros(orig_shape, dtype=mask_tibia.dtype)
    mask_femur_ = np.zeros(orig_shape, dtype=mask_femur.dtype)

    if pos == 'begin':
        mask_femur_[..., :mask_femur.shape[-1]] = mask_femur
        mask_tibia_[..., :mask_tibia.shape[-1]] = mask_tibia
    elif pos == 'end':
        mask_femur_[..., -mask_femur.shape[-1]:] = mask_femur
        mask_tibia_[..., -mask_tibia.shape[-1]:] = mask_tibia
    else:
        raise ValueError

    return mask_femur_, mask_tibia_


def draw_one_side(whole_leg, img):
    return whole_leg.draw(img=img)


def process_patient(patient, model,
                    device, thresh=0.5,
                    draw=True, min_size=300,
                    ):
    if isinstance(patient, list):
        patient = patient[0]
    image = patient.pixel_array.squeeze()
    img_right, img_left, offset_left = crop_img(image)
    _mask_femur_left, _mask_tibia_left = segment_image(model, img_left,
                                                       device, thresh,
                                                       min_size)
    mask_femur_left, mask_tibia_left = project_back_masks(
        _mask_femur_left, _mask_tibia_left, image.shape, pos='end'
    )

    _mask_femur_right, _mask_tibia_right = segment_image(model, img_right,
                                                         device, thresh,
                                                         min_size)
    mask_femur_right, mask_tibia_right = project_back_masks(
        _mask_femur_right, _mask_tibia_right, image.shape, pos='begin'
    )

    whole_mask_left = np.array([np.zeros_like(mask_femur_left),
                                mask_femur_left,
                                mask_tibia_left])
    whole_mask_right = np.array([np.zeros_like(mask_femur_right),
                                 mask_femur_right,
                                 mask_tibia_right])

    whole_leg_left = WholeLeg(
        None,
        whole_mask_left, is_left=True, onehot_masks=True)

    whole_leg_right = WholeLeg(
        None,
        whole_mask_right, is_left=False, onehot_masks=True
    )

    angles = defaultdict(dict)

    for angle in ['HKA', 'AMA']:

        def get_angle(leg_proc: WholeLeg, angle_name):
            try:
                angle_val = getattr(leg_proc, angle_name.lower())

                if isinstance(angle_val, property):
                    angle_val = angle_val.__get__(leg_proc)

            except Exception as e:
                print('Extraction of %s failed' % angle_name)
                angle_val = None

            return angle_val

        angles['Left'][angle] = get_angle(whole_leg_left, angle.lower())
        angles['right'][angle] = get_angle(whole_leg_right, angle.lower())

    whole_mask = np.logical_or(whole_mask_right.astype(np.bool),
                               whole_mask_left.astype(np.bool)).astype(np.uint8)

    norm_img = (range_normalization(image[None, None].astype(np.float32),
                                    (0, 1)) * 255).astype(np.uint8)[0]

    norm_img = np.moveaxis(norm_img, 0, -1)
    if norm_img.shape[-1] == 1:
        norm_img = np.concatenate([norm_img, norm_img, norm_img], -1)
    pil_image = Image.fromarray(norm_img)
    if draw:
        pil_image = draw_one_side(whole_leg_left, pil_image)
        pil_image = draw_one_side(whole_leg_right, pil_image)
        segmented_img = draw_mask_overlay(norm_img, whole_mask * 255, alpha=0.5)
    else:
        segmented_img = pil_image

    return angles, segmented_img, pil_image


def get_recursive_files(root):
    dirs = [root]
    files = []
    while dirs:
        curr_dir = dirs.pop(0)
        items = [os.path.join(curr_dir, x) for x in os.listdir(curr_dir)]

        for item in items:
            if os.path.isdir(item):
                dirs.append(item)
            elif os.path.isfile(item):
                files.append(item)

    return files


def read_data(root, max_files=None):
    files = get_recursive_files(root)[:max_files]

    images = []
    for f in tqdm(files):
        _file = os.path.basename(f)
        try:
            if _file.startswith('IM_') and int(_file[3:]) >= 0:
                images.append(pydicom.read_file(f))
        except Exception as e:
            tqdm.write(e)

    return images


def create_accession_dict(data: list):
    return_dict = defaultdict(list)

    for sample in data:
        key = sample.AccessionNumber
        return_dict[key].append(sample)

    return dict(return_dict)


if __name__ == '__main__':
    data_root = '/Users/justusschock/Downloads/IMediaExport'
    model_path = '/Users/justusschock/Downloads/checkpoint_traced.pt'
    output_dir = '/Users/justusschock/Downloads/MedicadOutputs'

    os.makedirs(output_dir, exist_ok=True)

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    device = torch.device(device)

    print('Loading Model')
    model = torch.jit.load(model_path, map_location=device)

    print('Reading data')
    data = create_accession_dict(read_data(data_root))

    for key, val in tqdm(list(data.items())):
        angles, segmented_image, pil_image = process_patient(val, model, device)

        curr_out_path = os.path.join(output_dir, key)

        with open(curr_out_path + '.json', 'w') as f:
            json.dump(angles, f, indent=4, sort_keys=True)

        segmented_image.save(curr_out_path + '_segmented.png')
        pil_image.save('_annotated.png')
