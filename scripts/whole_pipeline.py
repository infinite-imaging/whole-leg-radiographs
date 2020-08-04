import torch
from delira_unet import UNetTorch
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import regionprops, label
from skimage.transform import resize
from batchgenerators.augmentations.normalizations import range_normalization
from whole_leg.streamlined_mask_processor import WholeLeg
import numpy as np
from PIL import Image


def crop_img(img, segs=None, contains_left=True, contains_right=True, crop=0.6):

    result = {}
    offsets = {}
    seg_results = {}

    if contains_left and contains_right:

        num_pixels = int(img.shape[1] * img.shape)
        result["right"] = img[:, :num_pixels]
        result["left"] = img[:, -num_pixels:]

        if segs is not None:
            seg_results["left"] = segs[:, -num_pixels:]
            seg_results["right"] = segs[:, num_pixels:]

        offsets["left"] = num_pixels
        offsets["right"] = 0

    elif contains_left and not contains_right:
        result["left"] = img
        offsets["left"] = 0
        if segs is not None:
            seg_results["left"] = segs

    elif contains_right and not contains_left:
        result["right"] = img
        offsets["right"] = 0
        if segs is not None:
            seg_results["right"] = segs

    if segs is None:
        return result, offsets
    else:
        return result, offsets, seg_results


def prep_mask(mask, num_largest_areas=2, fill=True, img_size=None):

    if fill:
        mask = binary_fill_holes(mask)
    label_mask = label(mask)
    rprops = regionprops(label_mask)
    rprops_sorted = sorted(rprops, reverse=True, key=lambda x: x.area)[
        :num_largest_areas
    ]

    new_mask = np.zeros_like(mask)
    for prop in rprops_sorted:
        for pt in prop.coords:
            new_mask[pt[0], pt[1]] = 1

    if img_size is not None:
        new_mask = resize(new_mask, img_size, preserve_range=True)

    return new_mask


def process_patient(
    img,
    segs,
    contains_left=True,
    contains_right=True,
    crop=0.6,
    img_size=(1024, 256),
    device="cpu",
    thresh=0.5,
    fill_holes=True,
    num_largest_areas=2,
    **kwargs
):

    img = img.squeeze()

    crop_results = crop_img(
        img,
        segs=segs,
        contains_left=contains_left,
        contains_right=contains_right,
        crop=crop,
    )

    if segs is None:
        cropped_imgs, offsets = crop_results
    else:
        cropped_imgs, offsets, cropped_segs = crop_results

    lines, points, circles = [], [], []

    img = resize(img, img_size, preserve_range=True).astype(np.uint8)

    for key in cropped_imgs.keys():
        is_left = key == "left"

        _img = cropped_imgs[key]
        _offset = offsets[key]
        orig_shape = _img.shape

        mask = segs

        mask_femur = mask[1] > thresh
        mask_tibia = mask[2] > thresh

        resized_mask_femur = prep_mask(
            mask_femur,
            num_largest_areas=num_largest_areas,
            fill=fill_holes,
            img_size=img_size,
        )

        resized_mask_tibia = prep_mask(
            mask_tibia,
            num_largest_areas=num_largest_areas,
            fill=fill_holes,
            img_size=img_size,
        )

        processor = WholeLeg(
            img,
            mask=np.array(
                [
                    np.zeros_like(resized_mask_femur),
                    resized_mask_femur,
                    resized_mask_tibia,
                ]
            ),
            is_left=is_left,
        )

        _lines = [
            (processor.femur.mechanical_axis_line, False, "red"),
            (processor.tibia.mechanical_axis_line, False, "red"),
            (processor.femur.anatomical_axis_line, True, "green"),
            (processor.tibia.anatomical_axis_line, True, "green"),
            (processor.femur.shrinked_level_line, True, "blue"),
            (processor.tibia.level_line_1, True, "orange"),
            (processor.tibia.level_line2, True, "blue"),
            (processor.mechanical_axis_line, False, "purple"),
        ]
        _points = [
            (processor.femur.mechanical_axis, "red"),
            (processor.tibia.mechanical_axis, "red"),
            (
                processor._convert_separate_coords_to_pt_tuples(
                    processor.femur.circle_points
                ),
                "orange",
            ),
            (processor.femur.shrinked_knee_level, "blue"),
            (processor.tibia.knee_level, "orange"),
            (processor.tibia.ankle_level, "blue"),
        ]

        _circles = [
            (processor.femur.circle_params, "red"),
        ]

        for (line, convert, color) in _lines:
            if convert:
                line = line[0], line[1] + _offset
            else:
                line = np.array(line)
                line[:, 1] += _offset
            lines.append((line, convert, color))

        for pts, color in _points:
            pts = np.array(pts)
            pts[:, 1] += _offset

            points.append((pts, color))

        for (yc, xc, circ_rad), color in _circles:
            circles.append(((yc, xc + _offset, circ_rad), color))


if __name__ == "__main__":
    from delira_unet import UNetTorch
    import os
    from tqdm import tqdm
    from delira import set_debug_mode
    from whole_leg import (
        WholeLegDataset,
        draw_mask_overlay,
        combine_gt_pred_masks,
        HistogramEqualization,
        CopyTransform,
        AddGridTransform,
        SingleBoneDataset,
    )
    from delira.training import Predictor
    import numpy as np
    from batchgenerators.transforms import RangeTransform, Compose
    from delira.training.backends import convert_torch_to_numpy
    from functools import partial
    from delira.data_loading import DataManager, SequentialSampler

    data_path = ""
    save_path = ""
    model_checkpoint = ""
    split = "Test"

    data_path = os.path.join(data_path, split)
    save_path = os.path.join(save_path, split)
    os.makedirs(save_path, exist_ok=True)

    device = "cpu"
    crop = 0.6
    img_size = (1024, 256)
    thresh = 0.5
    fill_holes = True
    num_largest_areas = 2

    set_debug_mode(True)

    transforms = Compose(
        [
            CopyTransform("data", "data_orig"),
            CopyTransform("label", "seg"),
            # HistogramEqualization(),
            RangeTransform((-1, 1)),
            # AddGridTransform(),
        ]
    )

    print("Load Model")
    model = UNetTorch(
        3, 1, norm_layer="Instance", per_class=False, depth=6, start_filts=16
    )
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["model"])
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    print("Load Data")
    dset = WholeLegDataset(
        root_path=data_path, include_flipped=False, img_size=img_size
    )  # , bone_label=1)
    dmgr = DataManager(dset, 1, 4, transforms, sampler_cls=SequentialSampler)

    print("Start Predictions")

    predictor = Predictor(
        model,
        key_mapping={"x": "data"},
        convert_batch_to_npy_fn=convert_torch_to_numpy,
        prepare_batch_fn=partial(
            model.prepare_batch, input_device=device, output_device=device
        ),
    )

    with torch.no_grad():
        for idx, (preds_batch, _) in enumerate(
            predictor.predict_data_mgr(dmgr, verbose=True)
        ):
            # gt = preds_batch["label"]
            preds = preds_batch["pred"][0]
            img = (preds_batch["data_orig"][0] * 255).astype(np.uint8)

            # gt = np.concatenate([gt == 0, gt == 1, gt == 2])

            preds = preds > thresh
            img_size = preds_batch["img_size"][0]

            # switch between image side (lr) and anatomical side
            is_left = not preds_batch["is_left"]
            try:
                out_img = process_patient(
                    img,
                    segs=preds,
                    contains_left=is_left,
                    contains_right=not is_left,
                    crop=crop,
                    img_size=img_size,
                    device=device,
                    thresh=thresh,
                    fill_holes=fill_holes,
                    num_largest_areas=num_largest_areas,
                )

                out_img.save(os.path.join(save_path, "Patient_%03d_AutoSeg.png" % idx))
            except Exception as e:
                print("Autoseg Patient %03d failed with: %s" % (idx, str(e)))

            # try:
            #     out_img = process_patient(img,
            #                         segs=gt,
            #                         contains_left=is_left,
            #                         contains_right=not is_left,
            #                         crop=crop,
            #                         img_size=img_size,
            #                         device=device,
            #                         thresh=thresh,
            #                         fill_holes=fill_holes,
            #                         num_largest_areas=num_largest_areas)
            #
            #     out_img.save(os.path.join(save_path, "Patient_%03d_ManualSeg.png" % idx))
            # except Exception as e:
            #     print("Manualseg Patient %03d failed with: %s" % (idx, str(e)))
