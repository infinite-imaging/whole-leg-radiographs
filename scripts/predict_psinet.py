from whole_leg import (
    WholeLegDataset,
    draw_mask_overlay,
    combine_gt_pred_masks,
    HistogramEqualization,
    CopyTransform,
    AddGridTransform,
    SingleBoneDataset,
    PsiNet,
    MultiObjectiveDataset,
)
from delira.training import Predictor
import numpy as np
import torch
import os
from tqdm import tqdm
from batchgenerators.transforms import RangeTransform, Compose
from delira.training.backends import convert_torch_to_numpy
from functools import partial
from delira.data_loading import DataManager, SequentialSampler
from delira_unet import UNetTorch
from delira import set_debug_mode

if __name__ == "__main__":
    checkpoint_path = ""
    data_path = ""
    save_path = ""

    set_debug_mode(True)

    transforms = Compose(
        [
            CopyTransform("data", "data_orig"),
            # HistogramEqualization(),
            RangeTransform((-1, 1)),
            # AddGridTransform(),
        ]
    )

    img_size = (1024, 256)
    thresh = 0.5

    print("Load Model")
    model = torch.jit.load(checkpoint_path)
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    print("Load Data")
    dset = MultiObjectiveDataset(
        root_path=data_path, include_flipped=True, img_size=img_size, contourwidth=5
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

    os.makedirs(os.path.join(save_path, "per_class_all_masks"), exist_ok=True)
    # os.makedirs(os.path.join(save_path, "per_mask_all_classes"), exist_ok=True)

    with torch.no_grad():
        for idx, (preds_batch, _) in enumerate(
            predictor.predict_data_mgr(dmgr, verbose=True)
        ):
            gt = preds_batch["label_seg"]
            preds = preds_batch["pred_seg"][0]
            img = preds_batch["data_orig"][0]
            preds = np.argmax(preds, axis=0)

            img = np.concatenate([img, img, img]) * 255
            gt = np.concatenate([gt == 0, gt == 1, gt == 2])
            preds = np.arra([preds == 0, preds == 1, preds == 2])

            preds = (preds > thresh).astype(np.uint8) * 255
            gt = (gt > thresh).astype(np.uint8) * 255
            # draw_mask_overlay(image=img,
            #                   mask=gt,
            #                   least_squares=True
            #                   ).save(
            #     os.path.join(save_path, "per_mask_all_classes",
            #                  "image_%03d_gt.png" % idx))
            # draw_mask_overlay(image=img,
            #                   mask=preds,
            #                   least_squares=True
            #                   ).save(
            #     os.path.join(save_path, "per_mask_all_classes",
            #                  "image_%03d_pred.png" % idx))

            combined_masks = (
                combine_gt_pred_masks(mask_pred=preds, mask_gt=gt) > thresh
            ).astype(np.uint8) * 255

            draw_mask_overlay(image=img, mask=combined_masks[1]).save(
                os.path.join(
                    save_path, "per_class_all_masks", "image_%03d_femur.png" % idx
                )
            )

            draw_mask_overlay(image=img, mask=combined_masks[2]).save(
                os.path.join(
                    save_path, "per_class_all_masks", "image_%03d_tibia.png" % idx
                )
            )

