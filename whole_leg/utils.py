import torch
import numpy as np
from scipy.ndimage.morphology import binary_erosion


def make_onehot_npy(labels, n_classes):

    labels = labels.reshape(-1).astype(np.uint8)
    return np.eye(n_classes)[labels]


def make_onehot_torch(labels, n_classes):

    idx = labels.to(dtype=torch.long)

    new_shape = list(labels.unsqueeze(dim=1).shape)
    new_shape[1] = n_classes
    labels_onehot = torch.zeros(*new_shape, device=labels.device, dtype=labels.dtype)
    labels_onehot.scatter_(1, idx.unsqueeze(dim=1), 1)
    return labels_onehot


def combine_gt_pred_masks(mask_pred, mask_gt=None):

    if mask_gt is None:
        return mask_pred

    else:
        masks_total = []
        for idx in range(mask_pred.shape[0]):
            mask_same = np.logical_and(mask_pred[idx], mask_gt[idx])
            mask_gt_only = np.logical_and(mask_gt[idx], np.logical_not(mask_pred[idx]))
            mask_pred_only = np.logical_and(
                mask_pred[idx], np.logical_not(mask_gt[idx])
            )

            masks_total.append(np.array([mask_gt_only, mask_same, mask_pred_only]))
        return np.array(masks_total)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
