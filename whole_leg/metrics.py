import numpy as np
import logging
import torch
from torch import Tensor

from .utils import make_onehot_npy, make_onehot_torch
from scipy.spatial.distance import dice


logger = logging.getLogger(__file__)


def compute_stat_score(pred, label, n):

    cls_pred = np.argmax(pred, axis=1)

    tp = ((cls_pred == n) * (label == n)).sum()
    fp = ((cls_pred == n) * (label != n)).sum()
    tn = ((cls_pred != n) * (label != n)).sum()
    fn = ((cls_pred != n) * (label == n)).sum()

    return tp, fp, tn, fn


def dice_score_including_background(
    pred, label, bg=False, cls_logging=False, nan_score=1.0, no_fg_score=1.0
):

    if not np.count_nonzero(pred) > 0:
        logger.warning("Prediction only contains zeros. Dice score might be ambigious.")

    bg = 1 - int(bool(bg))

    n_classes = pred.shape[1]
    score = 0
    for i in range(bg, n_classes):
        tp, fp, tn, fn = compute_stat_score(pred, label, i)

        if not np.any(label == i):
            score_cls = no_fg_score
        elif np.isclose((2 * tp + fp + fn), 0):
            score_cls = nan_score
        else:
            score_cls = (2 * tp) / (2 * tp + fp + fn)

        if cls_logging:
            logger.info({"value": {"value": score_cls, "name": "dice_cls_" + str(i)}})
            pass

        score += score_cls
    return score / (n_classes - bg)
