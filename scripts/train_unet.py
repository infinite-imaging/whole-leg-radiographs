import logging
import sys

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger("Execute_Logger")

import os

from sklearn.model_selection import train_test_split

from delira.training import PyTorchExperiment
from delira.utils import DeliraConfig
from delira import get_current_debug_mode, set_debug_mode
from delira.data_loading import DataManager
from delira.data_loading.sampler import SequentialSampler, RandomSampler
from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch
from delira_unet import (
    UNetTorch,
    RAdam,
    dice_score_including_background,
    SoftDiceLossPyTorch,
)
from whole_leg import (
    WholeLegDataset,
    HistogramEqualization,
    AddGridTransform,
    SingleBoneDataset,
    DelayedCosineAnnealingLRCallback,
    DistanceLoss,
    FocalLossWithLogits,
    ContourTransform,
)
from batchgenerators.transforms import (
    RangeTransform,
    Compose,
    ZeroMeanUnitVarianceTransform,
)
from torchtools.optim import RangerLars
import json
import torch

set_debug_mode(False)

data_path = ""
config_path = ""
save_path = "/tmp/ContourUnet"

bone_label = None

data_path = os.path.expanduser(data_path)
save_path = os.path.expanduser(save_path)

num_epochs = 100

with open(config_path, "r") as f:
    config = DeliraConfig(**json.load(f))

base_transforms = [
    # HistogramEqualization(),
    RangeTransform((-1, 1)),
    # AddGridTransform()
]
train_specific_transforms = []
test_specific_transforms = []

train_transforms = Compose(base_transforms + train_specific_transforms)
test_transforms = Compose(base_transforms + test_specific_transforms)

if get_current_debug_mode():
    train_dir = "Test"
    test_dir = "Test"
else:
    train_dir = "Train"
    test_dir = "Test"

train_path = os.path.join(data_path, train_dir)
test_path = os.path.join(data_path, test_dir)

if bone_label is None:
    dset = WholeLegDataset(
        train_path, include_flipped=True, img_size=config.img_size, contourwidth=5
    )
    dset_test = WholeLegDataset(
        test_path, include_flipped=True, img_size=config.img_size, contourwidth=5
    )
else:
    dset = SingleBoneDataset(
        train_path,
        include_flipped=True,
        img_size=config.img_size,
        bone_label=bone_label,
    )
    dset_test = SingleBoneDataset(
        test_path, include_flipped=True, img_size=config.img_size, bone_label=bone_label
    )

# idx_train, idx_val = train_test_split(
#     list(range(len(dset))), test_size=config.val_split,
#     random_state=config.seed)

# dset_train = dset.get_subset(idx_train)
# dset_val = dset.get_subset(idx_val)

dset_train = dset
dset_val = dset_test
mgr_train = DataManager(
    dset_train, config.batchsize, 4, train_transforms, sampler_cls=RandomSampler
)
mgr_val = DataManager(
    dset_val, config.batchsize, 4, test_transforms, sampler_cls=SequentialSampler
)
mgr_test = DataManager(
    dset_test, config.batchsize, 4, test_transforms, sampler_cls=SequentialSampler
)

experiment = PyTorchExperiment(
    config,
    UNetTorch,
    name="BaselineUnetFocalLoss",
    save_path=save_path,
    checkpoint_freq=config.checkpoint_freq,
    gpu_ids=config.gpu_ids,
    val_score_key=config.val_score_key,
    metric_keys=config.metric_keys,
)

experiment.save()
net = experiment.run(
    mgr_train, mgr_val, val_score_mode=config.val_score_mode, verbose=config.verbose
)
net.eval()
experiment.test(
    net,
    mgr_test,
    verbose=config.verbose,
    metrics=config.nested_get("metrics"),
    metric_keys=config.metric_keys,
)
