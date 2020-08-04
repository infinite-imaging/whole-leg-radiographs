from torch.optim.lr_scheduler import CosineAnnealingLR
from torchtools.lr_scheduler import DelayerScheduler

from delira.training.callbacks import DefaultPyTorchSchedulerCallback


class DelayedCosineAnnealingLRCallback(DefaultPyTorchSchedulerCallback):
    def __init__(self, optimizer, delay_epochs, cosine_annealing_epochs):
        super().__init__()

        base_scheduler = CosineAnnealingLR(optimizer, cosine_annealing_epochs)
        self.scheduler = DelayerScheduler(optimizer, delay_epochs, base_scheduler)
