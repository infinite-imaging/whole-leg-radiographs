import logging
from abc import abstractmethod
import torch
from delira.models.backends.torch.abstract_network import AbstractPyTorchNetwork
from delira.models.backends.torch.utils import scale_loss


class BaseSegmentationTorchNetwork(AbstractPyTorchNetwork):
    def __init__(self, *args, **kwargs):

        super().__init__()

        self._build_model(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict:

        raise NotImplementedError

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses={}, fold=0, **kwargs):

        assert (
            optimizers and losses
        ) or not optimizers, "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict["data"]
            preds = model(inputs)

            if data_dict:

                for key, crit_fn in losses.items():
                    _loss_val = crit_fn(preds["pred"], data_dict["label"])
                    loss_vals[key] = _loss_val.detach()
                    total_loss = total_loss + _loss_val

        if optimizers:
            optimizers["default"].zero_grad()
            # perform loss scaling via apex if half precision is enabled
            with scale_loss(total_loss, optimizers["default"]) as scaled_loss:
                scaled_loss.backward()
            optimizers["default"].step()

        return (
            {k: v.cpu().numpy() for k, v in loss_vals.items()},
            {k: v.detach() for k, v in preds.items()},
        )

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):

        data = torch.from_numpy(batch.pop("data")).to(input_device).to(torch.float)
        label = (
            torch.from_numpy(batch.pop("label"))
            .to(output_device)
            .to(torch.long)
            .squeeze(1)
        )
        return {"data": data, "label": label, **batch}

