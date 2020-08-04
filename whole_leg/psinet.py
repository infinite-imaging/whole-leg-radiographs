import logging
from abc import abstractmethod
import torch
from delira.models.backends.torch.abstract_network import AbstractPyTorchNetwork
from delira.models.backends.torch.utils import scale_loss

from delira_unet.model.nd_wrapper import (
    ConvWrapper as ConvNdTorch,
    PoolingWrapper as PoolingNdTorch,
    NormWrapper as NormNdTorch,
)
import torch
from torch.nn import functional as F
from functools import partial


class PsiNet(AbstractPyTorchNetwork):
    def __init__(
        self,
        num_classes,
        in_channels=1,
        depth=5,
        start_filts=64,
        n_dim=2,
        norm_layer="Batch",
        up_mode="transpose",
        merge_mode="concat",
    ):

        super().__init__()

        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for '
                'upsampling. Only "transpose" and '
                '"upsample" are allowed.'.format(up_mode)
            )

        if merge_mode in ("concat", "add"):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                '"{}" is not a valid mode for'
                "merging up and down paths. "
                'Only "concat" and '
                '"add" are allowed.'.format(up_mode)
            )

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                'up_mode "upsample" is incompatible '
                'with merge_mode "add" at the moment '
                "because it doesn't make sense to use "
                "nearest neighbour to reduce "
                "depth channels (by half)."
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self._norm_layer = norm_layer

        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleDict(
            [
                ["seg", torch.nn.ModuleList()],
                ["contour", torch.nn.ModuleList()],
                ["distance", torch.nn.ModuleList()],
            ]
        )

        self.conv_final = None

        self._build_model(
            n_dim=n_dim,
            num_classes=num_classes,
            in_channels=in_channels,
            depth=depth,
            start_filts=start_filts,
            norm_layer=norm_layer,
        )

        self.reset_params()
        self.final_activations = torch.nn.ModuleDict(
            [
                ["seg", torch.nn.LogSoftmax(dim=1)],
                ["contour", torch.nn.LogSoftmax(dim=1)],
                ["distance", torch.nn.Sigmoid()],
            ]
        )

    @staticmethod
    def weight_init(m):

        if isinstance(m, ConvNdTorch):
            torch.nn.init.xavier_normal_(m.conv.weight)
            torch.nn.init.constant_(m.conv.bias, 0)

    def reset_params(self):

        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x) -> dict:

        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        results = {k: x for k in self.up_convs.keys()}
        for key, convs in self.up_convs.items():
            for i, module in enumerate(convs):
                before_pool = encoder_outs[-(i + 2)]
                results[key] = module(before_pool, results[key])

            results[key] = self.final_activations[key](
                self.conv_final[key](results[key])
            )

        return {"pred_" + k: v for k, v in results.items()}

    def _build_model(
        self,
        n_dim,
        num_classes,
        in_channels=1,
        depth=5,
        start_filts=64,
        norm_layer="Batch",
    ) -> None:
        def conv3x3(
            n_dim, in_channels, out_channels, stride=1, padding=1, bias=True, groups=1
        ):
            return ConvNdTorch(
                n_dim,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )

        def upconv2x2(n_dim, in_channels, out_channels, mode="transpose"):
            if mode == "transpose":
                return ConvNdTorch(
                    n_dim,
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    transposed=True,
                )
            else:
                # out_channels is always going to be the same
                # as in_channels
                if n_dim == 2:
                    upsample_mode = "bilinear"
                elif n_dim == 3:
                    upsample_mode = "trilinear"
                else:
                    raise ValueError

                return torch.nn.Sequential(
                    torch.nn.Upsample(mode=upsample_mode, scale_factor=2),
                    conv1x1(n_dim, in_channels, out_channels),
                )

        def conv1x1(n_dim, in_channels, out_channels, groups=1):
            return ConvNdTorch(
                n_dim, in_channels, out_channels, kernel_size=1, groups=groups, stride=1
            )

        class DownConv(torch.nn.Module):
            def __init__(
                self, n_dim, in_channels, out_channels, pooling=True, norm_layer="Batch"
            ):
                super().__init__()

                self.n_dim = n_dim
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.pooling = pooling

                self.conv1 = conv3x3(self.n_dim, self.in_channels, self.out_channels)
                self.norm1 = NormNdTorch(norm_layer, n_dim, self.out_channels)
                self.conv2 = conv3x3(self.n_dim, self.out_channels, self.out_channels)
                self.norm2 = NormNdTorch(norm_layer, n_dim, self.out_channels)

                if self.pooling:
                    self.pool = PoolingNdTorch("Max", n_dim, 2)

            def forward(self, x):
                x = F.relu(self.norm1(self.conv1(x)))
                x = F.relu(self.norm2(self.conv2(x)))
                before_pool = x
                if self.pooling:
                    x = self.pool(x)
                return x, before_pool

        class UpConv(torch.nn.Module):
            def __init__(
                self,
                n_dim,
                in_channels,
                out_channels,
                merge_mode="concat",
                up_mode="transpose",
            ):
                super().__init__()

                self.n_dim = n_dim
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.merge_mode = merge_mode
                self.up_mode = up_mode

                self.upconv = upconv2x2(
                    self.n_dim, self.in_channels, self.out_channels, mode=self.up_mode
                )

                if self.merge_mode == "concat":
                    self.conv1 = conv3x3(
                        self.n_dim, 2 * self.out_channels, self.out_channels
                    )
                else:
                    # num of input channels to conv2 is same
                    self.conv1 = conv3x3(self.n_dim, out_channels, self.out_channels)
                self.norm1 = NormNdTorch(norm_layer, n_dim, self.out_channels)
                self.conv2 = conv3x3(self.n_dim, self.out_channels, self.out_channels)
                self.norm2 = NormNdTorch(norm_layer, n_dim, self.out_channels)

            def forward(self, from_down, from_up):
                from_up = self.upconv(from_up)
                if self.merge_mode == "concat":
                    x = torch.cat((from_up, from_down), 1)
                else:
                    x = from_up + from_down
                x = F.relu(self.norm1(self.conv1(x)))
                x = F.relu(self.norm1(self.conv2(x)))
                return x

        outs = in_channels
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(n_dim, ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            self.up_convs["seg"].append(
                UpConv(
                    n_dim, ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode
                )
            )
            self.up_convs["contour"].append(
                UpConv(
                    n_dim, ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode
                )
            )
            self.up_convs["distance"].append(
                UpConv(
                    n_dim, ins, outs, up_mode=self.up_mode, merge_mode=self.merge_mode
                )
            )

        self.conv_final = torch.nn.ModuleDict(
            [
                ["seg", conv1x1(n_dim, outs, num_classes)],
                ["contour", conv1x1(n_dim, outs, num_classes)],
                ["distance", conv1x1(n_dim, outs, num_classes)],
            ]
        )

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
                    _loss_val = crit_fn(
                        preds["pred_seg"],
                        preds["pred_contour"],
                        preds["pred_distance"],
                        data_dict["label_seg"],
                        data_dict["label_contour"],
                        data_dict["label_distance"],
                    )
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
            .squeeze(1)
            .to(torch.long)
        )
        contour = (
            torch.from_numpy(batch.pop("contour"))
            .to(output_device)
            .squeeze(1)
            .to(torch.long)
        )
        distance = (
            torch.from_numpy(batch.pop("distance")).to(output_device).to(torch.float)
        )
        return {
            "data": data,
            "label_seg": label,
            "label_contour": contour,
            "label_distance": distance,
            **batch,
        }

