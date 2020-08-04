import torch
from .basic_networks import BaseSegmentationTorchNetwork


class DoubleConv(torch.nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.InstanceNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.InstanceNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, **kwargs):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            self.up = torch.nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SimpleUNet(BaseSegmentationTorchNetwork):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)
        self.up1 = UpBlock(1024, 256)
        self.up2 = UpBlock(512, 128)
        self.up3 = UpBlock(256, 64)
        self.up4 = UpBlock(128, 64)
        self.outc = OutConv(64, num_classes)
        self.activation = torch.nn.Softmax(dim=1)

    def _build_model(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if not self.training:
            x = self.activation(x)

        return {"pred": x}
