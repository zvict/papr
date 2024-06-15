# credit: https://github.com/princeton-vl/SNP
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from .utils import activation_func


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='none'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == 'instance':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        elif norm == 'batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        elif norm == 'none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm='none'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == 'instance':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif norm == 'batch':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif norm == 'none':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, single=False, norm='none'):
        super().__init__()
        # print('down norm:', norm)

        if single:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                SingleConv(in_channels, out_channels, norm=norm)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, norm=norm)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            if single:
                self.conv = SingleConv(
                    in_channels, out_channels, in_channels // 2, norm=norm)
            else:
                self.conv = DoubleConv(
                    in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            if single:
                self.conv = SingleConv(in_channels, out_channels, norm=norm)
            else:
                self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpSample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, single=False, norm='none'):
        super().__init__()
        # print('up norm:', norm)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            if single:
                self.conv = SingleConv(
                    in_channels, out_channels, in_channels // 2, norm=norm)
            else:
                self.conv = DoubleConv(
                    in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2)
            if single:
                self.conv = SingleConv(in_channels, out_channels, norm=norm)
            else:
                self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, single=True, norm='none', render_scale=1, last_act='none',
                 use_amp=False, amp_dtype=torch.float16, affine_layer=-1):
        super(SmallUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.affine_layer = affine_layer

        assert (render_scale == 1 or render_scale == 2)
        self.render_scale = render_scale

        self.inc = SingleConv(n_channels, 128, norm=norm)
        self.down1 = Down(128, 256, single=single, norm=norm)
        self.down2 = Down(256, 512, single=single, norm=norm)
        self.up1 = Up(512, 256, bilinear, single=single, norm=norm)
        self.up2 = Up(256, 128, bilinear, single=single, norm=norm)

        if render_scale == 2:
            self.up3 = UpSample(128, 128, bilinear, single=False, norm=norm)

        self.outc = OutConv(128, n_classes)
        self.last_act = activation_func(last_act)

    def forward(self, x, log=False, gamma=None, beta=None):
        if self.affine_layer >= 0:
            assert gamma is not None and beta is not None

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            if self.affine_layer == 0:
                B, C, H, W = x.shape
                assert gamma.shape == (C,) and beta.shape == (C,)
                # print(gamma.mean(), beta.mean())
                x = x * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

            x1 = self.inc(x)
            if self.affine_layer == 1:
                B, C, H, W = x1.shape
                assert gamma.shape == (C,) and beta.shape == (C,)
                x1 = x1 * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

            x2 = self.down1(x1)
            if self.affine_layer == 2:
                B, C, H, W = x2.shape
                assert gamma.shape == (C,) and beta.shape == (C,)
                x2 = x2 * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

            x3 = self.down2(x2)
            if self.affine_layer == 3:
                B, C, H, W = x3.shape
                assert gamma.shape == (C,) and beta.shape == (C,)
                x3 = x3 * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

            x = self.up1(x3, x2)
            if self.affine_layer == 4:
                B, C, H, W = x.shape
                assert gamma.shape == (C,) and beta.shape == (C,)
                x = x * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

            x = self.up2(x, x1)
            if self.affine_layer == 5:
                B, C, H, W = x.shape
                assert gamma.shape == (C,) and beta.shape == (C,)
                x = x * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)

            if self.render_scale == 2:
                x = self.up3(x)
            logits = self.outc(x)

            logits = self.last_act(logits)

            if log:
                print(x1.dtype, x2.dtype, x3.dtype, x.dtype, logits.dtype)

            return logits
