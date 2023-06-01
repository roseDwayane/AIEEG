# https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernel_size=7):
        super().__init__()
        padding = int((kernel_size - 1) / 2)

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
            #nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            #nn.ReLU(inplace=True)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels,kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = F.interpolate()
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x = self.up(x1)
        # input is CHW
        #diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        #diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(OutConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)

class UNet0(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet0, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, kernel_size=7)
        self.down1 = Down(64, 128, kernel_size=7)
        self.down2 = Down(128, 256,kernel_size=5)
        #self.down3 = Down(256, 512,kernel_size=3)
        #self.up1 = Up(512, 256, kernel_size=3)
        self.up2 = Up(256, 128, kernel_size=3)
        self.up3 = Up(128, 64, kernel_size=3)
        self.outc = OutConv(64, n_classes,kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x = self.up1(x4, x3)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, kernel_size=7)
        self.down1 = Down(64, 128, kernel_size=7)
        self.down2 = Down(128, 256,kernel_size=5)
        self.down3 = Down(256, 512,kernel_size=3)
        self.up1 = Up(512, 256, kernel_size=3)
        self.up2 = Up(256, 128, kernel_size=3)
        self.up3 = Up(128, 64, kernel_size=3)
        self.outc = OutConv(64, n_classes,kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, kernel_size=7)
        self.down1 = Down(64, 128, kernel_size=7)
        self.down2 = Down(128, 256,kernel_size=5)
        self.down3 = Down(256, 512,kernel_size=3)
        self.down4 = Down(512, 1024, kernel_size=3)
        self.up = Up(1024, 512, kernel_size=3)
        self.up1 = Up(512, 256, kernel_size=3)
        self.up2 = Up(256, 128, kernel_size=3)
        self.up3 = Up(128, 64, kernel_size=3)
        self.outc = OutConv(64, n_classes,kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits