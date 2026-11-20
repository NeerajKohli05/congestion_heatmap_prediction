"""
Residual Attention U-Net for Circuit Congestion Heatmap Prediction.
Designed specifically for multi-channel spatial circuit features and 2D overflow regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Residual Convolutional Block with GroupNorm/BatchNorm and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class AttentionGate(nn.Module):
    """
    Attention Gate to focus on localized routing congestion hotspots
    and filter out low-congestion background regions.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Match dimensions if upsampling had small size discrepancy
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CongestionUNet(nn.Module):
    """
    Multi-Channel Residual Attention U-Net for Congestion Map Regression.
    Input: (B, in_channels, H, W) e.g., 6 channels (Pin, Cell, RUDY_H, RUDY_V, RUDY_Total, Blockage)
    Output: (B, out_channels, H, W) e.g., 2 channels (H-Overflow, V-Overflow)
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        base_channels: int = 32
    ):
        super().__init__()
        c = base_channels

        # Encoder (Downsampling path)
        self.enc1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 8, c * 16)

        # Decoder (Upsampling path) with Attention Gates
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=c * 8, F_l=c * 8, F_int=c * 4)
        self.dec4 = ConvBlock(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=c * 4, F_l=c * 4, F_int=c * 2)
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=c * 2, F_l=c * 2, F_int=c)
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=c, F_l=c, F_int=c // 2)
        self.dec1 = ConvBlock(c * 2, c)

        # Final 1x1 Convolution + ReLU (congestion demand/overflow >= 0)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=1),
            nn.ReLU()  # Congestion overflow ratio cannot be negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with Attention Gates
        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([x4, d4], dim=1))

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([x1, d1], dim=1))

        out = self.out_conv(d1)
        return out
