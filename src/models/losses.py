"""
Custom Loss Functions for Circuit Routing Congestion Heatmap Prediction.
Specifically designed for Electronic Design Automation (EDA) where high-congestion
regions and structural routing channels must be accurately captured.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HotspotWeightedLoss(nn.Module):
    """
    Weighted Huber / Smooth L1 loss with adaptive continuous weighting for congestion.
    Gives progressively higher importance to congested tiles and critical hotspots (> hotspot_threshold).
    """

    def __init__(self, hotspot_threshold: float = 0.70, hotspot_weight: float = 3.0, beta: float = 0.05):
        super().__init__()
        self.hotspot_threshold = hotspot_threshold
        self.hotspot_weight = hotspot_weight
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        # Huber loss
        huber = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)

        # Continuous quadratic emphasis on congested routing tracks
        weights = 1.0 + self.hotspot_weight * (target ** 2)

        # Additional step-up boost for critical DRC-risk hotspots
        hotspot_mask = target > self.hotspot_threshold
        weights = weights + (self.hotspot_weight * hotspot_mask.float())

        weighted_loss = huber * weights
        return weighted_loss.mean()


class PearsonCorrelationLoss(nn.Module):
    """
    Differentiable Pearson Correlation Coefficient (PCC) Loss.
    Maximizes linear correlation between predicted and ground-truth routing congestion.
    Loss = 1.0 - mean(PCC)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions per sample and per channel: (B, C, H*W)
        B, C = pred.shape[0], pred.shape[1]
        p_flat = pred.view(B, C, -1)
        t_flat = target.view(B, C, -1)

        p_mean = p_flat.mean(dim=2, keepdim=True)
        t_mean = t_flat.mean(dim=2, keepdim=True)

        p_diff = p_flat - p_mean
        t_diff = t_flat - t_mean

        numerator = (p_diff * t_diff).sum(dim=2)
        denominator = torch.sqrt((p_diff ** 2).sum(dim=2) * (t_diff ** 2).sum(dim=2) + self.eps)

        pcc = numerator / (denominator + self.eps)
        return 1.0 - pcc.mean()


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) Loss.
    Ensures spatial structures, gradients, and channel boundaries match ground truth.
    """

    def __init__(self, window_size: int = 7, in_channels: int = 2):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.register_buffer("kernel", self._create_gaussian_kernel(window_size, in_channels))

    def _create_gaussian_kernel(self, window_size: int, channels: int) -> torch.Tensor:
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = torch.outer(g, g)
        kernel_4d = kernel_2d.expand(channels, 1, window_size, window_size)
        return kernel_4d

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = self.kernel.to(pred.device)
        padding = self.window_size // 2

        mu_x = F.conv2d(pred, c, padding=padding, groups=self.in_channels)
        mu_y = F.conv2d(target, c, padding=padding, groups=self.in_channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.relu(F.conv2d(pred * pred, c, padding=padding, groups=self.in_channels) - mu_x_sq)
        sigma_y_sq = F.relu(F.conv2d(target * target, c, padding=padding, groups=self.in_channels) - mu_y_sq)
        sigma_xy = F.conv2d(pred * target, c, padding=padding, groups=self.in_channels) - mu_xy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


class CompositeCongestionLoss(nn.Module):
    """
    Combined objective function for Circuit Routing Congestion Prediction:
    L_total = L_hotspot_weighted + lambda_ssim * L_ssim + lambda_pcc * L_pcc
    """

    def __init__(
        self,
        hotspot_threshold: float = 0.70,
        hotspot_weight: float = 3.0,
        lambda_ssim: float = 0.15,
        lambda_pcc: float = 0.20,
        in_channels: int = 2
    ):
        super().__init__()
        self.hotspot_loss = HotspotWeightedLoss(hotspot_threshold, hotspot_weight)
        self.ssim_loss = SSIMLoss(window_size=7, in_channels=in_channels)
        self.pcc_loss = PearsonCorrelationLoss()
        self.lambda_ssim = lambda_ssim
        self.lambda_pcc = lambda_pcc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_hotspot = self.hotspot_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_pcc = self.pcc_loss(pred, target)
        return l_hotspot + self.lambda_ssim * l_ssim + self.lambda_pcc * l_pcc
