"""Synthetic sunglasses occlusion (eye-region dark ellipse)."""

from __future__ import annotations

import torch


def apply_sunglasses(x: torch.Tensor, darkness: float = 0.05) -> torch.Tensor:
    """
    x: (B,C,H,W) or (C,H,W) normalized roughly in [-1,1] or [0,1].
    Draws a dark horizontal band / twin ellipses over the upper face (eyes).
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    b, c, h, w = x.shape
    out = x.clone()
    # eye band: ~28%–48% of height, full width with soft elliptical cutouts
    y0, y1 = int(0.28 * h), int(0.48 * h)
    band = out[:, :, y0:y1, :]
    yy = torch.linspace(-1, 1, y1 - y0, device=x.device).view(1, 1, -1, 1)
    xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, -1)
    # two ellipses for left/right lenses
    left = ((xx + 0.35) / 0.35) ** 2 + (yy / 0.9) ** 2
    right = ((xx - 0.35) / 0.35) ** 2 + (yy / 0.9) ** 2
    mask = ((left <= 1.0) | (right <= 1.0)).float()
    # bridge bar
    bridge = (xx.abs() < 0.12) & (yy.abs() < 0.35)
    mask = torch.clamp(mask + bridge.float(), 0, 1)
    fill = torch.full_like(band, darkness * 2 - 1.0)  # near black in [-1,1]
    band = band * (1 - mask) + fill * mask
    out[:, :, y0:y1, :] = band
    return out[0] if single else out
