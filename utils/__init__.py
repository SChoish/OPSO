# OPSO 공통 유틸리티
"""
공통 손실 함수 및 헬퍼.
- expectile_loss: Expectile regression (오프라인/온라인 V 학습)
- last_valid_index_from_mask: right-aligned convention; 1=valid, 0=pad; returns (B,) indices.
"""

import torch
import torch.nn.functional as F


def last_valid_index_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Right-aligned convention: valid on the right, padding on the left.
    mask: (B, L), 1 = valid, 0 = pad. Returns (B,) long tensor of last valid index per row
    (index of rightmost valid token). All-zero rows yield 0.
    """
    B, L = mask.shape
    device = mask.device
    idxs = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    masked = idxs * (mask > 0).long()
    return masked.max(dim=1).values


def expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float, reduction: str = 'mean') -> torch.Tensor:
    """
    Expectile regression loss (IQL 스타일).
    L(u) = |τ - 1(u<0)| * u^2, u = pred - target.
    - u < 0 (pred < target): weight = 1 - τ
    - u >= 0 (pred >= target): weight = τ
    tau > 0.5 이면 τ 쪽이 커서 overestimation에 더 패널티 → upper expectile / 낙관적 V에 사용.
    """
    diff = pred - target  # u
    weight = torch.where(diff >= 0, tau, 1.0 - tau)  # |τ - 1(u<0)|
    loss = weight * (diff ** 2)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'none':
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")
