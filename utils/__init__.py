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
    Right-aligned convention: valid tokens on the right, padding on the left.
    mask: (B, L), 1 = valid, 0 = pad. Returns (B,) long tensor of last valid index per row.
    """
    lengths = mask.sum(dim=1).clamp(min=1).long()
    return (lengths - 1)


def expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float, reduction: str = 'mean') -> torch.Tensor:
    """
    Expectile regression loss.
    
    tau: expectile level (0 < tau < 1)
    - tau = 0.5: MSE (평균).
    - tau > 0.5: upper expectile → 낙관적(optimistic) V 학습에 사용.
    - tau < 0.5: lower expectile → 비관적(pessimistic) 추정.
    
    현재 구현: weight = (1-tau) when pred >= target, else tau.
    → pred < target 인 구간에 더 큰 패널티를 주면 under-estimate를 줄이므로,
      tau < 0.5 이면 lower expectile(비관적), tau > 0.5 이면 upper(낙관적)에 해당.
    """
    diff = pred - target
    weight = torch.where(diff >= 0, 1.0 - tau, tau)
    loss = weight * (diff ** 2)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'none':
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")
