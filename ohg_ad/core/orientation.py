from __future__ import annotations
import math
from typing import Tuple
import torch
from .kernels import build_angle_kernels
from .operators import conv2d_reflect

Tensor = torch.Tensor

def ensure_bchw(img: Tensor) -> Tensor:
    if img.dim() == 2:
        img = img[None, None]
    elif img.dim() == 3:
        img = img[None]
    assert img.dim() == 4 and img.size(1) == 1, "Expect grayscale BCHW."
    return img

def oriented_Q(img: Tensor, mu: float, lam: float, dtheta_deg: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    扫描 θ∈[0,360) 的 Q(x,y,θ)，取全局极大/极小：||∇I||、θ1、θ2（弧度）
    返回: (grad_mag, theta1, theta2) 形状均为 B×1×H×W。
    """
    img = ensure_bchw(img)
    device = img.device
    dtype = img.dtype
    kernels, angles = build_angle_kernels(mu, lam, dtheta_deg,
                                          device_str=str(device),
                                          dtype_str=str(dtype).split(".")[-1])
    # 反射边界
    Q = conv2d_reflect(img, kernels)  # (B, n_angles, H, W)
    Qmax, idx_max = Q.max(dim=1, keepdim=True)
    Qmin, idx_min = Q.min(dim=1, keepdim=True)
    grad_mag = (Qmax - Qmin)
    angles_rad = angles.mul_(torch.pi / 180.0).view(1, -1, 1, 1)
    theta1 = torch.gather(angles_rad.expand(img.size(0), -1, img.size(2), img.size(3)), 1, idx_max)
    theta2 = torch.gather(angles_rad.expand(img.size(0), -1, img.size(2), img.size(3)), 1, idx_min)
    return grad_mag, theta1, theta2

def wrap_angle(a: Tensor) -> Tensor:
    return (a % (2 * math.pi))

def gamma_and_rhos(grad_mag: Tensor, theta1: Tensor, theta2: Tensor, a: float):
    """
    γ 直化（式(7)）+ 表格规则计算 ρ1, ρ2（弧度）
    注意：张量指数用 torch.exp；常数指数用 math.exp（避免 torch.exp(float) 报错）
    """
    beta = (theta1 - theta2).abs()
    beta = torch.minimum(beta, 2 * math.pi - beta)  # -> [0, π]

    term_flat  = torch.exp(-(grad_mag / (a**2 + 1e-12)))      # Tensor
    term_const = math.exp(-(1.0 / (a**2 + 1e-12)))            # float 常数

    gamma = beta + math.pi * term_flat - math.pi * term_const  # Tensor（广播安全）
    pi_tensor = torch.tensor(math.pi, device=gamma.device, dtype=gamma.dtype)
    gamma = torch.minimum(gamma, pi_tensor)

    cond = (theta1 > theta2)
    half = (gamma - beta) * 0.5
    rho1 = torch.where(cond, theta1 + half, theta1 - half)
    rho2 = torch.where(cond, theta2 - half, theta2 + half)
    return gamma, wrap_angle(rho1), wrap_angle(rho2)
