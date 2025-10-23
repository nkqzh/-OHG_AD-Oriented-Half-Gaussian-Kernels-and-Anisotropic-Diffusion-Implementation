# -*- coding: utf-8 -*-
"""
方向扫描与直化 / Orientation Scan & Rectification
------------------------------------------------
1) 方向能量 Q(x,y,θ) = (I * K_θ)(x,y)，从中取：
   ||∇I|| ≈ max_θ Q - min_θ Q；θ1=argmax Q，θ2=argmin Q
2) 直化角 γ（式(7)）以及两条“直化方向” ρ1, ρ2：
   记 β = wrap(|θ1 - θ2|) ∈ [0,π]，
   γ  = β + π·exp(-||∇I||/a^2) - π·exp(-1/a^2)  （≤ π）

   决定 ρ1/ρ2 的表格规则等价于：
     cond = (θ1 > θ2)
     half = (γ - β)/2
     ρ1 = θ1 ± half, ρ2 = θ2 ∓ half  （符号由 cond 决定）
"""
from __future__ import annotations
import math
from typing import Tuple
import torch
from .kernels import build_angle_kernels
from .operators import conv2d_reflect

Tensor = torch.Tensor

def ensure_bchw(img: Tensor) -> Tensor:
    if img.dim() == 2:  img = img[None, None]
    elif img.dim() == 3: img = img[None]
    assert img.dim() == 4 and img.size(1) == 1, "Expect grayscale BCHW."
    return img

def oriented_Q(img: Tensor, mu: float, lam: float, dtheta_deg: float) -> Tuple[Tensor, Tensor, Tensor]:
    img = ensure_bchw(img)
    device, dtype = img.device, img.dtype
    kernels, angles = build_angle_kernels(mu, lam, dtheta_deg,
                                          device_str=str(device),
                                          dtype_str=str(dtype).split(".")[-1])
    Q = conv2d_reflect(img, kernels)  # (B, n_angles, H, W)
    Qmax, idx_max = Q.max(dim=1, keepdim=True)
    Qmin, idx_min = Q.min(dim=1, keepdim=True)
    grad_mag = (Qmax - Qmin)
    angles_rad = angles.mul_(torch.pi/180.0).view(1, -1, 1, 1)
    theta1 = torch.gather(angles_rad.expand(img.size(0), -1, img.size(2), img.size(3)), 1, idx_max)
    theta2 = torch.gather(angles_rad.expand(img.size(0), -1, img.size(2), img.size(3)), 1, idx_min)
    return grad_mag, theta1, theta2

def wrap_angle(a: Tensor) -> Tensor:
    return (a % (2*math.pi))

def gamma_and_rhos(grad_mag: Tensor, theta1: Tensor, theta2: Tensor, a: float):
    beta = (theta1 - theta2).abs()
    beta = torch.minimum(beta, 2 * math.pi - beta)  # -> [0, π]

    term_flat  = torch.exp(-(grad_mag / (a**2 + 1e-12)))
    term_const = math.exp(-(1.0 / (a**2 + 1e-12)))

    gamma = beta + math.pi * term_flat - math.pi * term_const
    pi_tensor = torch.tensor(math.pi, device=gamma.device, dtype=gamma.dtype)
    gamma = torch.minimum(gamma, pi_tensor)

    cond = (theta1 > theta2)
    half = (gamma - beta) * 0.5
    rho1 = torch.where(cond, theta1 + half, theta1 - half)
    rho2 = torch.where(cond, theta2 - half, theta2 + half)
    return gamma, wrap_angle(rho1), wrap_angle(rho2)
