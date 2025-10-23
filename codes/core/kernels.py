# -*- coding: utf-8 -*-
"""
半高斯导数核 / Oriented Half-Gaussian Derivative Kernels
---------------------------------------------------------
对每个 θ 构造一组“一半截断”的各向异性高斯一阶导核，配合 **反射边界卷积** 用于
Q(x,y,θ) = (I * K_θ)(x,y) 的方向扫描。

坐标旋转:
  x' =  cosθ·x + sinθ·y
  y' = -sinθ·x + cosθ·y

核定义（去 DC、L1 归一）:
  K_θ(x,y) = H(-y') · x' · exp( -x'^2/(2λ^2) - y'^2/(2μ^2) )

其中 H 为 Heaviside（取 y'<0 半平面），μ/λ 控制细长方向尺度。
"""
from __future__ import annotations
import math, functools
import torch

def _half_gaussian_derivative(mu: float, lam: float, theta_rad: float,
                              device: torch.device, dtype: torch.dtype):
    R = int(max(7, math.ceil(3 * max(mu, lam))))
    H = W = 2 * R + 1
    yy, xx = torch.meshgrid(
        torch.linspace(-R, R, H, device=device, dtype=dtype),
        torch.linspace(-R, R, W, device=device, dtype=dtype),
        indexing="ij",
    )
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    x_loc =  c * xx + s * yy
    y_loc = -s * xx + c * yy
    H_heavi = (y_loc < 0).to(dtype) + 0.5 * (y_loc == 0).to(dtype)
    ker = H_heavi * x_loc * torch.exp(-(x_loc**2) / (2 * lam**2) - (y_loc**2) / (2 * mu**2))
    ker = ker - ker.mean()
    ker = ker / (ker.abs().sum() + 1e-8)
    return ker[None, None, :, :]

@functools.lru_cache(maxsize=128)
def build_angle_kernels(mu: float, lam: float, dtheta_deg: float,
                        device_str: str = "cpu", dtype_str: str = "float32"):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    angles = torch.arange(0.0, 360.0, dtheta_deg, device=device, dtype=dtype)
    stacks = []
    for th in angles.tolist():
        stacks.append(_half_gaussian_derivative(mu, lam, math.radians(th), device, dtype))
    return torch.cat(stacks, dim=0), angles  # (n_angles,1,h,w), (n_angles,)
