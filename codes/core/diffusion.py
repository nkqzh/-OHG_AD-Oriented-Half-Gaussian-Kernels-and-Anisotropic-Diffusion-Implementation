from __future__ import annotations
import math
import torch
from typing import Tuple
from .params import OHGADParams
from .orientation import oriented_Q, gamma_and_rhos
from .operators import second_derivative_maps, second_dir_derivative

Tensor = torch.Tensor

def fk_fh(grad_mag: Tensor, gamma: Tensor, k: float, h: float) -> Tuple[Tensor, Tensor]:
    term1 = torch.exp(- (grad_mag / (k + 1e-12))**2)
    term2 = torch.exp(- ((math.pi - gamma) / (math.pi * (k + 1e-12)))**2)
    fk = 0.5 * (term1 + term2)
    fh = torch.exp(- (grad_mag / (h + 1e-12))**2)
    return fk, fh

def ohgad_step(img: Tensor, p: OHGADParams) -> Tensor:
    grad_mag, theta1, theta2 = oriented_Q(img, p.mu, p.lam, p.dtheta_deg)
    gamma, rho1, rho2 = gamma_and_rhos(grad_mag, theta1, theta2, p.a)
    eta = 0.5 * (theta1 + theta2)
    I_xx, I_yy, I_xy = second_derivative_maps(img)
    # 正定扩散：两个直化方向的二阶导之和 + eta 方向项
    I_rho1 = second_dir_derivative(I_xx, I_yy, I_xy, rho1)
    I_rho2 = second_dir_derivative(I_xx, I_yy, I_xy, rho2)
    I_eteta  = second_dir_derivative(I_xx, I_yy, I_xy, eta)
    fk, fh = fk_fh(grad_mag, gamma, p.k, p.h)
    dI = fk * (I_rho1 + I_rho2 + fh * I_eteta)
    return img + p.dt * dI

def ohgad_denoise(img: Tensor, p: OHGADParams) -> Tensor:
    x = img.clone()
    for _ in range(p.iters):
        x = ohgad_step(x, p)
        x = x.clamp(0.0, 1.0)
    return x
