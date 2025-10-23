from __future__ import annotations
import math
import functools
import torch

def _half_gaussian_derivative(mu: float, lam: float, theta_rad: float, device: torch.device, dtype: torch.dtype):
    R = int(max(7, math.ceil(3 * max(mu, lam))))
    H = W = 2 * R + 1
    yy, xx = torch.meshgrid(
        torch.linspace(-R, R, H, device=device, dtype=dtype),
        torch.linspace(-R, R, W, device=device, dtype=dtype),
        indexing="ij",
    )
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    x_loc =  c * xx + s * yy
    y_loc = -s * xx + c * yy
    H_heavi = (y_loc < 0).to(dtype) + 0.5 * (y_loc == 0).to(dtype)
    ker = H_heavi * x_loc * torch.exp(-(x_loc**2) / (2 * lam**2) - (y_loc**2) / (2 * mu**2))
    ker = ker - ker.mean()
    ker = ker / (ker.abs().sum() + 1e-8)
    return ker[None, None, :, :]

@functools.lru_cache(maxsize=128)
def build_angle_kernels(mu: float, lam: float, dtheta_deg: float, device_str: str = "cpu", dtype_str: str = "float32"):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    angles = torch.arange(0.0, 360.0, dtheta_deg, device=device, dtype=dtype)
    stacks = []
    for th in angles.tolist():
        stacks.append(_half_gaussian_derivative(mu, lam, math.radians(th), device, dtype))
    return torch.cat(stacks, dim=0), angles  # (n_angles,1,h,w), (n_angles,)
