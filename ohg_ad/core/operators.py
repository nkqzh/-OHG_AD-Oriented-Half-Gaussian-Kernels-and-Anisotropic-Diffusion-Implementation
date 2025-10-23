from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple

Tensor = torch.Tensor

def conv2d_reflect(img: Tensor, weight: Tensor) -> Tensor:
    pad_h = (weight.size(-2) - 1) // 2
    pad_w = (weight.size(-1) - 1) // 2
    img_pad = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    return F.conv2d(img_pad, weight)

def second_derivative_maps(img: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    device = img.device; dtype = img.dtype
    kxx = torch.tensor([[0,0,0],[1,-2,1],[0,0,0]], dtype=dtype, device=device)[None,None]
    kyy = torch.tensor([[0,1,0],[0,-2,0],[0,1,0]], dtype=dtype, device=device)[None,None]
    kxy = (1/4.0)*torch.tensor([[1,0,-1],[0,0,0],[-1,0,1]], dtype=dtype, device=device)[None,None]
    I_xx = conv2d_reflect(img, kxx)
    I_yy = conv2d_reflect(img, kyy)
    I_xy = conv2d_reflect(img, kxy)
    return I_xx, I_yy, I_xy

def second_dir_derivative(I_xx: Tensor, I_yy: Tensor, I_xy: Tensor, eta: Tensor) -> Tensor:
    c = torch.cos(eta); s = torch.sin(eta)
    return (c*c)*I_xx + 2.0*(c*s)*I_xy + (s*s)*I_yy
