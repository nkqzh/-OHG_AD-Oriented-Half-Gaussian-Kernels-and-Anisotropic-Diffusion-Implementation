# -*- coding: utf-8 -*-
"""指标包装 / Metric wrappers: PSNR & SSIM"""
from __future__ import annotations
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim

def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    return float(_psnr(a, b, data_range=data_range))

def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    return float(_ssim(a, b, data_range=data_range))
