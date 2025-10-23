# -*- coding: utf-8 -*-
"""
OHG-AD 参数定义 / Parameters for OHG-AD
---------------------------------------
核心 PDE 及核参数集中在一个 dataclass 中，便于统一传递与记录。

Notation（与论文/报告一致）:
- k, h: 控制函数强度（边缘保持/平坦区）
- a   : γ 直化强度（低噪 0.1；较高噪声可取 0.2）
- μ, λ: 半高斯核沿 (y', x') 的尺度
- Δθ  : 角度离散（度）
- dt  : 显式欧拉步长（建议 ≤0.1）
- iters: 迭代步数
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class OHGADParams:
    k: float = 0.3
    h: float = 0.1
    a: float = 0.1
    mu: float = 5.0
    lam: float = 1.0
    dtheta_deg: float = 2.0
    dt: float = 0.05
    iters: int = 12
