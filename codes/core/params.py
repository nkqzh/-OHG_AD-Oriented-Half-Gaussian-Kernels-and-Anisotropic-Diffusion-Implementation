from __future__ import annotations
from dataclasses import dataclass

@dataclass
class OHGADParams:
    # 连续控制 + 半各向异性高斯半核参数（与论文/报告一致）
    k: float = 0.3          # 边缘保持项强度
    h: float = 0.1          # 平坦区各向同性项强度
    a: float = 0.1          # γ 直化强度（低噪0.1；高噪0.2）
    mu: float = 5.0         # 半高斯长度 μ
    lam: float = 1.0        # 半高斯宽度 λ
    dtheta_deg: float = 2.0 # 角度离散 Δθ=π/90≈2°
    dt: float = 0.05        # 显式欧拉步长（建议 ≤0.1）
    iters: int = 12
