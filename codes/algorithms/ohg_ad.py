# -*- coding: utf-8 -*-
"""
高层算法封装 / High-level API
-----------------------------
- to_gray_tensor / save_gray_tensor：简化 I/O 与设备管理
- ohgad_denoise(img, params)：核心 denoise 入口（调用 core.diffusion）
- run_once(...)：单次实验，含 I/O、指标计算与 report/metrics 写盘
"""
from __future__ import annotations
import os, json, torch, numpy as np
from imageio.v2 import imread, imwrite
from ..core.params import OHGADParams
from ..core.diffusion import ohgad_denoise as _denoise
from ..core.metrics import psnr as _psnr, ssim as _ssim

def to_gray_tensor(path: str, device: str = "cpu"):
    arr = imread(path)
    if arr.ndim == 3:
        arr = (0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]).astype(np.float32)
    arr = arr.astype(np.float32)
    if arr.max() > 1.1: arr = arr / 255.0
    t = torch.from_numpy(arr)[None, None].to(device)
    return t

def save_gray_tensor(t: torch.Tensor, path: str):
    arr = t.detach().cpu().clamp(0,1)[0,0].numpy()
    arr = (arr*255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, arr)

def ohgad_denoise(img: torch.Tensor, params: OHGADParams) -> torch.Tensor:
    return _denoise(img, params)

def run_once(input_path: str, gt_path: str | None, sigma: float, params: OHGADParams,
             device: str = "cuda", out_dir: str = "results/exp"):
    """
    单次实验：从 input 读取灰度图，按 sigma 加噪（若 sigma>0），运行去噪并评估。
    输出：
      - noisy/denoised PNG
      - report.json（完整信息）
      - metrics.jsonl（可累计的逐行日志）
    """
    import time
    from dataclasses import asdict

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    clean = to_gray_tensor(input_path, device=str(dev))
    if sigma > 0:
        torch.manual_seed(0)
        noise = torch.randn_like(clean) * (sigma / 255.0)
        noisy = (clean + noise).clamp(0.0, 1.0)
        gt = clean
    else:
        noisy = clean
        gt = to_gray_tensor(gt_path, device=str(dev)) if gt_path else None

    t0 = time.time()
    with torch.no_grad():
        den = ohgad_denoise(noisy, params)
    elapsed = time.time() - t0

    stem = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    noisy_out = os.path.join(out_dir, f"{stem}_noisy.png")
    den_out   = os.path.join(out_dir, f"{stem}_den_ohgad.png")
    save_gray_tensor(noisy, noisy_out)
    save_gray_tensor(den,   den_out)

    report = {}
    if gt is not None:
        gt_np, noisy_np, den_np = gt[0,0].cpu().numpy(), noisy[0,0].cpu().numpy(), den[0,0].cpu().numpy()
        report = {
            "psnr": _psnr(den_np, gt_np, 1.0),
            "ssim": _ssim(den_np, gt_np, 1.0),
            "psnr(noisy,gt)": _psnr(noisy_np, gt_np, 1.0),
            "psnr(den,gt)": _psnr(den_np, gt_np, 1.0)
        }

    summary = {
        "input": input_path,
        "gt": gt_path,
        "sigma": sigma,
        "params": asdict(params),
        "device": str(dev),
        "outputs": {"noisy": noisy_out, "denoised": den_out},
        "report": report,
        "elapsed_sec": round(elapsed, 3),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    metrics_row = {
        "img": os.path.basename(input_path), "sigma": sigma, "params": summary["params"],
        "psnr": report.get("psnr"), "ssim": report.get("ssim"),
        "psnr(noisy,gt)": report.get("psnr(noisy,gt)"), "psnr(den,gt)": report.get("psnr(den,gt)"),
        "elapsed_sec": summary["elapsed_sec"],
    }
    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")

    print(json.dumps({"noisy": noisy_out, "denoised": den_out, "report": report}, ensure_ascii=False, indent=2))
    return {"noisy": noisy_out, "denoised": den_out, "report": report}
