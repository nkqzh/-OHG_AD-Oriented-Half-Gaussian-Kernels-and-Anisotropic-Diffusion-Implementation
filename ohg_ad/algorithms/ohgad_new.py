from __future__ import annotations
import torch, numpy as np, os, json
from imageio.v2 import imread, imwrite
from . import __package__ as _pkg  # noqa: F401
from ..core.params import OHGADParams
from ..core.diffusion import ohgad_denoise as _denoise
from ..core.metrics import psnr as _psnr, ssim as _ssim

def to_gray_tensor(path: str, device: str = "cpu"):
    arr = imread(path)
    if arr.ndim == 3:
        arr = (0.2989 * arr[...,0] + 0.5870 * arr[...,1] + 0.1140 * arr[...,2]).astype(np.float32)
    arr = arr.astype(np.float32)
    if arr.max() > 1.1: arr = arr / 255.0
    t = torch.from_numpy(arr)[None,None].to(device)
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

    with torch.no_grad():
        den = ohgad_denoise(noisy, params)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    noisy_out = os.path.join(out_dir, f"{stem}_noisy.png")
    den_out   = os.path.join(out_dir, f"{stem}_den_ohgad.png")
    save_gray_tensor(noisy, noisy_out)
    save_gray_tensor(den, den_out)

    report = {}
    if gt is not None:
        gt_np, noisy_np, den_np = gt[0,0].cpu().numpy(), noisy[0,0].cpu().numpy(), den[0,0].cpu().numpy()
        report = {
            "psnr": _psnr(den_np, gt_np, 1.0),
            "ssim": _ssim(den_np, gt_np, 1.0),
            "psnr(noisy,gt)": _psnr(noisy_np, gt_np, 1.0),
            "psnr(den,gt)": _psnr(den_np, gt_np, 1.0)
        }
    os.makedirs(out_dir, exist_ok=True)
    print(json.dumps({"noisy": noisy_out, "denoised": den_out, "report": report}, ensure_ascii=False, indent=2))
    return {"noisy": noisy_out, "denoised": den_out, "report": report}
