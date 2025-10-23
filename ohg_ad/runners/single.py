from __future__ import annotations
import os, json, time, torch
from dataclasses import asdict
from ..core.params import OHGADParams
from ..algorithms.ohgad_new import run_once

def main_cli(args=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--gt', default=None)
    p.add_argument('--sigma', type=float, default=0.0)
    p.add_argument('--dtheta', type=float, default=2.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--iters', type=int, default=12)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='results/exp')
    # PDE/核/控制参数
    p.add_argument('--k', type=float, default=0.3)
    p.add_argument('--h', type=float, default=0.1)
    p.add_argument('--a', type=float, default=0.1)
    p.add_argument('--mu', type=float, default=5.0)
    p.add_argument('--lam', type=float, default=1.0)
    # ✅ 注意：带短横线的参数，属性名是下划线形式
    p.add_argument('--auto-iters', dest='auto_iters', action='store_true')
    p.add_argument('--iters-max',  dest='iters_max',  type=int, default=30)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--ref', choices=['psnr','ssim'], default='psnr')
    args = p.parse_args(args)

    # 可复现
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    params = OHGADParams(
        k=args.k, h=args.h, a=args.a, mu=args.mu, lam=args.lam,
        dtheta_deg=args.dtheta, dt=args.dt, iters=args.iters
    )

    # ✅ 这里用 args.auto_iters（下划线），不要写成 args.auto-iters
    if not args.auto_iters:
        start = time.time()
        rep = run_once(args.input, args.gt, args.sigma, params, args.device, args.out)
        rep["elapsed_sec"] = round(time.time() - start, 3)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    # 自动选最佳 iters（early-stop）
    best = None
    stay = 0
    for it in range(1, args.iters_max + 1):
        p2 = OHGADParams(**{**asdict(params), "iters": it})
        rep = run_once(
            args.input, args.gt, args.sigma, p2, args.device,
            os.path.join(args.out, f"iters_{it:02d}")
        )
        score = rep["report"].get(args.ref, float("-inf"))
        if (best is None) or (score > best["score"]):
            best = {"iters": it, "score": score, "report": rep["report"]}
            stay = 0
        else:
            stay += 1
        if stay >= args.patience:
            break

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best": best}, ensure_ascii=False, indent=2))
