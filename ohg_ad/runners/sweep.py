from __future__ import annotations
import os, json, itertools, yaml, numpy as np
from dataclasses import asdict
from ..core.params import OHGADParams
from ..algorithms.ohgad_new import run_once

def _grid(space: dict):
    keys = list(space.keys())
    vals = [space[k] if isinstance(space[k], (list, tuple)) else [space[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def run_sweep(cfg_path: str, out_dir: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    images = cfg["dataset"]["images"]
    sigma  = cfg["dataset"].get("sigma", 0)
    defaults = cfg.get("defaults", {})
    space = cfg["search"]

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "metrics.jsonl")
    best = None

    for pt in _grid(space):
        params_dict = {**defaults, **{k:v for k,v in pt.items() if k in ["k","h","a","mu","lam","dtheta","dt","iters"]}}
        # normalize keys to OHGADParams
        p = OHGADParams(k=params_dict.get("k",0.3), h=params_dict.get("h",0.1), a=params_dict.get("a",0.1),
                        mu=params_dict.get("mu",5.0), lam=params_dict.get("lam",1.0),
                        dtheta_deg=params_dict.get("dtheta",2.0), dt=params_dict.get("dt",0.05),
                        iters=params_dict.get("iters",8))
        for img in images:
            rep = run_once(img, img, sigma, p, defaults.get("device","cuda"),
                           os.path.join(out_dir, f"k{p.k}_h{p.h}_a{p.a}_mu{p.mu}_lam{p.lam}_dt{p.dt}_it{p.iters}"))
            row = {
                "img": os.path.basename(img), "sigma": sigma, "params": asdict(p),
                "psnr": rep["report"].get("psnr"), "ssim": rep["report"].get("ssim")
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            score = rep["report"].get(cfg.get("ref_metric","psnr"), -1e9)
            if (best is None) or (score > best.get("score",-1e9)):
                best = {"score": score, "row": row}

    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    return {"best": best, "log": log_path}
