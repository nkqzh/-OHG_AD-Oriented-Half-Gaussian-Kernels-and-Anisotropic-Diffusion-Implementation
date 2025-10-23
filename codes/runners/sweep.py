# -*- coding: utf-8 -*-
"""
网格/细化搜索 / Hyper-parameter sweep
-------------------------------------
- 从 YAML 读取数据集/默认参数/搜索空间
- 所有参数组合的输出目录统一放到 <out>/runs/<参数摘要> 下，避免顶层目录爆炸
- 顶层仅保留：metrics.jsonl、best.json、config.used.yaml
- 解决 Windows 文件锁：metrics.jsonl 采用安全追加，必要时回退到进程专属文件
"""
from __future__ import annotations
import os, json, time, itertools, yaml
from dataclasses import asdict
from typing import Dict, Any

from ..core.params import OHGADParams
from ..algorithms.ohg_ad import run_once  # 若算法文件名不同，这里对应改一下

def _fmt_float(x: float, ndigits: int = 4) -> str:
    """把浮点数格式化为紧凑字符串：最多 ndigits 小数，去掉多余 0 和点"""
    s = f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _param_dirname(p: OHGADParams) -> str:
    return (
        f"k{_fmt_float(p.k)}__h{_fmt_float(p.h)}__a{_fmt_float(p.a)}__"
        f"mu{_fmt_float(p.mu)}__lam{_fmt_float(p.lam)}__"
        f"dt{_fmt_float(p.dt)}__it{int(p.iters)}"
    )

def _grid(space: dict):
    keys = list(space.keys())
    vals = [space[k] if isinstance(space[k], (list, tuple)) else [space[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def _safe_append_jsonl(path: str, row: Dict[str, Any], retries: int = 10, delay: float = 0.25) -> str:
    line = json.dumps(row, ensure_ascii=False)
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)

    if os.path.isdir(path):
        alt = os.path.join(dirpath, f"metrics.{os.getpid()}.jsonl")
        with open(alt, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return alt

    for _ in range(retries):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return path
        except PermissionError:
            time.sleep(delay)

    alt = os.path.join(dirpath, f"metrics.{int(time.time())}.{os.getpid()}.jsonl")
    with open(alt, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return alt

# ---------- 主流程 ----------
def run_sweep(cfg_path: str, out_dir: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    images = cfg["dataset"]["images"]
    sigma  = cfg["dataset"].get("sigma", 0)
    defaults = cfg.get("defaults", {})
    space = cfg["search"]
    ref_metric = cfg.get("ref_metric", "psnr")

    os.makedirs(out_dir, exist_ok=True)
    runs_parent = os.path.join(out_dir, cfg.get("runs_dir", "runs"))
    os.makedirs(runs_parent, exist_ok=True)

    with open(os.path.join(out_dir, "config.used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    log_path = os.path.join(out_dir, "metrics.jsonl")
    best = None

    for pt in _grid(space):
        params_dict = {
            **defaults,
            **{k: v for k, v in pt.items() if k in ["k","h","a","mu","lam","dtheta","dt","iters"]}
        }
        p = OHGADParams(
            k=params_dict.get("k", 0.3),
            h=params_dict.get("h", 0.1),
            a=params_dict.get("a", 0.1),
            mu=params_dict.get("mu", 5.0),
            lam=params_dict.get("lam", 1.0),
            dtheta_deg=params_dict.get("dtheta", 2.0),
            dt=params_dict.get("dt", 0.05),
            iters=params_dict.get("iters", 8),
        )

        param_dir = os.path.join(runs_parent, _param_dirname(p))
        os.makedirs(param_dir, exist_ok=True)

        for img in images:
            rep = run_once(
                img, img, sigma, p,
                defaults.get("device", "cuda"),
                param_dir
            )

            row = {
                "img": os.path.basename(img),
                "sigma": sigma,
                "params": asdict(p),
                "psnr": rep["report"].get("psnr"),
                "ssim": rep["report"].get("ssim"),
                "outputs": rep
            }
            used_log = _safe_append_jsonl(log_path, row)

            score = row.get(ref_metric, float("-inf")) or float("-inf")
            if (best is None) or (score > best["score"]):
                best = {"score": score, "row": row, "log": used_log}

    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "best": best,
        "log": log_path,
        "runs_parent": runs_parent
    }, ensure_ascii=False, indent=2))
