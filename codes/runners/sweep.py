# codes/runners/sweep.py
from __future__ import annotations
import os, json, time, itertools, yaml
from dataclasses import asdict
from typing import Dict, Any

from ..core.params import OHGADParams
from ..algorithms.ohg_ad import run_once  # 若算法文件名不同，这里对应改一下

# ---------- 辅助：参数名格式化/目录名 ----------
def _fmt_float(x: float, ndigits: int = 4) -> str:
    """把浮点数格式化为紧凑字符串：最多 ndigits 小数，去掉多余 0 和点"""
    s = f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _param_dirname(p: OHGADParams) -> str:
    # 目录名稳定、可读，用双下划线分隔
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

# ---------- 安全追加 JSONL，带 Windows 锁回退 ----------
def _safe_append_jsonl(path: str, row: Dict[str, Any], retries: int = 10, delay: float = 0.25) -> str:
    line = json.dumps(row, ensure_ascii=False)
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)

    if os.path.isdir(path):
        # 若有人把 metrics.jsonl 建成了目录，就回退到进程专属文件
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

    # 最终回退
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

    # 顶层目录仅放汇总日志/最优记录；大量结果统一放 runs/ 下
    os.makedirs(out_dir, exist_ok=True)
    runs_parent = os.path.join(out_dir, cfg.get("runs_dir", "runs"))
    os.makedirs(runs_parent, exist_ok=True)

    # 记录实际使用配置
    with open(os.path.join(out_dir, "config.used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    log_path = os.path.join(out_dir, "metrics.jsonl")
    best = None

    for pt in _grid(space):
        # 组装参数（允许 defaults 覆盖缺省值）
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

        # 每组参数一个“结果目录”，统一收纳到 runs/ 下
        param_dir = os.path.join(runs_parent, _param_dirname(p))
        os.makedirs(param_dir, exist_ok=True)

        for img in images:
            rep = run_once(
                img, img, sigma, p,
                defaults.get("device", "cuda"),
                param_dir  # ← 输出都落到该参数目录里，文件名带图片名不会互相覆盖
            )

            row = {
                "img": os.path.basename(img),
                "sigma": sigma,
                "params": asdict(p),
                "psnr": rep["report"].get("psnr"),
                "ssim": rep["report"].get("ssim"),
                "outputs": rep  # 保留输出路径
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
