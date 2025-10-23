from __future__ import annotations
import json, os, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def _load_rows(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def plot_heatmap(jsonl_path: str, xgrid: str = "k", ygrid: str = "h",
                 metric: str = "psnr", save_dir: str | None = None,
                 reduce: str = "max"):
    """
    将 metric（psnr/ssim）映射成 xgrid×ygrid 的热图。
    当存在多个重复 (xgrid,ygrid)（例如不同 iters）时，reduce 可选 'max'/'mean'。
    """
    rows = _load_rows(jsonl_path)
    # 只保留需要的参数完整的数据
    rows = [r for r in rows if r.get(metric) is not None and r.get("params", {}).get(xgrid) is not None and r.get("params", {}).get(ygrid) is not None]
    if not rows:
        raise ValueError(f"No rows with metric={metric}, xgrid={xgrid}, ygrid={ygrid} in {jsonl_path}")

    # 汇总到表格
    table = defaultdict(list)
    xs, ys = set(), set()
    for r in rows:
        p = r["params"]
        x, y = p.get(xgrid), p.get(ygrid)
        xs.add(x); ys.add(y)
        table[(x, y)].append(r.get(metric))

    xs = sorted(xs)
    ys = sorted(ys)
    Z = np.empty((len(ys), len(xs)), dtype=float)
    Z[:] = np.nan
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            vals = table.get((x, y), [])
            if not vals:
                continue
            if reduce == "mean":
                Z[j, i] = float(np.mean(vals))
            else:  # default 'max'
                Z[j, i] = float(np.max(vals))

    plt.figure()
    im = plt.imshow(Z, origin='lower', aspect='auto')
    plt.xticks(range(len(xs)), xs); plt.yticks(range(len(ys)), ys)
    plt.xlabel(xgrid); plt.ylabel(ygrid)
    plt.title(f"{metric.upper()} heatmap ({reduce})")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    os.makedirs(save_dir or ".", exist_ok=True)
    outp = os.path.join(save_dir or ".", f"heatmap_{xgrid}_{ygrid}_{metric}_{reduce}.png")
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close()
