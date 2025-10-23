from __future__ import annotations
import json, os, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def plot_heatmap(jsonl_path: str, xgrid: str = "k", ygrid: str = "h", metric: str = "psnr", save_dir: str | None = None):
    table = defaultdict(dict)
    xs, ys = set(), set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            p = row["params"]
            x, y = p[xgrid], p[ygrid]
            xs.add(x); ys.add(y)
            table[y][x] = row.get(metric, None)
    xs = sorted(xs); ys = sorted(ys)
    Z = np.array([[table[y].get(x, np.nan) for x in xs] for y in ys], dtype=float)
    plt.figure()
    im = plt.imshow(Z, origin='lower', aspect='auto')
    plt.xticks(range(len(xs)), xs); plt.yticks(range(len(ys)), ys)
    plt.xlabel(xgrid); plt.ylabel(ygrid); plt.title(f"{metric} heatmap")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"heatmap_{xgrid}_{ygrid}_{metric}.png"), dpi=150, bbox_inches="tight")
    else:
        plt.show()
