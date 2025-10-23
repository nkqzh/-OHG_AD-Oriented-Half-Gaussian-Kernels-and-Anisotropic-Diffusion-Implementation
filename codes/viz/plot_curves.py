from __future__ import annotations
import json, os
import matplotlib.pyplot as plt
from collections import defaultdict

def _load_rows(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def _group_key(row, xkey):
    p = row.get("params", {})
    return (
        row.get("img"),
        p.get("k"), p.get("h"), p.get("a"),
        p.get("mu"), p.get("lam"),
        p.get("dtheta_deg") or p.get("dtheta"),
        p.get("dt"),
    )

def _smooth_curve(xs, ys, density_per_seg: int = 20):
    """Catmull–Rom 样条（端点用 clamped，避免过冲），返回致密采样的 (x_s, y_s)"""
    n = len(xs)
    if n <= 2 or density_per_seg <= 1:
        return xs, ys
    x_s, y_s = [], []
    for i in range(n - 1):
        x0, y0 = (xs[i-1], ys[i-1]) if i - 1 >= 0 else (xs[i], ys[i])
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i+1], ys[i+1]
        x3, y3 = (xs[i+2], ys[i+2]) if i + 2 < n else (xs[i+1], ys[i+1])
        m1 = 0.5 * (y2 - y0)
        m2 = 0.5 * (y3 - y1)
        for j in range(density_per_seg):
            t = j / float(density_per_seg)
            h00 =  2*t**3 - 3*t**2 + 1
            h10 =      t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 =      t**3 -    t**2
            y = h00*y1 + h10*m1 + h01*y2 + h11*m2
            x = x1 + (x2 - x1) * t
            if not x_s or x != x_s[-1]:
                x_s.append(x); y_s.append(float(y))
        # 末点
        if i == n-2:
            x_s.append(x2); y_s.append(float(y2))
    return x_s, y_s

def plot_curves(jsonl_path: str, xkey: str = "iters", metric: str = "psnr",
                save_dir: str | None = None, max_series_per_img: int = 8):
    rows = _load_rows(jsonl_path)
    rows = [r for r in rows if r.get("params", {}).get(xkey) is not None and r.get(metric) is not None]
    if not rows:
        raise ValueError(f"No rows with x={xkey} and metric={metric} in {jsonl_path}")
    by_img = defaultdict(list)
    for r in rows:
        by_img[r.get("img")].append(r)
    os.makedirs(save_dir or ".", exist_ok=True)
    for img, sub in by_img.items():
        groups = defaultdict(list)
        for r in sub:
            groups[_group_key(r, xkey)].append(r)
        keys_sorted = sorted(groups.keys())[:max_series_per_img]
        plt.figure()
        for key in keys_sorted:
            seq = groups[key]; seq.sort(key=lambda r: r["params"][xkey])
            xs = [r["params"][xkey] for r in seq]
            ys = [r[metric] for r in seq]
            xs_s, ys_s = _smooth_curve(xs, ys, density_per_seg=30)
            label = f"k={key[1]},h={key[2]},a={key[3]},dt={key[7]}"
            # 平滑曲线 + 原始采样点
            plt.plot(xs_s, ys_s, label=label)
            plt.plot(xs, ys, marker="o", linestyle="none", markersize=3)
        plt.xlabel(xkey); plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs {xkey}  [{img}]")
        if len(keys_sorted) > 1: plt.legend(fontsize=8)
        outp = os.path.join(save_dir or ".", f"curves_{metric}_vs_{xkey}_{os.path.splitext(str(img))[0]}.png")
        plt.savefig(outp, dpi=150, bbox_inches="tight"); plt.close()
