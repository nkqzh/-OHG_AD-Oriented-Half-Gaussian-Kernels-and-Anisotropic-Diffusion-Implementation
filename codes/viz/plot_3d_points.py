from __future__ import annotations
import json, os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

def _save_placeholder(save_dir: str | None, metric: str, img_filter: str | None, reduce: str, reason: str) -> str:
    os.makedirs(save_dir or ".", exist_ok=True)
    fname = f"3d_{metric}_kha_{'all' if not img_filter else os.path.splitext(img_filter)[0]}_{reduce}.png"
    outp = os.path.join(save_dir or ".", fname)
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0.5, 0.6, "NO 3D DATA", ha="center", va="center", fontsize=18)
    plt.text(0.5, 0.4, reason, ha="center", va="center", fontsize=10)
    plt.savefig(outp, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[3D] 无可用数据，已生成占位图：{outp}")
    return outp

def plot_3d_kha_metric(jsonl_path: str, metric: str = "ssim",
                       save_dir: str | None = None, img_filter: str | None = None,
                       reduce: str = "max", show: bool = False) -> str:
    """
    3D 散点：x=k, y=h, z=a，颜色映射 metric（默认 SSIM）。
    - 若同一 (k,h,a) 有多条（不同 iters），用 reduce='max' 或 'mean' 聚合。
    - 无可用数据时会生成占位图，并打印原因。
    - 返回最终保存的文件路径。
    """
    print(f"[3D] 读取日志：{jsonl_path}；metric={metric}；img_filter={img_filter or 'ALL'}；reduce={reduce}")
    rows = _load_rows(jsonl_path)
    if img_filter:
        rows = [r for r in rows if str(r.get("img")) == img_filter]
    rows = [r for r in rows if r.get(metric) is not None]

    if not rows:
        return _save_placeholder(save_dir, metric, img_filter, reduce, "加载行为空或没有该指标")

    # 聚合到 (k,h,a)
    bucket = {}
    missing = 0
    for r in rows:
        p = r.get("params", {})
        if not all(k in p for k in ("k", "h", "a")):
            missing += 1
            continue
        key = (p["k"], p["h"], p["a"])
        bucket.setdefault(key, []).append(float(r[metric]))

    if not bucket:
        return _save_placeholder(save_dir, metric, img_filter, reduce, f"缺少 (k,h,a)；丢弃条数={missing}")

    xs, ys, zs, cs = [], [], [], []
    for (k, h, a), vals in bucket.items():
        v = float(np.mean(vals)) if reduce == "mean" else float(np.max(vals))
        xs.append(k); ys.append(h); zs.append(a); cs.append(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=cs, cmap='viridis')
    ax.set_xlabel('k'); ax.set_ylabel('h'); ax.set_zlabel('a')
    ax.set_title(f"3D {metric.upper()} on (k,h,a) [{reduce}]")
    cb = fig.colorbar(sc, shrink=0.6, aspect=12); cb.set_label(metric.upper())

    os.makedirs(save_dir or ".", exist_ok=True)
    fname = f"3d_{metric}_kha_{'all' if not img_filter else os.path.splitext(img_filter)[0]}_{reduce}.png"
    outp = os.path.join(save_dir or ".", fname)
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print(f"[3D] 保存成功：{outp}（点数={len(xs)}，缺失(k,h,a)条数={missing}）")
    return outp
