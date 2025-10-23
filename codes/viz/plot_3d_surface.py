from __future__ import annotations
import json, os, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.tri as mtri
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict

def _load_rows(path: str):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    return rows

def _mode(vals):
    if not vals: return None
    d=defaultdict(int)
    for v in vals: d[v]+=1
    return max(d.items(), key=lambda kv: kv[1])[0]

def plot_surface_kh_metric(
    jsonl_path: str,
    metric: str = "ssim",
    a_value: float | None = None,
    img_filter: str | None = None,
    it_reduce: str = "max",          # 'max' 或 'mean' 聚合同一 (k,h,a) 的多 iters
    save_dir: str | None = None,
    title_suffix: str = "",
    cmap: str = "viridis",           # ← 颜色映射
    vmin: float | None = None,       # ← 固定色标下界（可选）
    vmax: float | None = None,       # ← 固定色标上界（可选）
    vclip: tuple[float,float] | None = None  # ← 百分位裁剪，如 (1,99)
) -> str:
    """
    构造 (k,h)->metric 的 3D 曲面（按 metric 着色并带 colorbar）。
    - 固定 a=a_value；若为 None，自动选出现次数最多的 a（例如 Man≈0.1, Barbara≈0.2）
    - 同一 (k,h,a) 可能有多个 iters：按 it_reduce('max' 或 'mean') 聚合
    - 规则网格 → plot_surface；非规则网格 → plot_trisurf（Delaunay）
    - 返回保存路径
    """
    rows = _load_rows(jsonl_path)
    if img_filter: rows = [r for r in rows if str(r.get("img"))==img_filter]
    rows = [r for r in rows if r.get(metric) is not None and r.get("params")]

    # 选 a
    if a_value is None:
        a_value = _mode([r["params"].get("a") for r in rows if r["params"].get("a") is not None])
    rows = [r for r in rows if r["params"].get("a")==a_value]
    os.makedirs(save_dir or ".", exist_ok=True)
    tag = f"{'all' if not img_filter else os.path.splitext(img_filter)[0]}"

    if not rows:
        outp = os.path.join(save_dir or ".", f"surface_{metric}_kh_a{a_value}_EMPTY.png")
        plt.figure(); plt.axis("off")
        plt.text(0.5,0.6,"NO 3D DATA",ha="center",va="center",fontsize=18)
        plt.text(0.5,0.4,f"a={a_value}",ha="center",va="center",fontsize=10)
        plt.savefig(outp,dpi=160,bbox_inches="tight"); plt.close()
        print(f"[surface] 无可用数据，已生成占位图：{outp}")
        return outp

    # 聚合 (k,h) -> metric（跨 iters）
    bucket = defaultdict(list)
    for r in rows:
        p = r["params"]; k, h = p.get("k"), p.get("h")
        if k is None or h is None: continue
        bucket[(k,h)].append(float(r[metric]))
    agg = (lambda vals: float(np.mean(vals))) if it_reduce=="mean" else (lambda vals: float(np.max(vals)))

    xs = sorted({k for (k,_) in bucket.keys()})
    ys = sorted({h for (_,h) in bucket.keys()})

    # 预计算色标范围
    all_vals = [agg(v) for v in bucket.values()]
    if vclip is not None and len(all_vals) > 2:
        lo, hi = np.percentile(all_vals, vclip[0]), np.percentile(all_vals, vclip[1])
        zmin, zmax = float(lo), float(hi)
    else:
        zmin, zmax = float(np.min(all_vals)), float(np.max(all_vals))
    vmin = zmin if vmin is None else vmin
    vmax = zmax if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 规则网格？
    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for (k,h), vals in bucket.items():
        i = xs.index(k); j = ys.index(h)
        Z[j,i] = agg(vals)

    if np.isfinite(Z).sum() == Z.size:  # 完整规则网格
        Xg, Yg = np.meshgrid(xs, ys)
        facecolors = cmap_obj(norm(Z))
        ax.plot_surface(
            Xg, Yg, Z,
            facecolors=facecolors,
            linewidth=0.4, edgecolor="k",
            antialiased=True, shade=False  # 由 colormap 控制颜色
        )
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj); mappable.set_array(Z)
    else:
        # 非规则网格 → 三角剖分曲面
        pts_x, pts_y, pts_z = [], [], []
        for (k,h), vals in bucket.items():
            pts_x.append(k); pts_y.append(h); pts_z.append(agg(vals))
        tri = mtri.Triangulation(np.array(pts_x), np.array(pts_y))
        arrz = np.array(pts_z)
        ax.plot_trisurf(
            tri, arrz,
            cmap=cmap_obj, norm=norm,
            linewidth=0.4, edgecolor="k",
            antialiased=True
        )
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj); mappable.set_array(arrz)

    ax.set_xlabel('k'); ax.set_ylabel('h'); ax.set_zlabel(metric.upper())
    title = f"3D surface: {metric.upper()} vs (k,h)  a={a_value}"
    if title_suffix: title += f"  {title_suffix}"
    ax.set_title(title)

    cb = fig.colorbar(mappable, ax=ax, shrink=0.65, aspect=14, pad=0.08)
    cb.set_label(metric.upper())

    outp = os.path.join(save_dir or ".", f"surface_{metric}_kh_a{a_value}_{tag}_{it_reduce}.png")
    plt.savefig(outp, dpi=220, bbox_inches="tight"); plt.close()
    print(f"[surface] 保存成功：{outp}  (vmin={vmin:.4f}, vmax={vmax:.4f}, cmap={cmap})")
    return outp
