# --- bootstrap: allow running without install ---
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------

import argparse, os
from codes.viz.plot_curves import plot_curves
from codes.viz.plot_heatmap import plot_heatmap
from codes.viz.plot_3d_points import plot_3d_kha_metric
from codes.viz.plot_3d_surface import plot_surface_kh_metric

def _normalize_plots_arg(s: str):
    tokens = [t.strip().lower() for t in s.replace(";", ",").split(",") if t.strip()]
    out = []
    for t in tokens:
        if t in ("3", "3d", "3-d", "three_d", "three-d"):
            t = "3d"
        elif t in ("curve", "curves"):
            t = "curves"
        elif t in ("heatmap", "heatmaps", "hm"):
            t = "heatmap"
        elif t in ("surface", "surf", "mesh"):
            t = "surface"
        if t not in out:
            out.append(t)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--plots", default="curves,heatmap,3d,surface")
    ap.add_argument("--metrics", default="psnr,ssim")
    ap.add_argument("--x", default="iters")
    ap.add_argument("--grid", default="k,h")
    ap.add_argument("--reduce", default="max", choices=["max","mean"])
    ap.add_argument("--img", default=None, help="仅绘制某张图（可选）")
    ap.add_argument("--a",   type=float, default=None, help="3D曲面固定的 a 值（缺省自动选择出现最多的 a）")
    ap.add_argument("--it-reduce", default="max", choices=["max","mean"], help="同 (k,h,a) 多 iters 的聚合方式")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cmap", default="viridis", help="曲面 colormap，如 viridis / plasma / inferno / turbo")
    ap.add_argument("--vmin", type=float, default=None, help="颜色下界（默认数据最小值/或经 vclip 裁剪）")
    ap.add_argument("--vmax", type=float, default=None, help="颜色上界（默认数据最大值/或经 vclip 裁剪）")
    ap.add_argument("--vclip", default=None, help="按百分位裁剪色域，例如 '1,99'")
    args = ap.parse_args()

    vclip_tuple = None
    if args.vclip:
        try:
            lo, hi = args.vclip.split(",")
            vclip_tuple = (float(lo), float(hi))
        except Exception:
            vclip_tuple = None

    os.makedirs(args.out or ".", exist_ok=True)
    plats = _normalize_plots_arg(args.plots)
    mets  = [m.strip() for m in args.metrics.split(",") if m.strip()]
    gx, gy = [s.strip() for s in args.grid.split(",")]

    if args.verbose:
        print(f"[PLOTS] log={args.log}")
        print(f"[PLOTS] out={args.out or os.getcwd()}")
        print(f"[PLOTS] plots={plats} metrics={mets} grid=({gx},{gy}) reduce={args.reduce} img={args.img} a={args.a} it_reduce={args.it_reduce}")

    if "curves" in plats:
        for m in mets:
            if args.verbose: print(f"[curves] metric={m} ...")
            plot_curves(args.log, xkey=args.x, metric=m, save_dir=args.out)
            if args.verbose: print(f"[curves] metric={m} done.")

    if "heatmap" in plats:
        for m in mets:
            if args.verbose: print(f"[heatmap] metric={m} grid=({gx},{gy}) ...")
            plot_heatmap(args.log, xgrid=gx, ygrid=gy, metric=m, save_dir=args.out, reduce=args.reduce)
            if args.verbose: print(f"[heatmap] metric={m} done.")

    if "3d" in plats:
        if args.verbose: print(f"[3D] start ...")
        p = plot_3d_kha_metric(args.log, metric="ssim", save_dir=args.out,
                               img_filter=args.img, reduce=args.reduce, show=args.show)
        if args.verbose: print(f"[3D] output => {p}")

    if "surface" in plats:
        if args.verbose: print(f"[surface] start ...")
        p = plot_surface_kh_metric(
            args.log, metric="ssim", a_value=args.a, img_filter=args.img,
            it_reduce=args.it_reduce, save_dir=args.out, title_suffix="",
            cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, vclip=vclip_tuple
        )
        if args.verbose: print(f"[surface] output => {p}")
