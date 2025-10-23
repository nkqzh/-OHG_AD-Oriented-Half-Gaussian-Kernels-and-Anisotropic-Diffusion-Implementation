from __future__ import annotations
import argparse, os
from ohg_ad.viz.plot_curves import plot_curves
from ohg_ad.viz.plot_heatmap import plot_heatmap

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="metrics.jsonl file")
    ap.add_argument("--plots", default="curves", help="curves,heatmap")
    ap.add_argument("--x", default="iters")
    ap.add_argument("--grid", default="k,h", help="e.g., k,h")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(args.out or ".", exist_ok=True)
    if "curves" in args.plots:
        plot_curves(args.log, xkey=args.x, save_dir=args.out)
    if "heatmap" in args.plots:
        gx, gy = args.grid.split(",")
        plot_heatmap(args.log, xgrid=gx, ygrid=gy, metric="psnr", save_dir=args.out)
