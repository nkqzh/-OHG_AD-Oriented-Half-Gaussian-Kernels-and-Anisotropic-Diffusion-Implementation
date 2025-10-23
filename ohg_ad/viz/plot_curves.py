from __future__ import annotations
import json, os, matplotlib.pyplot as plt

def plot_curves(jsonl_path: str, xkey: str = "iters", save_dir: str | None = None):
    xs, psnrs, ssims = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            xs.append(row["params"][xkey])
            psnrs.append(row.get("psnr", None))
            ssims.append(row.get("ssim", None))
    plt.figure()
    plt.plot(xs, psnrs, marker='o', label="PSNR")
    plt.plot(xs, ssims, marker='o', label="SSIM")
    plt.xlabel(xkey); plt.ylabel("metric"); plt.legend(); plt.title("Metrics vs " + xkey)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"curves_vs_{xkey}.png"), dpi=150, bbox_inches="tight")
    else:
        plt.show()
