from __future__ import annotations
import argparse
from codes.runners.sweep import run_sweep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_sweep(args.config, args.out)
