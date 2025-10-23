# -*- coding: utf-8 -*-
"""
聚合 sweep 结果：读取一个或多个 metrics.jsonl（或 metrics.*.jsonl），导出：
- summary.csv：逐行汇总
- topk.csv   ：按 score/psnr/ssim 排序（默认按 score，如果未指定则按 psnr）
- best_overall.json / best_by_image.json
可选：定义复合分数 score = alpha * SSIM + beta * PSNR_norm
PSNR_norm = (PSNR - min_psnr) / (max_psnr - min_psnr + 1e-9)
"""
import argparse, os, json, glob, csv, math
from collections import defaultdict

def load_rows(paths):
    rows = []
    for p in paths:
        for fp in glob.glob(p):
            if not os.path.isfile(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        row = json.loads(line)
                        row["_log"] = fp
                        rows.append(row)
                    except json.JSONDecodeError:
                        pass
    return rows

def compute_score(rows, alpha=0.7, beta=0.3, use_score=True):
    # 动态归一化 PSNR（按当前集合 min/max）
    ps = [r.get("psnr") for r in rows if r.get("psnr") is not None]
    if not ps:
        for r in rows: r["_score"] = None
        return
    mn, mx = min(ps), max(ps)
    for r in rows:
        psnr = r.get("psnr"); ssim = r.get("ssim")
        if psnr is None or ssim is None or not use_score:
            r["_score"] = psnr  # 默认退化为 psnr 排序
        else:
            psnr_norm = (psnr - mn) / (mx - mn + 1e-9)
            r["_score"] = alpha * ssim + beta * psnr_norm

def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 扁平化参数
    out_rows = []
    for r in rows:
        p = r.get("params", {})
        out = {
            "img": r.get("img"),
            "sigma": r.get("sigma"),
            "k": p.get("k"), "h": p.get("h"), "a": p.get("a"),
            "mu": p.get("mu"), "lam": p.get("lam"),
            "dtheta": p.get("dtheta_deg") or p.get("dtheta"),
            "dt": p.get("dt"), "iters": p.get("iters"),
            "psnr": r.get("psnr"), "ssim": r.get("ssim"),
            "score": r.get("_score"),
            "log_path": r.get("_log"),
        }
        out_rows.append(out)
    # 写 CSV
    hdr = ["img","sigma","k","h","a","mu","lam","dtheta","dt","iters","psnr","ssim","score","log_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader(); w.writerows(out_rows)

def pick_best(rows, key="score"):
    key = "_score" if key=="score" else key
    best = None
    for r in rows:
        v = r.get(key)
        if key=="_score": v = r.get("_score")
        if v is None:
            continue
        if (best is None) or (v > best["value"]):
            best = {"value": float(v), "row": r}
    return best

def group_by_image(rows):
    g = defaultdict(list)
    for r in rows:
        g[r.get("img")].append(r)
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="一个或多个 jsonl 路径/通配符，比如 results/sweeps/*/metrics*.jsonl")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--alpha", type=float, default=0.7, help="score = α*SSIM + β*PSNR_norm 中 α")
    ap.add_argument("--beta", type=float, default=0.3, help="score = α*SSIM + β*PSNR_norm 中 β")
    ap.add_argument("--order", choices=["score","psnr","ssim"], default="score", help="topk 排序依据")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--no-score", action="store_true", help="不计算复合分，只按 psnr 选优")
    args = ap.parse_args()

    rows = load_rows(args.logs)
    if not rows:
        raise SystemExit("No rows loaded from: " + " ".join(args.logs))
    compute_score(rows, alpha=args.alpha, beta=args.beta, use_score=(not args.no_score))

    os.makedirs(args.out, exist_ok=True)
    # 1) 全量 CSV
    save_csv(rows, os.path.join(args.out, "summary.csv"))

    # 2) topk
    key = {"score":"_score","psnr":"psnr","ssim":"ssim"}[args.order]
    sorted_rows = sorted([r for r in rows if r.get(key) is not None],
                         key=lambda r: r.get(key) if key!=" _score" else r.get("_score"),
                         reverse=True)
    save_csv(sorted_rows[:args.topk], os.path.join(args.out, "topk.csv"))

    # 3) best overall
    best_overall = pick_best(rows, key=args.order)
    with open(os.path.join(args.out, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_overall, f, ensure_ascii=False, indent=2)

    # 4) best by image
    best_by_img = {}
    for img, sub in group_by_image(rows).items():
        b = pick_best(sub, key=args.order)
        if b: best_by_img[img] = b
    with open(os.path.join(args.out, "best_by_image.json"), "w", encoding="utf-8") as f:
        json.dump(best_by_img, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "out": args.out,
        "files": ["summary.csv","topk.csv","best_overall.json","best_by_image.json"],
        "order": args.order, "alpha": args.alpha, "beta": args.beta
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
