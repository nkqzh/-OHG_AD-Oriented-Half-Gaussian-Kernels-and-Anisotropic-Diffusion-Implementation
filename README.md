# OHG-AD: Oriented Half-Gaussian Anisotropic Diffusion

**中文/English 双语** · Reproducible, modular, experiment-friendly.

- 核心算法：`ohg_ad.algorithms.ohgad_new`（与论文一致、可复现最佳指标）
- 模块化核心：`ohg_ad.core.*`（核生成、方向估计、二阶导/扩散、指标等）
- 实验工具：`scripts/run_single.py`（单次/自动最佳 iters），`scripts/run_sweep.py`（批量网格/细化）
- 可视化：`scripts/make_plots.py`（曲线/热图），结果保存在 `results/<exp>/`

## Quickstart

```bash
pip install -e .
# 单次复现实验（Man, σ=10）
python scripts/run_single.py --input data/USC_SIPI/man.tiff --gt data/USC_SIPI/man.tiff   --sigma 10 --dtheta 2 --dt 0.05 --iters 8   --mu 5 --lam 1 --k 0.3 --h 0.1 --a 0.1   --out results/man_sigma10_run

# 自动选择最佳 iters（有GT时以 PSNR 为准）
python scripts/run_single.py --input data/USC_SIPI/man.tiff --gt data/USC_SIPI/man.tiff   --sigma 10 --auto-iters --iters-max 30 --patience 3 --dt 0.05   --mu 5 --lam 1 --k 0.3 --h 0.1 --a 0.1   --out results/man_sigma10_autoiters

# 批量搜索（读取 YAML）
python scripts/run_sweep.py --config configs/sweeps/man_sigma10.yaml --out results/sweeps/man_sigma10
# 画图
python scripts/make_plots.py --log results/sweeps/man_sigma10/metrics.jsonl --plots curves,heatmap --x iters --grid k,h
```

## 数据/测试集
- 官方演示用：USC SIPI 标准图（`data/USC_SIPI/`）。我们不直接包含图片，但提供路径约定。
- 你可以把自己的图像放到 `data/` 并在命令行指定。

## 复现要点
- 反射边界（reflect padding）
- 主扩散项：沿 **两条直化方向** 的二阶导之和（`I_{ρ1ρ1}+I_{ρ2ρ2}`）+ `I_{ηη}`
- 连续控制函数 + 正定扩散；步长建议 `dt≤0.05~0.1`

## 目录结构
```
ohg_ad/
  algorithms/         # 高层算法封装（供用户直接调用）
    ohgad_new.py
  core/               # 核心模块（小而清晰，复用性强）
    params.py         # 参数/配置数据类
    kernels.py        # 半高斯半核生成与缓存
    orientation.py    # Q-map、θ1/θ2、γ/ρ1/ρ2
    operators.py      # 反射卷积、二阶导运算
    diffusion.py      # 扩散步、完整去噪流程
    metrics.py        # PSNR/SSIM 包装
  runners/
    single.py         # 单次/自动 iters 运行器
    sweep.py          # 网格/细化搜索
  viz/
    plot_curves.py    # 曲线绘图
    plot_heatmap.py   # 网格热图
configs/
  sweeps/*.yaml       # 搜索空间定义样例
scripts/
  run_single.py       # CLI: 单次/auto-iters
  run_sweep.py        # CLI: 搜索
  make_plots.py       # CLI: 画图
data/                 # 你的图片放这里（不随仓库提供）
results/              # 输出目录（默认在 .gitignore）
```

## 许可证
MIT（可按需要更改）。
