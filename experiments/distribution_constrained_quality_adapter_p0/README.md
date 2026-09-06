# Distribution-Constrained Quality Adapter — P0

本目录包含 MatterGen 第二创新新路线的最小真实验证。实现与真实结果对应提交为 `b0c22b5`，分支为 `experiment/distribution-constrained-quality-adapter-p0`。

## P0 结论

静态 Quality Adapter 可以稳定接入真实 `dft_mag_density` MatterGen checkpoint，并在 8 个全新配对 seeds 上给出值得进入 P1 的正向 signal。M2 相对 C0：E-hull `0.1607 → 0.0850 eV/atom`，Stable `37.5% → 75.0%`，NUS `37.5% → 62.5%`，RMSD `0.1680 → 0.0597`，全原子初始力模长均值 `0.2230 → 0.1461 eV/Å`。小样本不作显著性或普适性结论。

## 数据与模型

- clean-x0：官方 MatterGen Alex-MP `alex_mp_20/train.csv`，从 597,339 条合格训练记录固定 reservoir sampling 128 条；96 train / 32 validation。
- CHGNet：0.3.0，权重位于项目 `.cache/models/chgnet/0.3.0/`。
- MatterGen：`checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt`。
- MatterSim：`checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth`。
- 正式 reference：`data-release/alex-mp/reference_MP2020correction.gz`。
- Python 环境：项目内 `.venv`；数据、权重、缓存和实验产物均在 `/mnt/lis-wam-data/dxl/mattergen_v1`。

## 方法

- C0：原始 MatterGen，不挂载 Adapter。
- M1：同一静态 Adapter，unweighted mixed-field diffusion fine-tuning。
- M2：CHGNet force-max 分段权重（低 30%=`1.25`、中 40%=`1.0`、高 30%=`0.75`），50% replay 恢复权重 1，加 `lambda=0.05` 的同 noisy-state output anchor。
- Adapter：在 GemNet interaction block 1、2 后挂载 `LN(512) → Linear(64) → SiLU → Linear(512)` 残差支路，共 134,272 个可训练参数；最终投影零初始化。
- P0 按预注册范围只训练 static 版本，未启用 stage gate，未做 M3/P1/P2/formal256。

## 主要文件

- `data_summary.json`：CHGNet 标签分布与 signal 检查。
- `quality_labels.csv`：128 条 clean-x0 独立质量标签。
- `training_summary.csv`：M1/M2 真实训练汇总。
- `generation_results.csv`：24 个真实 1000-step 生成结果。
- `quality_results.csv`：MatterSim 与官方 MatterGen 指标。
- `final_report.md`：完整 29 项结论与限制。
- `mattergen/quality_adapter.py`：Adapter、质量加权损失与 output anchor。

大型运行产物保留在 `data/`、`checkpoints/`、`generation/`、`relaxation/`、`quality/`、`logs/`，由 `.gitignore` 排除；训练后的 M1/M2 Adapter checkpoint 各约 530 KiB，仍位于本项目实验目录内。

## 从项目内现有资产复现

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source .venv/bin/activate

python experiments/distribution_constrained_quality_adapter_p0/prepare_data.py \
  --zip-path .cache/datasets/alex_mp_20.zip \
  --output-root experiments/distribution_constrained_quality_adapter_p0/data \
  --count 128 --train-count 96 --seed 20260906

CUDA_VISIBLE_DEVICES=0 python experiments/distribution_constrained_quality_adapter_p0/label_quality.py \
  --data-root experiments/distribution_constrained_quality_adapter_p0/data \
  --model-path .cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --output-csv experiments/distribution_constrained_quality_adapter_p0/quality_labels.csv \
  --output-summary experiments/distribution_constrained_quality_adapter_p0/data_summary.json \
  --batch-size 16
```

M1/M2 使用同一训练命令，仅替换 `--method`、GPU 和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/distribution_constrained_quality_adapter_p0/train_adapter.py \
  --method M1 \
  --model-root checkpoints/official/hf_mattergen/checkpoints/dft_mag_density \
  --data-root experiments/distribution_constrained_quality_adapter_p0/data \
  --labels experiments/distribution_constrained_quality_adapter_p0/quality_labels.csv \
  --output-dir experiments/distribution_constrained_quality_adapter_p0/checkpoints/M1 \
  --steps 200 --batch-size 16 --learning-rate 3e-4 \
  --anchor-lambda 0.05 --replay-fraction 0.5 \
  --validation-every 20 --seed 20260906
```

生成与评价：

```bash
python experiments/distribution_constrained_quality_adapter_p0/run_generation.py
python experiments/distribution_constrained_quality_adapter_p0/aggregate_generation.py

CUDA_VISIBLE_DEVICES=0 python -m research.corrector_distillation.relax_many_individual \
  --structures-path experiments/distribution_constrained_quality_adapter_p0/structures/C0_generated.extxyz \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --output-dir experiments/distribution_constrained_quality_adapter_p0/relaxation/C0 \
  --device cuda --fmax 0.05 --cuda-3x3-det-workaround

# M1/M2 使用同一命令并可分别放到 GPU 1/2。
python experiments/distribution_constrained_quality_adapter_p0/run_quality_metrics.py
python experiments/distribution_constrained_quality_adapter_p0/summarize_results.py
```

生成和评价脚本默认拒绝覆盖已存在的完整/不完整输出；复现应使用新的空实验目录或先人工归档旧结果。
