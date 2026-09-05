# MatterGen Corrector Residual Distillation Formal256 最终报告

## 结论先行

本轮是对冻结方法 `V2_Frozen_Atomic75_Late30` 的全新 256 paired-seed confirmatory evaluation。最终判定为：

> **BORDERLINE**

速度目标得到稳定复现：严格单 H20、batch size 1、16 个 paired seeds 的平均加速为 **1.3246×**，20,000 次 paired bootstrap 95% CI 为 **[1.2675, 1.3767]**，完整区间高于冻结门槛 1.20×。

质量方面，E-hull、Stable、NUS、RMSD 四项预注册端点通过非劣效界限；RMSD 均值和高分位数还有明显改善。可是 mean per-structure maximum force、Force P95、Force P99 的比率 CI 太宽，分别未能证明满足 1.20、1.30、1.50 的冻结上限。它们的点估计没有超过 margin，paired difference CI 也没有确认有害差异；`force > 1` 与 `force > 2 eV/Å` 的极端比例 guardrail 均通过。因此没有达到 FAIL 条件，但也不能判 PASS，不能声称“质量无损”或“与 C0 等价”。

所有物理质量结果来自 **MatterSim-5M surrogate，不是 DFT**：`DFT_VERIFIED=False`。

## 1. 分支、冻结状态与数据完整性

- Formal branch：`experiment/corrector-residual-distillation-formal256`
- 冻结 V2 起点：`1b8b62284f694b08d778895da345c708204ec2db`
- 冻结方法：`V2_Frozen_Atomic75_Late30`
- Formal protocol 代码提交：`9ea6551b7ff8473fde7da92862e8d467d1351cf8`
- 预注册提交：`e8b7a0c59145e24ef231379f1838d19687dabcd3`
- Frozen manifest SHA256：`888b8aaa38c0e6e49425ee38c5e2a0ef871290ec2bd357dc37ad30e174377277`
- Formal seeds：67000–67255，共 256 个，与 V1/V2 train、validation、Stage-C、Adaptive CFG formal、E3-PCR formal 历史区间不重叠
- 主实验仅含 C0 与 V2，无其他候选
- C0 generation：256/256 success
- V2 generation：256/256 success
- MatterSim relaxation：C0 256/256、V2 256/256 success
- 两种方法内部结构 checksum 均唯一
- 生成失败：0；松弛失败：0

冻结 V2 在 0%–70% progress 使用 residual Adapter，并按 atomic-risk 触发 exact fallback；70%–100% 使用 original MatterGen exact second forward。Atomic-risk threshold 为 `0.8289545722625317`，Adapter epoch 18、2661 parameters、SHA256 `a2fd0f17c27b760958b60c1d1386577883635a34fb2101eb08e38fd8472702c7`。

## 2. 256-seed 主质量结果

以下 delta 均为 `V2 - C0`。Stable、NUS、Novel、Unique 的单位为 proportion。

| Metric | C0 mean | V2 mean | Delta | Relative change | Direction |
|---|---:|---:|---:|---:|---|
| E-hull (eV/atom) | 0.102141 | 0.105879 | +0.003738 | +3.66% | lower |
| Stable | 0.585938 | 0.589844 | +0.003906 | +0.67% | higher |
| NUS | 0.316406 | 0.332031 | +0.015625 | +4.94% | higher |
| Novel | 0.691406 | 0.714844 | +0.023438 | +3.39% | higher |
| Unique | 0.988281 | 0.976562 | -0.011719 | -1.19% | higher |
| RMSD (Å) | 0.080374 | 0.053015 | -0.027359 | -34.04% | lower |
| Per-structure mean force (eV/Å) | 0.134200 | 0.129559 | -0.004641 | -3.46% | lower |
| Per-structure maximum force (eV/Å), mean | 0.284241 | 0.290312 | +0.006071 | +2.14% | lower |

完整分布统计：

| Metric | Method | Median | Std | P5 | P25 | P75 | P95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E-hull | C0 | 0.078648 | 0.099214 | -0.000551 | 0.039114 | 0.146139 | 0.259729 | 0.575332 |
| E-hull | V2 | 0.085336 | 0.108237 | 0.004598 | 0.045705 | 0.138408 | 0.276319 | 1.076208 |
| RMSD | C0 | 0.014359 | 0.223442 | 0.000899 | 0.002372 | 0.046513 | 0.446296 | 1.436865 |
| RMSD | V2 | 0.015843 | 0.139862 | 0.000850 | 0.002653 | 0.045374 | 0.156823 | 1.175922 |
| Mean force | C0 | 0.075017 | 0.208414 | 0.003940 | 0.029056 | 0.168847 | 0.405264 | 2.542824 |
| Mean force | V2 | 0.076230 | 0.164441 | 0.006894 | 0.029745 | 0.178756 | 0.410823 | 1.689141 |
| Maximum force | C0 | 0.135569 | 0.503171 | 0.006728 | 0.054621 | 0.351400 | 0.933156 | 6.189528 |
| Maximum force | V2 | 0.147118 | 0.496591 | 0.010156 | 0.049574 | 0.358604 | 0.983010 | 5.581361 |

## 3. 256-seed paired statistics

20,000 次 paired bootstrap，seed 20260905；区间为 mean paired difference `V2 - C0` 的双侧 95% CI。

| Metric | Mean delta | 95% CI | Interpretation |
|---|---:|---:|---|
| E-hull | +0.003738 | [-0.011487, +0.019798] | 未检测到明确差异；不能据此声称等价 |
| Stable | +0.003906 | [-0.074219, +0.082031] | 未检测到明确差异 |
| NUS | +0.015625 | [-0.054688, +0.089844] | 未检测到明确差异 |
| Novel | +0.023438 | [-0.042969, +0.089844] | 未检测到明确差异 |
| Unique | -0.011719 | [-0.035156, +0.011719] | 未检测到明确差异 |
| RMSD | -0.027359 | [-0.060648, +0.004557] | 点估计改善，CI 跨 0 |
| Mean force | -0.004641 | [-0.034466, +0.024391] | 未检测到明确差异 |
| Maximum force | +0.006071 | [-0.077454, +0.087330] | 未检测到明确差异 |

Wilcoxon/Pratt 与离散指标 McNemar 结果保存在 `paired_statistics.csv`。本报告不把 `p > 0.05` 解释为等价。

## 4. Force tail

这里的 force 是每个生成结构在松弛前的最大原子力，单位 eV/Å。

| Tail statistic | C0 | V2 | V2/C0 | Bootstrap ratio 95% CI |
|---|---:|---:|---:|---:|
| P50 | 0.135569 | 0.147118 | 1.0852 | [0.8721, 1.3781] |
| P75 | 0.351400 | 0.358604 | 1.0205 | [0.7911, 1.3420] |
| P90 | 0.676026 | 0.631737 | 0.9345 | [0.7179, 1.2162] |
| P95 | 0.933156 | 0.983010 | 1.0534 | [0.6550, 1.3898] |
| P99 | 1.654406 | 2.046264 | 1.2369 | [0.3363, 2.8184] |
| Max | 6.189528 | 5.581361 | 0.9017 | not bootstrapped |

极端高力比例：

- `force > 1`：C0 5.078%，V2 4.688%，delta -0.391 pp，95% CI [-3.906, +3.125] pp；通过 +5 pp guardrail。
- `force > 2`：C0 0.781%，V2 1.172%，delta +0.391 pp，95% CI [-1.172, +1.953] pp；通过 +2 pp guardrail。

P95/P99 的点估计均未超过冻结 margin，但 ratio CI 上界过宽，所以只能写“非劣效未建立”，不能写“明确恶化”。

## 5. RMSD tail

| Tail statistic | C0 (Å) | V2 (Å) | V2/C0 | Bootstrap ratio 95% CI |
|---|---:|---:|---:|---:|
| P50 | 0.014359 | 0.015843 | 1.1033 | [0.8281, 1.7809] |
| P75 | 0.046513 | 0.045374 | 0.9755 | [0.7541, 1.3191] |
| P90 | 0.121454 | 0.098294 | 0.8093 | [0.3337, 1.5043] |
| P95 | 0.446296 | 0.156823 | 0.3514 | [0.1644, 1.2897] |
| P99 | 1.225423 | 0.797609 | 0.6509 | [0.2985, 1.1565] |
| Max | 1.436865 | 1.175922 | 0.8184 | not bootstrapped |

V2 的 RMSD 均值、P90、P95、P99 和最大值均较低，说明本轮没有出现由少量 V2 bad seeds 拉高 RMSD 均值的信号；P50 略高且 CI 较宽。

## 6. Non-inferiority

非劣效分析已执行。Margins 在读取任何 formal256 结果前冻结并提交；采用双侧 95% paired-bootstrap interval 作为更保守的界限。

| Endpoint | Scale / margin | Estimate | 95% CI | Pass |
|---|---|---:|---:|---|
| E-hull | absolute harm ≤ 0.025 eV/atom | +0.003738 | [-0.011487, 0.019798] | yes |
| Stable | absolute harm ≤ 0.10 | +0.003906 | [-0.074219, 0.082031] | yes |
| NUS | absolute harm ≤ 0.10 | +0.015625 | [-0.054688, 0.089844] | yes |
| RMSD | absolute harm ≤ 0.02 Å | -0.027359 | [-0.060648, 0.004557] | yes |
| Mean maximum force | ratio ≤ 1.20 | 1.0214 | [0.7734, 1.3529] | no |
| Force P95 | ratio ≤ 1.30 | 1.0534 | [0.6550, 1.3898] | no |
| Force P99 | ratio ≤ 1.50 | 1.2369 | [0.3363, 2.8184] | no |
| Force > 1 proportion | harm ≤ 0.05 | -0.003906 | [-0.039062, 0.031250] | yes |
| Force > 2 proportion | harm ≤ 0.02 | +0.003906 | [-0.011719, 0.019531] | yes |

未通过的三项是统计不确定性造成的“未能证明非劣效”，而不是 CI 已经证明超过 margin。正式 PASS 规则要求 5/5 primary endpoints 和 4/4 force-tail guardrails 全部通过，因此不能判 PASS。

## 7. Strict single-H20 speed benchmark

环境：physical GPU0，NVIDIA H20，batch size 1，同一精度与项目环境，paired seeds 67000–67015。

| Quantity | C0 | V2 |
|---|---:|---:|
| Mean pure generation time/sample | 116.562 s | 88.597 s |
| Samples/hour from mean time | 30.885 | 40.633 |
| Mean peak GPU memory | 296.758 MB | 296.550 MB |

- Mean paired speedup：**1.324642×**
- Median paired speedup：1.318216×
- Speedup std：0.115065
- 20,000 paired-bootstrap 95% CI：**[1.267466, 1.376734]**
- V2 logical score calls：mean 1406.94/sample；C0 固定 2000/sample
- V2 exact second-forward calls：mean 406.94/sample
- Adapter calls：固定 700/sample
- Fallback calls：mean 406.94/sample
- Adapter coverage：**59.306%**
- Forward reduction：**29.653%**
- V2 peak GPU memory maximum：329.323 MB

seed 67009 是速度慢例：coverage 30.3%、forward reduction 15.15%、speedup 1.044×，与更多 exact fallback（697）一致。它被完整保留，没有删除或调参。

## 8. 8×H20 throughput

主运行采用固定 GPU seed shards 与 paired global waves，每卡一个 worker，batch size 1。

- C0 active wall time：6359.04 s（105.98 min），144.93 samples/hour
- V2 active wall time：5617.78 s（93.63 min），164.05 samples/hour
- 两方法顺序运行总 wall time：11976.83 s（199.61 min）

8-GPU throughput 仅是系统吞吐指标，不能替代上面的单 H20 paired algorithmic speedup。

## 9. 最终判定与论文建议

### 最终判定：BORDERLINE

判定依据：

1. 单 H20 平均速度 1.3246×，CI 完整高于 1.20×，速度目标明确成功。
2. E-hull、Stable、NUS、RMSD 的非劣效通过；Novel 点估计提高，Unique 仅小幅降低且 CI 跨 0。
3. Force 的均值差、P95 差和极端比例均未显示明确有害差异；最大值还更低。
4. 但 mean maximum force、P95、P99 三个 ratio CI 未满足预注册非劣效上限，因此正式 PASS 条件不成立。
5. 没有至少两个 primary endpoints 的 CI 在有害方向排除 0；没有 endpoint 明确恶化且超过 margin；多个 tail point estimates 也没有明确超过 margin。因此 FAIL 条件同样不成立。

### 能否正式作为硕士论文最终第二创新

**目前不建议直接以“质量无损加速”正式替换原第二创新。** 可以把 V2 作为有清晰速度收益、主体质量指标有利但 Force tail 非劣效证据不足的候选方法或阶段性结果；若论文第二创新的主张必须是“约 1.3× 且质量达到预注册非劣”，本轮证据尚不够。

### 核心限制

核心问题不是平均 E-hull、Stable、NUS 或 RMSD 恶化，而是 Force 分布重尾导致 P95/P99 ratio CI 很宽。256 seeds 下极端样本仍很少，比例 guardrail 通过，但相对分位数估计不稳定，无法满足严格 PASS 规则。

### 是否推荐 Periodic Exact Anchor V3

**推荐作为下一轮研究方向，但本轮不自动执行。** 建议在后续独立开发实验中加入 Periodic Exact Anchor，以周期性 exact second forward 约束 closed-loop accumulated error；先用 4–8 seed smoke、16/32 seed 初验，再决定是否进入 64-seed 独立验证。不得使用本轮 formal256 seeds 调参。

## 10. 真实执行命令

项目环境、权重、缓存和输出均位于 `/mnt/lis-wam-data/dxl/mattergen_v1`。

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source .venv/bin/activate
export TMPDIR=/mnt/lis-wam-data/dxl/mattergen_v1/.tmp
export XDG_CACHE_HOME=/mnt/lis-wam-data/dxl/mattergen_v1/.cache
export HF_HOME=/mnt/lis-wam-data/dxl/mattergen_v1/.cache/huggingface
export TORCH_HOME=/mnt/lis-wam-data/dxl/mattergen_v1/.cache/torch

python -m research.corrector_distillation.run_formal256_generation --mode main
python -m research.corrector_distillation.aggregate_formal256_generation --mode main
python -m research.corrector_distillation.run_formal256_generation --mode single-h20
python -m research.corrector_distillation.aggregate_formal256_generation --mode single-h20
python -m research.corrector_distillation.formal256_speed_statistics

CUDA_VISIBLE_DEVICES=0 python -m research.corrector_distillation.relax_many_individual \
  --structures-path experiments/corrector_residual_distillation_formal256/structures/C0_generated.extxyz \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --output-dir experiments/corrector_residual_distillation_formal256/relaxation/precomputed/C0 \
  --device cuda --cuda-3x3-det-workaround

CUDA_VISIBLE_DEVICES=1 python -m research.corrector_distillation.relax_many_individual \
  --structures-path experiments/corrector_residual_distillation_formal256/structures/V2_Frozen_Atomic75_Late30_generated.extxyz \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --output-dir experiments/corrector_residual_distillation_formal256/relaxation/precomputed/V2_Frozen_Atomic75_Late30 \
  --device cuda --cuda-3x3-det-workaround

python -m research.corrector_distillation.evaluate_quality --method C0 \
  --structures-path experiments/corrector_residual_distillation_formal256/structures/C0_generated.extxyz \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --reference-path data-release/alex-mp/reference_MP2020correction.gz \
  --output-dir experiments/corrector_residual_distillation_formal256/quality/C0 \
  --temporary-dir experiments/corrector_residual_distillation_formal256/runtime_tmp \
  --device cpu \
  --precomputed-relaxation-dir experiments/corrector_residual_distillation_formal256/relaxation/precomputed/C0

python -m research.corrector_distillation.evaluate_quality --method V2_Frozen_Atomic75_Late30 \
  --structures-path experiments/corrector_residual_distillation_formal256/structures/V2_Frozen_Atomic75_Late30_generated.extxyz \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --reference-path data-release/alex-mp/reference_MP2020correction.gz \
  --output-dir experiments/corrector_residual_distillation_formal256/quality/V2_Frozen_Atomic75_Late30 \
  --temporary-dir experiments/corrector_residual_distillation_formal256/runtime_tmp \
  --device cpu \
  --precomputed-relaxation-dir experiments/corrector_residual_distillation_formal256/relaxation/precomputed/V2_Frozen_Atomic75_Late30

python -m research.corrector_distillation.formal256_statistics
```

## 11. 结果文件

- `generation_per_seed.csv`：512 次生成记录
- `formal256_generation_audit.json`：生成完整性结果
- `quality_per_seed.csv`：512 行逐方法质量记录
- `paired_per_seed.csv`：256 行严格配对记录
- `quality_summary.csv`：完整分布统计
- `paired_statistics.csv`：paired bootstrap、Wilcoxon/Pratt、McNemar
- `force_tail_analysis.csv`、`rmsd_tail_analysis.csv`：tail 分析
- `noninferiority_results.csv`：冻结非劣效判定
- `single_h20_benchmark.csv`、`single_h20_per_seed.csv`、`single_h20_metrics.csv`：单卡速度
- `eight_h20_throughput.json`：8 卡系统吞吐

大结构、轨迹、日志、缓存和权重不进入 Git，但全部保存在项目目录。最终 Git HEAD 由包含本报告的提交确定，并在最终交付信息中记录。
