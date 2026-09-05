# Corrector residual distillation v2

本实验只解决 V1 暴露出的 closed-loop distribution shift：在 V1 Adapter+Fallback
实际访问的状态上，用冻结的原始 MatterGen second score 生成 DAgger 监督，再结合
late-stage exact schedule 与可解释的 field-aware risk 决策。V1 正式产物及
63000–63031 永久只读；本轮不做 A0 组合。

## 固定设计

- 项目、虚拟环境、权重、缓存与输出全部位于
  `/mnt/lis-wam-data/dxl/mattergen_v1`。
- DAgger train：64000–64015。
- V2 validation：65000–65015。
- 最终独立测试：66000–66063；只允许在候选配置和 checkpoint 已冻结、已提交后运行。
- `dft_mag_density=0.1`，constant CFG 2.0，1000 diffusion steps，
  1 corrector step，batch size 1，deterministic。
- MatterGen 与 MatterSim 权重保持冻结。Adapter 仍为 2,661 参数量级。
- 所有新增 sampler 功能默认关闭。

## progress 的实际方向

`PredictorCorrector` 的 timestep 从 `T` 单调下降到 `eps`，循环索引从 0
增至 999；`progress=i/(N-1)`。因此 `progress=0` 是高噪声生成起点，
`progress=1` 是低噪声最终阶段。所谓 last 20/30/40% exact 分别对应
`late_exact_start=0.8/0.7/0.6`。

## 选择规则

Stage-B 首先排除严重结构异常和 max-force outlier，随后依次比较 RMSD、mean
force、E-hull/Stable/NUS，最后比较 speedup。只有 speedup 至少 1.15× 的候选
才可冻结，优先要求 1.20×。Stage-C 不再调 checkpoint、risk threshold 或 schedule。

## 复现入口

```bash
cd /mnt/lis-wam-data/dxl/mattergen_v1
source .venv/bin/activate
export TMPDIR=/mnt/lis-wam-data/dxl/mattergen_v1/.tmp
PYTHONDONTWRITEBYTECODE=1 python -m pytest -s \
  mattergen/diffusion/tests/test_residual_distillation.py -q
```

大型 DAgger tensor、teacher tensor、生成结构、松弛轨迹、日志和模型权重由
`.gitignore` 排除；Git 只保存源码、配置、校验清单、小型 CSV/JSON 和报告。

## 最终结果

冻结方法为 `V2_Frozen_Atomic75_Late30`：前 70% timestep 使用 2,661 参数
DAgger Adapter，并仅用 atomic risk 决定是否 exact fallback；最后 30% 永远
exact。66000–66063 的三臂独立测试共 192/192 成功。

- 严格单 H20、batch size 1：V2 87.384 s/sample，41.198 samples/h，配对
  speedup 1.362×（95% bootstrap CI 1.321–1.406），forward reduction 30.71%。
- 8×H20 seed-level throughput：C0 152.166、V1 170.913、V2 178.295 samples/h。
- MatterSim-5M：C0/V2 的 E-hull 为 0.11251/0.13251 eV/atom，Stable
  56.25%/50.00%，NUS 31.25%/26.56%，RMSD 0.06803/0.08093 Å，force mean
  0.3105/0.3452 eV/Å，max force 2.1948/2.6163 eV/Å。
- 主质量、RMSD 和 force 配对差值的 95% CI 均跨 0；严重的约 2× RMSD
  退化未复现，但点估计没有完全追平 C0。结论是 formal256 最低门槛边界通过，
  只能以冻结配置做确认实验，不能继续用结果调参。

完整结论见 `final_report.md`，机器可读准入判定见
`formal256_admission.json`，生成与质量审计见 `stage_c_generation_audit.json` 和
`stage_c_quality_audit.json`。

## 最终评估复现

H20/CUDA 12.1 上 MatterSim 的 batched `torch.linalg.det(cell)` 会间歇报 driver
invalid argument。正式结果使用显式开关，对 CUDA 3×3 cell 使用可微解析行列式，
并逐结构弛豫；64 个随机 float64 矩阵相对 CPU `torch.linalg.det` 的最大绝对误差
为 `8.88e-16`，梯度均有限。每个方法先运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m research.corrector_distillation.relax_many_individual \
  --structures-path <METHOD_generated.extxyz> \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --output-dir <PRECOMPUTED_DIR> --device cuda \
  --cuda-3x3-det-workaround
```

再在 CPU 上计算官方指标：

```bash
CUDA_VISIBLE_DEVICES='' python -m research.corrector_distillation.evaluate_quality \
  --method <METHOD> --structures-path <METHOD_generated.extxyz> \
  --potential-path checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth \
  --reference-lmdb-path \
    experiments/corrector_residual_distillation_v1/relaxation/tmp/tmp5rqvz9t3/reference_MP2020correction \
  --reference-is-ordered --device cpu --output-dir <QUALITY_DIR> \
  --precomputed-relaxation-dir <PRECOMPUTED_DIR>
```
