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
