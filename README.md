# 创新点二正式验证：Q3 E3-PCR

> 当前分支：`feature/q3-e3-pcr-formal256`

本分支冻结第二创新点 **Q3 E3-PCR Learned-Gated Post-generation Refinement** 的独立 256-seed 正式结果。

## 冻结结论

```text
FINAL_STATE=E3_G_FORMAL_CONFIRMED
FROZEN_COMMIT=0275cbf08ed3c6321cea7d06f7a3a8edb83b7483
FORMAL_SEEDS=40000–40255
N=256
```

| 指标 | E3-G 相对 C0 |
|---|---:|
| 预松弛最大力 | **-23.28%** |
| Harm rate | 18.359% |
| 原子种类 | 不修改 |
| MatterGen 主干 | 不重新训练 |
| 采样轨迹 | 不修改 |

## 方法

Q3 在生成结束后执行受控的等变局部修正。一个 129 参数门控器根据 14 个结构和力学特征判断是否接受 E3-PCR 候选；trust region 和 fallback 负责限制不安全更新。

## 工作过程

1. 从六个后生成质量模块中快速筛选候选。
2. 在新 32-seed 池中确认 Q3 的力改善和质量安全性。
3. 进行 64-seed 冻结验证，确认效果，但发现“学习门控优于 always-on”的机制证据不足。
4. 使用完全独立的 256 seeds 做最终检验。
5. 冻结模型、阈值、统计口径和正式 commit。

## 如何理解结果

- 正向结论是降低预松弛最大力，而不是证明 DFT 稳定性提升。
- 该模块不改变生成采样轨迹，因此可以与 Adaptive CFG 串联。
- 论文应如实说明 64-seed 阶段的门控机制消融不充分；正式结论以独立 256-seed 的整体模块效果为准。

## 证据入口

正式报告和统计文件位于：

- [`reports/q3_e3_pcr/formal256/`](reports/q3_e3_pcr/formal256/)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

## 算法与执行链

Q3 不重新运行或修改扩散轨迹，而是在 C0 结构生成后执行：

```text
C0 structure
→ CHGNet 预测能量、力和局部环境
→ 提取 14 维结构/力学风险特征
→ 129 参数 QualityNetwork 输出 gate 分数
→ E3 等变 position proposal
→ trust-region、最短距离和有限性检查
→ gate 接受则输出 E3-G，否则 exact fallback 到 C0
→ 独立 MatterSim relaxation
```

原子种类和原子顺序始终不变；position proposal 使用力方向并限制 wrapped displacement。正式实验同时保留 C0、Always-on E3-A 和 Learned-Gated E3-G，用于区分“refiner 有效”与“gate 机制有效”。

## 实现位置

| 文件 | 作用 |
|---|---|
| [`research/postgen_fastgate/refiner_eval.py`](research/postgen_fastgate/refiner_eval.py) | CHGNet 推理、position proposal、安全检查、gate 训练和结构修正 |
| [`research/postgen_fastgate/features.py`](research/postgen_fastgate/features.py) | 14 维候选特征提取 |
| [`research/postgen_fastgate/model.py`](research/postgen_fastgate/model.py) | 129 参数 `QualityNetwork` |
| [`research/q3_formal256.py`](research/q3_formal256.py) | 256-seed 冻结契约、生成、修正、relax、统计和最终判定 |
| [`configs/q3_e3_pcr_frozen64.json`](configs/q3_e3_pcr_frozen64.json) | 从 64-seed 阶段冻结的唯一配置 |
| [`tests/test_q3_formal256.py`](tests/test_q3_formal256.py) | 正式 seed、manifest、质量门槛和统计测试 |

## 数据与实现效果

正式集合为 `40000–40255`，与 gate 训练范围 `20000–20063` 无交集。每个 seed 的 C0 只生成一次，E3-A 与 E3-G 都由同一个 C0 输入派生；总计完成 256 个生成和 768 个 MatterSim relaxation。

建议按以下顺序检查数据：

- [冻结与 seed 审计](reports/q3_e3_pcr/formal256/formal_seed_audit.json)
- [逐结构主指标](reports/q3_e3_pcr/formal256/Q3_E3_PCR/official_metrics_per_structure.csv)
- [正式配对统计](reports/q3_e3_pcr/formal256/formal_paired_statistics.csv)
- [门控机制报告](reports/q3_e3_pcr/formal256/gate_mechanism_report.md)
- [完整最终报告](reports/q3_e3_pcr/formal256/final_report.md)

## 复现入口

服务器环境中运行：

```bash
bash tools/q3_e3_pcr_formal256/status.sh
bash tools/q3_e3_pcr_formal256/run.sh
```

专项测试：

```bash
python -m pytest tests/test_q3_formal256.py tests/test_q3_frozen64.py -q
```

runner 使用 `/data/dxl` 的环境、权重和结果目录，并支持 progress/resume；在其他机器上运行前必须先调整这些服务器路径。
