# CG-TDR Phase 0

> 当前分支：`feature/cg-tdr`

本分支验证 Convergence-Guided Teacher Distillation Residual 路线能否学习安全、有效的采样修正。

## 最终结论

```text
GATE_V2_IMPLEMENTATION_VALID=True
EIGHT_SEED_GO=False
CG_TDR_MVP_NO_GO=True
THIRTY_TWO_SEED_STARTED=False
FORMAL_SEEDS_STARTED=False
```

## 主要结果

- V1 学到的 residual 比零残差更差，position cosine 为 `-0.044`。
- T1 的 RMSD 中位变化为 `+18.30%`。
- T2 的 RMSD 均值变化为 `+10.84%`，最大力变化为 `+20.02%`。
- V2P 虽满足安全约束，但输出接近零修正，没有形成正向门控。
- V2C 的 RMSD 中位变化仍为 `+18.29%`。
- 目标专项测试 `24/24` 通过。

## 工作过程

1. 复用已完成的 Teacher 数据和映射。
2. 训练 V1 residual 模型并检查方向一致性。
3. 根据失败原因实现带质量约束的 Gate V2。
4. 在 8-seed smoke 中比较 position-only 与联合候选。
5. 因核心 RMSD/force 指标恶化，按 gate 规则停止，未进入 32/64 seeds。

## 如何理解

工程链路有效，失败来自学习到的修正方向缺乏跨样本泛化，而不是 runner 或 Teacher cache 未运行。该路线已经完成最小可证伪验证，不应通过延长训练自动翻转结论。

## 证据

- [最终评估报告](research/cg_tdr/artifacts/eval/cg_tdr_eval_final.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 方法实现

CG-TDR 先从 A0 与 Teacher 数据构造目标 residual，再训练轻量模型预测采样修正；Gate V2 使用质量标签判断修正是否安全：

```text
Teacher/A0 paired states
→ 构建 residual 与安全标签
→ 训练 residual model / gate
→ 采样时预测 position 或联合修正
→ gate 通过才应用，否则 identity fallback
→ MatterSim 独立评价
```

## 代码位置

| 文件 | 内容 |
|---|---|
| [`model.py`](research/cg_tdr/model.py) | residual 与 gate 网络 |
| [`train.py`](research/cg_tdr/train.py) | V1 训练 |
| [`train_gate_v2.py`](research/cg_tdr/train_gate_v2.py) | Gate V2 训练 |
| [`sampler.py`](research/cg_tdr/sampler.py) | 采样期修正与 fallback |
| [`experiment_generation_v2.py`](research/cg_tdr/experiment_generation_v2.py) | V2 生成评估 |
| [`tests/test_gate_v2.py`](research/cg_tdr/tests/test_gate_v2.py) | gate 标签、阈值和回退测试 |

## 数据与失败证据

- [Teacher/代码映射](research/cg_tdr/artifacts/code_map.md)
- [Residual 学习诊断](research/cg_tdr/artifacts/eval/residual_learning_diagnostics.md)
- [逐结构 residual 统计](research/cg_tdr/artifacts/eval/residual_learning_per_structure.csv)
- [V1 报告](research/cg_tdr/artifacts/eval/v1/v1_report.md)
- [V2 报告](research/cg_tdr/artifacts/eval/v2/v2_report.md)
- [最终报告](research/cg_tdr/artifacts/eval/cg_tdr_eval_final.md)

## 复现入口

```bash
bash research/cg_tdr/scripts/status_eval.sh
python -m pytest research/cg_tdr/tests/test_model.py research/cg_tdr/tests/test_gate_v2.py -q
```

完整评估 runner 为 `research/cg_tdr/scripts/run_eval.sh`。V2P 的安全 flat output 与 V2C 的 RMSD 恶化均保留在逐结构表中，因此 No-Go 可被独立复核。
