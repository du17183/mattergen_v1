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
