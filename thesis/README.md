# MatterGen 论文规划与可复现图表

本目录是冻结实验归档 `thesis_archive/` 的论文写作层，不执行生成、训练、松弛、DFT 或任何 GPU 任务。两项主创新可以并行表述：

1. **Multi-field Residual-driven Online Adaptive CFG**：在采样阶段根据 cell、position、atom 三字段条件残差在线调节 CFG，完整保留 Predictor/Corrector。
2. **Learned-Gated Safe-Bounded Equivariant Post-Generation Crystal Refiner (E3-PCR)**：在后生成阶段以 129 参数 Gate 决定是否执行 5 步有界位置精修，并在 Gate-off 或拒绝时精确回退。

## 一键复现

```bash
python -m venv .venv-thesis
. .venv-thesis/bin/activate
python -m pip install -r thesis_archive/requirements-analysis.txt
python thesis/scripts/generate_all.py
python thesis/scripts/validate_outputs.py
```

所有代码均使用仓库相对路径并可在普通 CPU 笔记本运行。图表的唯一科学输入是 `thesis_archive/`；不从汇总均值伪造逐 seed 数据。

## 关键入口

- [冻结论文结论](PAPER_CLAIMS_FINAL.md)
- [论文目录](THESIS_OUTLINE.md)
- [章节写作计划](CHAPTER_WRITING_PLAN.md)
- [图表规划](FIGURE_TABLE_PLAN.md)
- [图索引](figures/generated/figure_index.md)
- [表索引](tables/table_index.md)
- [技能使用清单](SKILL_USAGE_MANIFEST.md)
- [论文局限性](PAPER_LIMITATIONS.md)

## 证据规则

- `STABILITY_SOURCE=MatterSim-5M surrogate`
- `DFT_VERIFIED=False`
- `PROPERTY_TARGET_VERIFIED=False`
- 两次独立 64-seed 组合实验并列报告，不合并为预注册 128-seed 实验。
- Training-overlap 与 Mixed 256 只用于泄漏诊断，禁止包装为独立验证。

