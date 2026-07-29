# 《基于深度学习的材料逆向生成》论文工作区

学校正式登记论文题目为《基于深度学习的材料逆向生成》。本目录是冻结实验归档`thesis_archive/`的论文写作层，不执行生成、训练、松弛、DFT或任何GPU任务。本文采用预训练MatterGen作为条件晶体扩散生成基线，MatterGen不是本文提出的方法。两项主创新为：

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

- [学校登记题目（冻结）](THESIS_TITLE_FINAL.md)
- [论文研究定位（冻结）](THESIS_POSITIONING_FINAL.md)
- [MatterGen命名与归属规则](MATTERGEN_NAMING_POLICY.md)
- [最终章节、图表与公式编号](CHAPTER_NUMBERING_FINAL.md)
- [论文定位真实性验证](THESIS_POSITIONING_VALIDATION.md)
- [第3–6章可追溯证据包与网页 ChatGPT 写作入口](evidence_packs/README.md)
- [证据包真实性与统计验证报告](evidence_packs/EVIDENCE_PACK_VALIDATION.md)
- [论文写作工作包状态](WRITING_PACKAGE_STATUS.md)
- [旧版第3章正文初稿（历史编号）](chapters/chapter3_draft.md)
- [旧版第4章正文初稿（历史编号）](chapters/chapter4_draft.md)
- [旧版第5章正文初稿（历史编号）](chapters/chapter5_draft.md)
- [旧版第6章正文初稿（历史编号）](chapters/chapter6_draft.md)
- [正文/附录图表安排](MAIN_TEXT_APPENDIX_PLAN.md)
- [7 张核心图 V2 重绘交接](figures/CORE_FIGURES_V2_REDRAW.md)
- [12 张图通用重绘说明](figures/REDRAW_GUIDE.md)
- [10 组表重排说明](tables/REDRAW_GUIDE.md)

- [结论—章节—图表一致性审查](CONSISTENCY_REVIEW.md)
- [冻结论文结论](PAPER_CLAIMS_FINAL.md)
- [旧版论文目录（历史参考）](THESIS_OUTLINE.md)
- [旧版章节写作计划（历史编号）](CHAPTER_WRITING_PLAN.md)
- [旧版图表规划（历史编号）](FIGURE_TABLE_PLAN.md)
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

