# 实验卡：创新点一：Multi-field Residual-driven Online Adaptive CFG

**实验状态：FORMAL_INNOVATION1_CONFIRMED=True**

- 方法：创新点一：Multi-field Residual-driven Online Adaptive CFG
- seed 范围：20000–20255
- 独立性：独立正式评估；本方法不使用 Q3 Gate 训练数据
- 主要结果：相对 C0：E-hull -0.003435 eV/atom，Stable +5.859 pp，NUS +3.516 pp。
- 数据路径：统一副本在 `thesis_archive/data/innovation1`。
- 报告路径：统一副本在 `thesis_archive/reports`。
- 冻结提交：`5de00419eea2d8a9be303638f2db8ece15a22366`；PR：https://github.com/du17183/mattergen_v1/pull/1
- 适用论文结论：可用于创新点一正式主结论。
- 禁止使用方式：不得宣称 DFT 或目标属性验证。
- 限制：MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- 复现说明：冻结实验不可改写；分析使用归档分支 CPU 脚本。
