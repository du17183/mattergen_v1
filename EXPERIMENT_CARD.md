**DIAGNOSTIC_ONLY — NOT_VALID_AS_INDEPENDENT_VALIDATION**

# 实验卡：Q3 Gate 训练重叠泄漏诊断

**实验状态：LEAKAGE_INFLATION_DETECTED=True**

- 方法：Q3 Gate 训练重叠泄漏诊断
- seed 范围：20000–20255（训练重叠 64；held-out 192）
- 独立性：DIAGNOSTIC_ONLY；Mixed 256 非独立
- 主要结果：训练重叠 harm 0/64；held-out harm 31/192；Fisher p=6.87e-5。
- 数据路径：统一副本在 `thesis_archive/data/leakage_diagnostic`。
- 报告路径：统一副本在 `thesis_archive/reports`。
- 冻结提交：`01e9b2c30e5c58e05eaae908ba291c518b977d03`；PR：https://github.com/du17183/mattergen_v1/pull/16
- 适用论文结论：held-out 192 仅可作补充；整体用于泄漏方法学诊断。
- 禁止使用方式：NOT_VALID_AS_INDEPENDENT_VALIDATION；Mixed 256 不得进正式主结果。
- 限制：MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- 复现说明：冻结实验不可改写；分析使用归档分支 CPU 脚本。
