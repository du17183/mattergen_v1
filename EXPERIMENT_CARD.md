# 实验卡：A0 + E3-G 第一次独立兼容性验证

**实验状态：A0_E3G_COMPATIBILITY_GO=True**

- 方法：A0 + E3-G 第一次独立兼容性验证
- seed 范围：41000–41063
- 独立性：与 Q3 训练和全部开发范围交集为 0
- 主要结果：预松弛最大力均值降低 27.10%。
- 数据路径：统一副本在 `thesis_archive/data/compatibility_1`。
- 报告路径：统一副本在 `thesis_archive/reports`。
- 冻结提交：`ba2303c284210fdae0a35bb0153a8ef3af45a54c`；PR：https://github.com/du17183/mattergen_v1/pull/14
- 适用论文结论：可用于组合兼容性独立证据。
- 禁止使用方式：样本量 64，不替代 E3-PCR 正式 256。
- 限制：MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- 复现说明：冻结实验不可改写；分析使用归档分支 CPU 脚本。
