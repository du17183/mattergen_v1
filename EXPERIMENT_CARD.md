# 实验卡：创新点二：Learned-Gated E3-PCR 正式 256

**实验状态：FINAL_STATE=E3_G_FORMAL_CONFIRMED**

- 方法：创新点二：Learned-Gated E3-PCR 正式 256
- seed 范围：40000–40255
- 独立性：正式 seeds 与 Q3 训练 seeds 20000–20063 交集为 0
- 主要结果：E3-G 最大力均值降低 23.28%，harm rate 18.359%，质量代理指标保持。
- 数据路径：统一副本在 `thesis_archive/data/innovation2`。
- 报告路径：统一副本在 `thesis_archive/reports`。
- 冻结提交：`0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`；PR：https://github.com/du17183/mattergen_v1/pull/13
- 适用论文结论：可用于创新点二独立正式主结论。
- 禁止使用方式：不得把 MatterSim 代理评价表述为 DFT。
- 限制：MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- 复现说明：冻结实验不可改写；分析使用归档分支 CPU 脚本。
