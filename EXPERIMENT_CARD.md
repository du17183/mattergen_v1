**SOURCE_DATA_INCOMPLETE — NO_EFFECT_ESTIMATE_PRODUCED**

# 实验卡：A0 256 复用资格审计

**实验状态：SOURCE_DATA_INCOMPLETE — NO_EFFECT_ESTIMATE_PRODUCED**

- 方法：A0 256 复用资格审计
- seed 范围：候选 20000–20255
- 独立性：64 个 seeds 与 Q3 训练重叠，故整个旧批次不具正式复用资格
- 主要结果：未启动组合效果计算；无效应估计。
- 数据路径：统一副本在 归档分支 `thesis_archive/experiments/source_data_incomplete_audit/source`。
- 报告路径：统一副本在 `thesis_archive/reports`。
- 冻结提交：`c1df24a8e5d118dcc99d7fb65b34e7f53be28969`；PR：https://github.com/du17183/mattergen_v1/pull/15
- 适用论文结论：只用于说明数据资格审计和停止决策。
- 禁止使用方式：不得写成方法 No-Go，不得生成或引用效果估计。
- 限制：MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- 复现说明：冻结实验不可改写；分析使用归档分支 CPU 脚本。
