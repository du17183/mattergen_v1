# 实验卡：A0 + E3-G 第二次完全独立复现

**实验状态：A0_E3G_INDEPENDENT64_GO=True**

- 方法：A0 + E3-G 第二次完全独立复现
- seed 范围：50000–50063
- 独立性：预注册全新 seeds；无训练、调参或历史测试交集
- 主要结果：预松弛最大力均值降低 19.02%。
- 数据路径：统一副本在 `thesis_archive/data/compatibility_2`。
- 报告路径：统一副本在 `thesis_archive/reports`。
- 冻结提交：`22e1db74a59476562f1f746cd4210b9420cbdf05`；PR：https://github.com/du17183/mattergen_v1/pull/17
- 适用论文结论：可用于第二次完全独立复现证据。
- 禁止使用方式：不得只选择该批次而隐去第一次独立验证。
- 限制：MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。
- 复现说明：冻结实验不可改写；分析使用归档分支 CPU 脚本。
