# 写作防护规则

1. Adaptive CFG不得称统计显著；必须同时给出“方向正向”和“配对统计未显著”。
2. MatterSim-5M不得称DFT、真实热力学真值或实验可合成性。
3. CHGNet不得称真实磁性验证；它在E3-PCR中是辅助特征/更新代理。
4. `PROPERTY_TARGET_VERIFIED=False`；条件值0.1不能写成输出属性已命中。
5. Gate不得称绝对安全；formal256仍有算法语义harm样本。
6. Always-on平均最大力下降更多（−28.87% vs −23.28%），不得隐藏。
7. 两个64-seed cohort必须分别报告，不得包装为预注册128或事后pooled主结论。
8. Mixed 256不得用于独立正式结论；training overlap只用于诊断；held-out 192只作补充。
9. No-Go不得包装成贡献；只能用于说明假设边界、停止证据和可复用基础设施。
10. Q3只作为历史候选代号；论文正式名称使用Learned-Gated E3-PCR。
11. Win/Tie/Loss必须写明口径：formal E3-PCR主表用raw continuous；组合复现可用1e-6算法语义并将Gate-off计精确平局。
12. `displacement_mean`在统一归档逐seed CSV中缺失，不得从汇总值反推逐seed值。
13. 源码公式必须指明commit；解释性公式不得标为exact。
14. 未由当前仓库支持的事实必须写`NOT_SUPPORTED_BY_CURRENT_REPOSITORY`。
