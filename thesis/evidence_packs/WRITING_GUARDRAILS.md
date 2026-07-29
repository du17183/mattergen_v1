# 写作防护规则

## 论文题目与基线归属

1. 学校正式中文题目固定为“基于深度学习的材料逆向生成”，不得由项目内部方法名称替换。
2. 题目宽于实际实验范围；正文必须明确本文研究对象为周期晶体材料，基础方法为条件扩散生成。
3. MatterGen必须如实说明为预训练条件晶体扩散生成基线、实验框架和实现载体，不得隐藏，也不得写成本文提出的方法。
4. MatterGen原有晶体表示、扩散主干、Predictor–Corrector和原始CFG不得包装为本文创新；本文贡献限于Adaptive CFG和Learned-Gated E3-PCR。
5. `dft_mag_density=0.1`只能写成条件生成任务，不能写成已完成独立目标属性真实性验证。
6. MatterSim-5M不得称DFT、真实热力学真值或实验可合成性；CHGNet也不得称DFT或真实磁性验证。

## 既有实验防护

1. Adaptive CFG不得称统计显著；必须同时给出“方向正向”和“配对统计未显著”。
2. CHGNet不得称真实磁性验证；它在E3-PCR中是辅助特征/更新代理。
3. Gate不得称绝对安全；formal256仍有算法语义harm样本。
4. Always-on平均最大力下降更多（−28.87% vs −23.28%），不得隐藏。
5. 两个64-seed cohort必须分别报告，不得包装为预注册128或事后pooled主结论。
6. Mixed 256不得用于独立正式结论；training overlap只用于诊断；held-out 192只作补充。
7. No-Go不得包装成贡献；只能用于说明假设边界、停止证据和可复用基础设施。
8. Q3只作为历史候选代号；论文正式名称使用Learned-Gated E3-PCR。
9. Win/Tie/Loss必须写明口径：formal E3-PCR主表用raw continuous；组合复现可用1e-6算法语义并将Gate-off计精确平局。
10. `displacement_mean`在统一归档逐seed CSV中缺失，不得从汇总值反推逐seed值。
11. 源码公式必须指明commit；解释性公式不得标为exact。
12. 未由当前仓库支持的事实必须写`NOT_SUPPORTED_BY_CURRENT_REPOSITORY`。
