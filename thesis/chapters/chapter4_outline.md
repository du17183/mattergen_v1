# 第4章 Learned-Gated E3-PCR 写作提纲

## 4.1 问题与设计原则

降低预松弛最大力，同时固定原子种类和晶胞；对低风险结构尽量不干预。

## 4.2 14维风险特征与 Gate

- 列出特征类别（结构/局部几何/风险统计，具体名从实现附录引用）。
- 129 参数、threshold=0.5、训练 seeds 20000–20063。
- 训练集只用于 Gate，不用于正式 40000–40255。

## 4.3 有界等变位置精修

- 5 steps，eta=0.01。
- 单步 radius=0.02 Å，累计上限=0.10 Å。
- backtrack≤3、安全检查、拒绝回退。
- Figure 3。

## 4.4 三臂实验

C0、E3-A Always-on、E3-G Learned-gated；n=256；相同输入结构、代理评价器和配对统计。

## 4.5 主效果

- Figure 6 / Table 3。
- E3-G max force 0.342964→0.263107 eV/Å，−23.28%。
- CI、Holm p、163/0/93。
- Stable/NUS/E-hull/Novel/Unique 保持。

## 4.6 Gate 机制

- Figure 7 / Table 4。
- coverage 100%→66.406%。
- harm 25.391%→18.359%；low-force harm 29.688%→17.969%。
- gain retention 80.657%；McNemar p=0.000534。
- displacement 减少。

## 4.7 Confidence 诊断

Figure 8 报告 Spearman 相关与 Gate-on/off；趋势仅描述，不作因果。

## 4.8 局限

Always-on 平均降力更大；Gate 仍有 18.359% harm；未做 DFT；不改 cell/species。

