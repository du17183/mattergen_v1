# 第2章 理论基础与相关工作写作提纲

## 2.1 晶体表示

定义原子种类 \(A\)、分数坐标 \(X\)、晶格 \(L\) 及周期边界；解释位置等变、平移不变与晶格合法性。

## 2.2 MatterGen 扩散过程

- 离散原子字段与连续位置/晶格字段。
- 条件模型、无条件分支和 score。
- Predictor–Corrector 采样流程。

## 2.3 CFG

从固定 CFG 融合公式出发，说明条件残差 \(r_k\) 及三字段尺度差异；综述动态 guidance，但明确本文不做 timestep/corrector skip。

## 2.4 等变后处理

解释图网络、局部环境、原子力、位置更新；区分训练自由物理引导、迭代松弛和本文小步有界精修。

## 2.5 安全门控

综述选择性预测、risk–coverage、trust region、backtracking 和 fallback，建立 Learned Gate 的理论位置。

## 2.6 评价与统计

- MatterSim-5M 是代理势，不是 DFT 真值。
- max force、RMSD、E-hull、Stable、NUS。
- 配对设计、bootstrap 95% CI、Wilcoxon/McNemar/Fisher。
- 训练—测试泄漏与证据资格。

## 2.7 研究空缺

归纳为“在线三字段条件强度控制 + 低风险选择性后生成修正 + 独立可信验证”。

