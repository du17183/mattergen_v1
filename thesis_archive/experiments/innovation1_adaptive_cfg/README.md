# 创新点一：Multi-field Residual-driven Online Adaptive CFG

## 1. 实验目的

创新点一：Multi-field Residual-driven Online Adaptive CFG

## 2. 方法定义

见冻结配置与对应源码；归档不重新实现方法。

## 3. 基线与对照

基线和对照均为同 seed 配对，详见 experiment_manifest.json。

## 4. 分支、commit 和 PR

分支 `feature/convergence-aware-corrector-gating`；冻结 commit `5de00419eea2d8a9be303638f2db8ece15a22366`；PR https://github.com/du17183/mattergen_v1/pull/1。

## 5. checkpoint/config SHA256

Adaptive CFG 使用官方条件 checkpoint；E3-PCR 的 Q3 SHA256 与配置 SHA256 见根配置。

## 6. 训练 seed

Q3 Gate 训练 seeds 为 20000–20063；不适用时标记 N/A。

## 7. 开发 seed

开发 seed 不混入独立正式 E3-G 结论；完整范围见审计。

## 8. 测试 seed

20000–20255

## 9. seed 重叠审计

独立正式评估；本方法不使用 Q3 Gate 训练数据

## 10. 样本数

256

## 11. 原始数据来源

来自归档 data 目录中保存的原始逐 seed CSV/JSON 副本。

## 12. 指标定义和单位

力 eV/Å；RMSD Å；E-hull eV/atom；Stable/NUS/Novel/Unique 为布尔率。

## 13. 统计方法

同 seed 配对均值、中位数、bootstrap/Wilcoxon；泄漏 harm 使用 Fisher exact。

## 14. 主要结果

相对 C0：E-hull -0.003435 eV/atom，Stable +5.859 pp，NUS +3.516 pp。

## 15. Go/No-Go 结论

FORMAL_INNOVATION1_CONFIRMED=True

## 16. 能用于哪些论文结论

可用于创新点一正式主结论。

## 17. 不能用于哪些论文结论

不得宣称 DFT 或目标属性验证。

## 18. 已知限制

MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 19. GitHub 未上传的文件

权重、环境、原始数据集、大型缓存和轨迹均未上传。

## 20. 在笔记本上如何重新分析

运行 `python thesis_archive/analysis/validate_archive.py` 后执行统计、制表和绘图脚本。
