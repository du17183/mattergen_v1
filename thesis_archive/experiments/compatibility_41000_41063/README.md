# A0 + E3-G 第一次独立兼容性验证

## 1. 实验目的

A0 + E3-G 第一次独立兼容性验证

## 2. 方法定义

见冻结配置与对应源码；归档不重新实现方法。

## 3. 基线与对照

基线和对照均为同 seed 配对，详见 experiment_manifest.json。

## 4. 分支、commit 和 PR

分支 `feature/a0-e3g-compatibility64`；冻结 commit `ba2303c284210fdae0a35bb0153a8ef3af45a54c`；PR https://github.com/du17183/mattergen_v1/pull/14。

## 5. checkpoint/config SHA256

Adaptive CFG 使用官方条件 checkpoint；E3-PCR 的 Q3 SHA256 与配置 SHA256 见根配置。

## 6. 训练 seed

Q3 Gate 训练 seeds 为 20000–20063；不适用时标记 N/A。

## 7. 开发 seed

开发 seed 不混入独立正式 E3-G 结论；完整范围见审计。

## 8. 测试 seed

41000–41063

## 9. seed 重叠审计

与 Q3 训练和全部开发范围交集为 0

## 10. 样本数

64

## 11. 原始数据来源

来自归档 data 目录中保存的原始逐 seed CSV/JSON 副本。

## 12. 指标定义和单位

力 eV/Å；RMSD Å；E-hull eV/atom；Stable/NUS/Novel/Unique 为布尔率。

## 13. 统计方法

同 seed 配对均值、中位数、bootstrap/Wilcoxon；泄漏 harm 使用 Fisher exact。

## 14. 主要结果

预松弛最大力均值降低 27.10%。

## 15. Go/No-Go 结论

A0_E3G_COMPATIBILITY_GO=True

## 16. 能用于哪些论文结论

可用于组合兼容性独立证据。

## 17. 不能用于哪些论文结论

样本量 64，不替代 E3-PCR 正式 256。

## 18. 已知限制

MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 19. GitHub 未上传的文件

权重、环境、原始数据集、大型缓存和轨迹均未上传。

## 20. 在笔记本上如何重新分析

运行 `python thesis_archive/analysis/validate_archive.py` 后执行统计、制表和绘图脚本。
