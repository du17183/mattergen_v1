# MatterGen 论文分析归档 v1

本项目研究条件晶体生成。创新点一 **Multi-field Residual-driven Online Adaptive CFG** 动态调节多字段条件残差；创新点二 **Learned-Gated E3-PCR** 用 129 参数 Q3 Gate 选择性执行局部位置精修，降低预松弛最大力并控制 refinement harm。

## 正式与补充证据

- Innovation1：20000–20255，256 配对，正式成立。
- Innovation2：40000–40255，256 完全独立，正式成立。
- 组合验证：41000–41063；完全独立复现：50000–50063；两批都必须报告。
- 泄漏诊断：20000–20063 与 Q3 训练重叠；20064–20255 held-out。Mixed 256 **不能作为独立正式结果**，held-out 192 仅为补充。
- SOURCE_DATA_INCOMPLETE 是复用资格失败，不是方法 No-Go，也无效应估计。

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

因此指标均为代理结果，不是 DFT 或真实磁性证明。按 README_FOR_LAPTOP.md 安装 CPU 依赖并运行验证、统计、表格和绘图脚本。权重与缓存不在 GitHub，重算分析不需要它们。seed 保留真实值，没有匿名化或删除不利样本。
