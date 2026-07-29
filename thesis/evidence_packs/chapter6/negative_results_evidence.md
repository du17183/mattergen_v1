# 第6章负面结果证据

## 正文代表性三条

| 路线 | 核心观察 | 停止理由 | 论文认识 | 证据恢复 |
|---|---|---|---|---|
| Corrector Gating | 约1.506×；forward −35.37% | E-hull +0.0224、Stable −9.77 pp、NUS −9.38 pp | 减少物理forward产生速度—质量冲突 | 正式数值由S11完整恢复；原服务器专用报告未全部归档 |
| RP-QTFG | 离线单步方向正向 | 在线RMSD系统恶化，延迟约+30%–49% | 局部代理梯度不等于稳定生成轨迹引导 | `S32_NEGATIVE_RESULTS`; NOT_FULLY_RECOVERED_FROM_ARCHIVE |
| CG-TDR | Gate utility可学习 | Teacher residual方向不能泛化，收益近零/RMSD问题 | 安全选择不能修复错误修正方向 | `S32_NEGATIVE_RESULTS`; NOT_FULLY_RECOVERED_FROM_ARCHIVE |

## 表格/附录路线

Residual Reuse、Budget-aware Gating、FN-PRA、CrystalREPA、Q1 UQ-PQR、Q2 RFR、Q4 CPRC、Q5 CQPS、Q6 NS-SetRank和GPU acceleration routes统一引用`S32_NEGATIVE_RESULTS`与Table 09。部分原始日志/报告不在GitHub，必须保留`NOT_FULLY_RECOVERED_FROM_ARCHIVE`，不得据摘要扩写新数值。
