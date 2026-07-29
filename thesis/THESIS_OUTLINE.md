# 硕士论文建议目录

建议题目：

> **面向条件晶体生成的残差自适应引导与安全有界后生成精修方法研究**

英文：

> **Residual-Adaptive Guidance and Safe-Bounded Post-Generation Refinement for Conditional Crystal Generation**

## 第1章 绪论

1. 研究背景：晶体结构发现、条件生成、扩散模型与代理势评价。
2. MatterGen 的价值与限制：条件控制、三字段生成、采样成本及预松弛几何风险。
3. 问题定义：
   - 固定 CFG 难以适应三个字段与不同采样阶段；
   - 生成结束后的高力结构需要安全、低成本且不改变组成的修正；
   - 训练泄漏会高估小型 Gate 的安全性。
4. 研究内容与贡献：
   - 多字段残差驱动在线 Adaptive CFG；
   - Learned-Gated Safe-Bounded E3-PCR；
   - 两次独立组合验证与泄漏诊断；
   - 系统性 No-Go 结果作为方法边界。
5. 论文组织。

## 第2章 相关理论与研究现状

1. 晶体表示：元素、分数坐标、晶格及周期边界。
2. 扩散生成：连续位置/晶格与离散原子字段。
3. Classifier-Free Guidance 与条件残差。
4. Predictor–Corrector 采样。
5. 等变图网络、原子力与后生成几何优化。
6. 安全门控、选择性预测、trust region 和 fallback。
7. 评价指标、代理势及统计推断。
8. 现有研究空缺。

## 第3章 多字段残差驱动在线 Adaptive CFG

1. C0 原始条件 MatterGen 与固定 CFG 基线。
2. 三字段残差定义及尺度不一致问题。
3. EMA 稳定化、在线尺度更新和 [0,5] 限幅。
4. 与完整 Predictor/Corrector 的集成；明确不跳步。
5. 实现与确定性。
6. 256-seed 正式实验设计。
7. E-hull、Stable、NUS 正向趋势与非显著性解释。
8. 本章小结。

## 第4章 Learned-Gated Safe-Bounded E3-PCR

1. 预松弛最大力问题与后生成精修假设。
2. 14 维风险特征及 129 参数 Gate。
3. 5 步等变位置精修。
4. 每步半径、累计 trust region、backtracking、安全检查。
5. Gate-off/拒绝的精确回退；原子种类与晶胞不变。
6. 三臂 C0/E3-A/E3-G 设计。
7. 独立 256-seed 主效果。
8. Gate 安全机制消融。
9. 局限性：Always-on 平均降力更大；Gate 不保证逐结构安全。

## 第5章 两项创新的组合与独立验证

1. 组合流程 A0+E3-G。
2. Cohort 1（41000–41063）设计与结果。
3. Cohort 2（50000–50063）预注册式独立复现与结果。
4. forest plot 与逐 seed 配对结果。
5. 两次效应大小差异；不进行事后强制合并。
6. 组合兼容性与推论边界。

## 第6章 可信性审计、负面结果与讨论

1. Gate 训练数据与正式 seed 隔离。
2. Training-overlap/Held-out 泄漏诊断。
3. 平均效果与安全性偏差的分离。
4. Mixed 256 资格否定。
5. Corrector Gating、REPA、physics guidance、GPU 优化等 No-Go 路线。
6. 为什么保留负面结果能加强研究可信度。
7. MatterSim 代理评价、未做 DFT、未验证真实磁密度的限制。
8. 未来工作：DFT 子集、外部数据集、Gate 校准及更强安全证据。

## 第7章 总结与展望

1. 回答两个研究问题。
2. 总结方法贡献、工程贡献和方法学贡献。
3. 重申结论资格与不能声称的内容。
4. 展望 DFT、实验验证、跨材料体系泛化及在线安全控制。

