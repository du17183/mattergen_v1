# 第6章 可信性审计、负面结果与讨论

> 正文初稿。本章用于说明证据资格和方法边界，不把诊断结果包装成新的性能创新。

## 6.1 为什么需要证据血缘

本项目经历了多轮方法筛选、开发实验、正式实验、组合验证和诊断实验。若只按方法名称汇总
均值，容易混淆训练数据、调参数据、正式 seed 和独立复现数据。因此，本文为每个结论记录
方法身份、seed 范围、样本量、checkpoint、commit、评价器和数据资格。

图4将证据分为五类：

1. **Formal**：参数冻结后的正式实验；
2. **Formal independent**：与训练 seed 分离的独立正式实验；
3. **Independent replication**：冻结配置上的全新 seed 复现；
4. **Diagnostic/Supplementary**：用于解释偏差或机制，不替代主验证；
5. **Invalid for independent claims**：含训练重叠或错误聚合，禁止用于独立结论。

证据资格描述“允许如何使用”，不代表效果大小。例如，某个诊断 cohort 可以显示很大的
改善，但仍不能替代正式独立实验。

## 6.2 Gate 训练数据与正式实验隔离

Learned Gate 使用 A0 historical outputs seeds 20000–20063 训练。E3-PCR 正式实验使用
40000–40255；组合 Cohort 1 使用 41000–41063；组合 Cohort 2 使用 50000–50063。三个正式/
独立范围与 Gate 训练范围交集均为零。

创新点一正式实验也使用 20000–20255，但它评价的是 Adaptive CFG 本身，不以 Gate 为研究
对象。问题出现在将同一范围的 A0 输出送入已用 20000–20063 训练的 Gate，再把全部 256 个
结果整体称为“独立组合验证”。为量化这种混用带来的偏差，本文保留训练重叠诊断，但明确
否定 Mixed 256 的独立资格。

## 6.3 训练泄漏诊断设计

泄漏诊断将原 20000–20255 范围拆成：

- **Training-overlap**：20000–20063，n=64；
- **Held-out**：20064–20255，n=192。

每个样本比较 A0 与 A0+E3-G。force gain 定义为

\[
G_F=F_{\max}^{A0}-F_{\max}^{A0+E3G},
\]

正值表示降力。refinement harm 定义为 Gate 执行精修后最大力增加。Gate-off exact fallback
不计为 harm。

研究问题被拆成两部分：

1. Training-overlap 是否夸大平均 force gain？
2. Training-overlap 是否夸大 Gate 的安全性，即低估 harm？

这种拆分避免只看平均值而忽略失败概率。

## 6.4 泄漏诊断结果

图11 A 面板显示两个子集的逐 seed force gain 分布。Training-overlap 并未显示出足以支持
“平均降力被明显夸大”的清晰证据；但 B 面板显示：

| 子集 | harm count | harm rate | 证据资格 |
|---|---:|---:|---|
| Training-overlap | 0/64 | 0.00% | Diagnostic only |
| Held-out | 31/192 | 16.15% | Supplementary held-out |

单侧 Fisher exact p=6.87×10⁻⁵。训练重叠数据给出 0% harm，而 held-out 数据有 16.15%
harm，说明重叠显著高估了 Gate 的表观安全性。

该结果揭示了一个重要区别：训练重叠不一定明显改变总体平均收益，却可能改变“哪些结构会
被安全选择”的估计。对于选择性干预模型，安全率比平均效应更容易受到训练重叠影响。

因此，本文冻结：

```text
Mixed 256: INVALID_FOR_INDEPENDENT_CLAIMS=True
```

Mixed 256 只用于泄漏诊断和方法学讨论，不能出现在摘要、创新点主结果或独立验证表中。

## 6.5 负面路线与停止证据

本项目没有把所有尝试都包装为正向结果。图12和负面结果表按照研究假设归纳代表性 No-Go
路线。

### 6.5.1 采样计算削减

**Unconditional residual reuse** 尝试减少 CFG 的逻辑 unconditional NFE。微基准显示，
联合 conditional/unconditional forward 与 conditional-only forward 耗时接近，最佳吞吐
提升仅约 1.16%。其根本原因是 batch size 1 时联合 batch 已能由 GPU 并行执行，减少逻辑
分支没有等比例减少物理 wall time，因此停止。

**Corrector Gating** 和 **Budget-aware Corrector Gating** 确实减少了整个物理 forward，
最高获得约 1.5× 速度，但 Stable、NUS 和 E-hull 出现明显恶化；更保守配置又无法同时满足
速度与质量门槛。这些结果表明直接删除采样步骤存在显著速度—质量折中。

### 6.5.2 表征对齐与教师监督

**FN-PRA Phase 1** 在条件模型上加入静态表示对齐，观察到 RMSD 和 NUS 的部分正向方向，
但 composition validity 和 Stable 均下降 6.25 个百分点。

**CrystalREPA 无条件复现** 尝试隔离条件 CFG 干扰，但 E-hull 恶化约 0.09424 eV/atom，
RMSD 同时恶化，未能复现预期方向。

**CG-TDR** 构建教师残差校正与安全 Gate，但测试集 residual 预测不优于零基线；安全 Gate
几乎退化为不干预，说明已有教师信号不足以支持在线修正。

这些结果说明，“其他体系中的表示对齐有效”不等于在当前 checkpoint、教师、层位置和数据
规模上可以直接迁移。

### 6.5.3 训练自由物理引导

**RP-QTFG** 在离线结构上观察到 CHGNet 修正方向与部分 MatterSim 指标一致，但将该方向
放入在线扩散采样后，RMSD 等指标恶化。离线局部下降方向没有自动转化为噪声轨迹中的安全
引导方向，因此在 Gate 0/后续筛选后停止。

### 6.5.4 后处理、选择与排序

Q1 UQ-PQR、Q2 RFR、Q4 CPRC、Q5 CQPS 和 Q6 NS-SetRank 分别尝试质量预测、力精修、
约束校正、质量保持选择和候选排序。部分路线能够降低力或提供排序信号，但 Novel 下降
约 12.50–30.25 个百分点，部分同时降低 Unique。项目最终没有用“代理质量更高”掩盖生成
多样性损失。

最终 E3-PCR 被保留，是因为它只做小幅 position-only 修改，并使用 Gate 与 exact fallback
限制干预；不是因为它在所有指标上绝对优于所有候选。

### 6.5.5 GPU 执行优化

Native Batch、BF16、静态图、局部 compile、持久化多 worker 和 NVIDIA MPS 等路线均进行
过快速筛选。部分工程设置提高吞吐，但 batch/精度改变可能影响逐 seed 输出；compile、
静态图和 MPS 的增量低于论文创新门槛。持久化多 worker 作为部署工程经验保留，但不作为
第二算法创新。

## 6.6 负面结果带来的方法认识

负面实验不是与论文主线无关的流水账，而是逐步收缩设计空间：

1. 减少“逻辑 NFE”不一定减少真实 GPU 时间；
2. 删除完整采样 forward 可以加速，但质量损失难以回避；
3. 表征对齐对层位置、教师和训练规模敏感；
4. 离线物理下降方向不保证在线扩散轨迹受益；
5. 多候选选择可能以 Novel/Unique 为代价提高某个代理指标；
6. 小型 Gate 在训练重叠数据上会呈现过度乐观的安全率。

这些观察共同促成最终设计：

```text
采样阶段不删 Predictor/Corrector
→ 只在线调节 CFG
→ 生成后进行小步 position-only 修正
→ 学习选择需要干预的结构
→ 所有不安全情况 exact fallback
→ 使用独立 seed 正式验证
```

## 6.7 有效性威胁

### 6.7.1 内部有效性

项目通过同 seed 配对、冻结参数、checkpoint/commit 哈希、Level 1 确定性和独立 seed 范围
降低实现偏差。但部分历史路线经历多轮筛选，仍可能存在选择性探索带来的乐观偏差。因此，
正式结论只使用冻结后的实验，开发阶段结果不与正式结果合并。

### 6.7.2 构念有效性

预松弛最大力、RMSD、E-hull、Stable 和 NUS 均由代理势计算。MatterSim-5M 能提供统一且
可扩展的比较，但代理指标并不等同 DFT 热力学稳定性、动力学稳定性、可合成性或实验真值。
CHGNet 磁矩体积密度也不等同真实 `dft_mag_density` 标签。

### 6.7.3 统计结论有效性

Adaptive CFG 的三个主要指标虽方向一致，但 CI 跨零且 Holm 校正后不显著。两个组合
cohort 只有 n=64，效应分别为 −27.10% 和 −19.02%，存在异质性。Win/Tie/Loss 又包含
原始连续差值和算法语义两种口径，必须在图表中明确区分。

### 6.7.4 外部有效性

当前实验集中于一个 MatterGen 条件 checkpoint、一个目标属性设置、一个材料数据域和一个
正式评价器。尚未证明对其他条件、其他材料体系、不同采样步数、不同代理势或真实 DFT
泛化。

## 6.8 后续研究方向

优先级最高的后续验证是对独立代表性子集进行 DFT single-point/relaxation，检验 MatterSim
方向是否保持。其次，应在全新条件或外部材料域上冻结 Gate 后评估，避免继续在同域 seed 上
扩大样本。第三，可研究 Gate 校准、选择性风险覆盖曲线和 conformal risk control，使
confidence 对 harm risk 具有可解释覆盖保证。第四，可在明确的晶格安全约束下扩展弱 cell
修正，但必须与 position-only 结果分开验证。最后，应与实验或领域数据库合作评估真实属性
命中和可合成性。

## 6.9 本章小结

本章建立了实验血缘和数据资格体系，并用训练重叠/held-out 对照证明重叠显著高估 Gate
安全性。Mixed 256 被明确禁止用于独立结论。系统性 No-Go 结果进一步说明，逻辑计算削减、
激进跳步、表示对齐、训练自由物理引导、后处理选择和 GPU 优化均存在特定边界。保留这些
结果使最终方法的设计选择、适用范围和未解决问题更加清晰。

## 本章图表与引用位置

- Figure 4：6.1 节，证据血缘。
- Figure 11：6.4 节，泄漏诊断。
- Figure 12：6.5 节正文用类别摘要，完整路线放附录。
- Table 1：证据 manifest。
- Table 6：泄漏诊断。
- Table 7：完整 No-Go 路线。
- Table 8：最终 claims 与资格，建议置于第7章前或附录。

## 本章不能删除的限定语

- “Training-overlap 和 Mixed 256 不能作为独立验证”；
- “平均降力未见清晰夸大，但 Gate 安全性被显著高估”；
- “代理势结果不等于 DFT 稳定性、可合成性或真实属性命中”；
- “负面路线的汇总不构造统一分数”。
