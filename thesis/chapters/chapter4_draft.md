# 第4章 学习门控的安全有界等变后生成晶体精修

> 正文初稿。结论口径以 `thesis/PAPER_CLAIMS_FINAL.md` 为准。E3-PCR 的力、能量和稳定性
> 结果均为代理势结果，不代表 DFT 或实验真值。

## 4.1 问题定义与研究假设

扩散生成模型输出的是未松弛晶体结构。即使组成和基本结构有效，局部原子环境仍可能包含
较大的预松弛力。直接进行较强的后处理可能降低平均最大力，但也可能移动原本风险较低的
结构，增加伤害率或损害生成多样性。因此，第二项研究问题定义为：

> 能否在不重新训练 MatterGen 主干、不改变原子种类和晶胞的前提下，用一个轻量风险 Gate
> 选择需要干预的结构，并通过小步、可拒绝、可精确回退的位置精修降低预松弛最大力？

本文提出 Safe-Bounded Equivariant Post-Generation Crystal Refiner（E3-PCR），并使用
Learned Gate 构成最终方法 E3-G。E3-PCR 根据冻结 CHGNet 的原子力执行有限步位置更新，
每一步均受单步半径、累计 trust region、backtracking 和安全条件约束。Learned Gate
只决定是否进入精修器；Gate-off 或任何拒绝均返回输入结构。

研究设计将“最大平均降力”和“减少有害干预”分成两个不同目标。Always-on E3-A 用于观察
所有结构都精修时的降力上限；Learned-gated E3-G 用于研究选择性干预的风险—收益折中。

## 4.2 结构风险特征

对生成结构 \(X\)，首先使用几何信息和冻结 CHGNet 预测构造 14 维特征
\(\phi(X)\in\mathbb R^{14}\)。全部特征如下：

| 类别 | 特征 |
|---|---|
| 尺度与密度 | 原子数、每原子体积、质量密度 |
| 几何与组成 | 最短原子距离、原子序数均值、原子序数标准差、晶胞条件数 |
| CHGNet 能量与力 | 每原子能量、force RMS、最大原子力、平均原子力 |
| CHGNet 应力与磁代理 | stress RMS、最大绝对应力、绝对 site magmom 体积密度 |

特征提取会检查非空结构、原子序数范围、正有限体积以及 CHGNet 预测的有限性。最短距离
采用周期性最小镜像距离；晶胞条件数用于表示晶格数值病态程度。磁密度特征只是 Gate 输入
中的一个 CHGNet 代理量，不构成真实 `dft_mag_density` 预测或属性验证。

## 4.3 轻量 Learned Gate

Gate 采用 StandardScaler 与单隐层 MLP：

\[
14 \rightarrow 8 \rightarrow 1,
\]

隐层激活为 tanh，正则项 \(\alpha_{\mathrm{MLP}}=0.1\)，随机种子为 20260728。网络总可训练
参数为

\[
14\times 8+8+8\times1+1=129.
\]

输出概率记为

\[
c(X)=P(y=1\mid \phi(X)).
\]

当 \(c(X)\ge \tau\) 时执行 E3-PCR，其中阈值 \(\tau=0.5\)；否则直接 exact fallback。

训练数据由 A0 的历史输出 seeds 20000–20063 构成，共 64 个结构。对于每个结构，离线比较
五步位置精修前后的初始最大力，并以“精修后最大力更低”作为二元标签。训练流程使用
8-fold stratified out-of-fold 概率进行冻结前检查，随后在 64 条训练数据上拟合最终 Gate。
正式 E3-PCR 实验 seeds 40000–40255 与训练 seeds 交集为零。

Gate 的目标不是直接回归力，也不是生成新原子坐标。它只学习一个选择函数：

\[
z(X)=\mathbb I[c(X)\ge 0.5],
\]

其中 \(z=1\) 表示允许尝试精修，\(z=0\) 表示保持输入结构。

## 4.4 有界等变位置更新

设第 \(j\) 步结构为 \(X^{(j)}\)，冻结 CHGNet 预测的原子力为
\(F^{(j)}\in\mathbb R^{N\times3}\)。基础位移提议为

\[
\Delta x_i^{(j)}
=\eta F_i^{(j)},
\qquad \eta=0.01.
\]

为限制单个原子的移动，对每个原子独立裁剪：

\[
\widehat{\Delta x}_i^{(j)}
=\Delta x_i^{(j)}
\min\left(
1,
\frac{R_{\mathrm{step}}}
{\|\Delta x_i^{(j)}\|_2+\epsilon}
\right),
\]

其中单步半径 \(R_{\mathrm{step}}=0.02\) Å。实现总共尝试 5 个 refinement step，并使用周期
边界 wrap 更新位置。由于更新方向来自旋转协变的原子力，并对每个原子使用旋转不变的范数
限幅，该位置更新保持平移、旋转和原子置换语义的一致性。

最终结构相对输入结构的 wrapped Cartesian 位移还受累计 trust region

\[
\max_i\|x_i^{(5)}-x_i^{(0)}\|_{\mathrm{MIC}}\le R_{\mathrm{cum}},
\qquad R_{\mathrm{cum}}=0.10\text{ Å}
\]

约束。原子种类和晶胞矩阵在整个过程中保持不变。

## 4.5 Backtracking 与安全验收

对每个 refinement step，系统最多尝试三个步长尺度：

\[
\lambda_b\in\{1,\tfrac12,\tfrac14\},
\qquad b=0,1,2.
\]

候选位移写为

\[
\Delta x_{i,b}^{(j)}
=\operatorname{clip}_{R_{\mathrm{step}}\lambda_b}
\left(\eta\lambda_bF_i^{(j)}\right).
\]

候选结构必须同时满足：

1. 坐标、晶胞和 CHGNet 输出均为有限值；
2. 周期最短原子距离不小于 0.5 Å；
3. 新 CHGNet energy 不高于该步原结构能量（容差 \(10^{-7}\)）；
4. 未违反累计位移约束。

第一个满足条件的候选被接受并进入下一步。若三个尺度全部被拒绝，该 step 返回原位置并
记录 fallback。Gate-off、CHGNet 错误、非有限提议、短键、能量升高或所有回溯失败，均不会
输出一个部分不安全的替代结构。

需要区分两类回退：

- **Gate-level fallback**：\(c<0.5\)，完全不进入精修；
- **step-level rejection**：进入精修后当前候选均不安全，该步保留旧结构。

最终输出始终与输入具有相同的 atomic numbers 和 cell。

## 4.6 算法流程

**算法4-1 Learned-Gated Safe-Bounded E3-PCR**

```text
输入：MatterGen 生成结构 X
冻结组件：CHGNet、StandardScaler、14→8→1 Gate

1. 提取 14 维风险特征 φ(X)
2. 计算 confidence c
3. 若 c < 0.5：
      返回 X（exact fallback）
4. 令 X_current ← X
5. 重复 5 个 refinement step：
      a. 计算 CHGNet energy 与原子力
      b. 依次尝试 scale ∈ {1, 1/2, 1/4}
      c. 生成 position-only proposal
      d. 应用单步 0.02 Å 和累计 0.10 Å trust region
      e. 检查有限性、最短距离、energy non-increase
      f. 接受首个安全 proposal；若均失败则保持 X_current
6. 返回 X_current

不变量：atomic numbers 不变；cell 不变；Gate 和 CHGNet 参数不更新
```

图3将风险估计、循环精修和安全决定分为三个阶段，并显式显示拒绝后的 exact fallback。

## 4.7 三臂正式实验设计

正式独立实验使用 seeds 40000–40255，共 256 个 C0 输入结构。每个 C0 只生成一次，E3-A
和 E3-G 都从同一个输入结构派生，因此三臂共享组成、晶胞和初始原子序列：

- **C0**：不执行后生成精修；
- **E3-A**：对全部结构执行 E3-PCR；
- **E3-G**：仅对 \(c\ge0.5\) 的结构执行 E3-PCR。

三臂共完成 768/768 MatterSim relaxation，relaxation failure、short bond 和 abnormal
cell 均为 0。主要终点是预松弛最大力 \(F_{\max}\)。配对差定义为

\[
\Delta F_{\max}=F_{\max}^{\mathrm{selected}}-F_{\max}^{\mathrm{C0}},
\]

因此负值表示改善。质量安全指标包括 RMSD、E-hull、Stable、NUS、Novel、Unique、
composition validity 和 structure validity。

## 4.8 E3-PCR 主效果

三臂聚合结果如下。

| 方法 | 最大力 (eV/Å) | RMSD (Å) | E-hull (eV/atom) | Stable | NUS | Novel | Unique |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.342964 | 0.049390 | 0.156136 | 44.531% | 22.266% | 71.875% | 98.438% |
| E3-A | 0.243956 | 0.045057 | 0.156179 | 44.531% | 22.266% | 71.875% | 98.438% |
| E3-G | 0.263107 | 0.045937 | 0.156177 | 44.531% | 22.266% | 71.875% | 98.438% |

E3-G 将平均预松弛最大力从 0.342964 降低至 0.263107 eV/Å，相对变化为 −23.28%。
配对均值差为 −0.079857 eV/Å，paired bootstrap 95% CI 为
[−0.144966, −0.032453]；Holm-adjusted Wilcoxon p=4.19×10⁻¹⁰，原始连续差值
Win/Tie/Loss=163/0/93。Leave-one-out 分析中均值差仍保持负值，说明平均结果并非完全由
单一样本决定。

E3-G 的 RMSD 平均降低约 0.003452 Å；E-hull 变化仅约 +0.000041 eV/atom。Stable、NUS、
Novel 和 Unique 的聚合比例与 C0 相同。上述结果支持“在 MatterSim-5M 代理质量指标基本
保持的同时降低预松弛最大力”，但不支持“提高 DFT 稳定性”。

图6的 A 面板用三臂 ECDF 展示最大力长尾，B 面板展示 256 个排序配对差。图中同时保留
93 个逐 seed 伤害样本，以防只看均值而误认为每个结构都改善。

## 4.9 Learned Gate 安全机制

Always-on E3-A 的平均最大力降至 0.243956 eV/Å，相对 C0 降低 28.87%，高于 E3-G 的
23.28%。因此 Gate 的价值不是获得更大的平均降力，而是减少不必要和有害的干预。

机制消融结果如下：

| 指标 | E3-A Always-on | E3-G Learned-gated |
|---|---:|---:|
| refinement coverage | 100.000% | 66.406% |
| overall harm | 25.391% | 18.359% |
| low-force harm | 29.688% | 17.969% |
| mean displacement | 0.010968 Å | 0.007580 Å |
| P95 displacement | 0.027870 Å | 0.025003 Å |
| mean force-gain retention | 100.000% | 80.657% |

Gate 将总体伤害率降低 7.032 个百分点，将低初始力子集伤害率降低 11.719 个百分点；
配对伤害差异 McNemar exact p=0.000534。与此同时，它保留了 80.657% 的 Always-on 平均
降力收益。平均位移降低 30.89%，P95 位移降低 10.29%。

图7将 intervention coverage、harm、displacement 和 gain retention 分开呈现。该结果说明
学习选择能够降低风险，但正式 E3-G 中仍有 18.359% 的结构出现最大力恶化，因而不能写成
“Gate 保证安全”。

## 4.10 Confidence 与真实改善

在正式 256 个样本中，Gate confidence 与真实 force gain

\[
G_F=F_{\max}^{\mathrm{C0}}-F_{\max}^{\mathrm{E3-G}}
\]

的 Spearman 相关系数为 \(\rho=0.375\)，p=5.44×10⁻¹⁰。该结果表明 confidence 与实际
改善存在中等程度单调关联，但散点仍包含高 confidence 的伤害样本。图8中的线性趋势仅作
描述，不代表 Gate 已完成概率校准，更不证明因果关系。

## 4.11 局限性与有效性边界

第一，E3-G 的平均降力小于 E3-A，说明选择性带来了可见的收益损失。第二，Gate 只有 64 条
训练结构和 129 个参数，训练重叠会显著高估安全率，第6章将专门审计这一问题。第三，
E3-PCR 只改位置，无法修复错误组成、元素种类或晶胞。第四，安全验收使用 CHGNet energy，
正式评价使用 MatterSim-5M；二者仍都是机器学习代理势。第五，当前没有 DFT、动力学或
实验合成验证，也没有真实磁密度命中验证。

## 4.12 本章小结

本章提出 Learned-Gated Safe-Bounded E3-PCR。14 维结构风险特征输入 129 参数 Gate，
选择性触发五步 CHGNet 力引导的位置精修；单步半径、累计 trust region、backtracking、
能量和几何检查共同限制修改，任何不安全情况均回退输入。独立 256-seed 实验显示 E3-G
平均降低预松弛最大力 23.28%，主要 MatterSim 代理质量比例保持不变。Learned Gate 相对
Always-on 降低覆盖和伤害并保留 80.657% 平均收益，但不保证每个结构改善。

## 本章图表与引用位置

- Figure 3：4.6 节，展示 Gate、循环精修和拒绝路径。
- Figure 6：4.8 节，主效果。
- Figure 7：4.9 节，安全机制消融。
- Figure 8：4.10 节，可放正文末或附录。
- Table 3：三臂质量结果。
- Table 4：Gate 覆盖、伤害、位移和收益保留。

## 本章不能删除的限定语

- “Gate 的平均降力不优于 Always-on”；
- “Gate 不保证逐结构安全，正式实验仍有伤害样本”；
- “仅修改原子位置，不修改原子种类和晶胞”；
- “所有稳定性和力结果为代理势评价，无 DFT 验证”。
