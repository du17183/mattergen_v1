# 第3章 多字段残差驱动在线自适应 Classifier-Free Guidance

> 正文初稿。引用编号需在学校论文模板中根据最终参考文献库统一替换。本章中的数值和结论
> 已按 `thesis/PAPER_CLAIMS_FINAL.md` 冻结。

## 3.1 研究动机

MatterGen 将晶体结构表示为晶胞、周期位置和原子种类三个相互耦合的字段，并通过扩散模型
逐步恢复结构。在条件生成中，Classifier-Free Guidance（CFG）通过条件分支与无条件分支的
差异增强目标属性的影响。原始条件 MatterGen 使用固定 guidance scale。该做法实现简单，
但默认一个常数能够适用于不同采样阶段以及形状、数值范围和物理含义不同的三个字段。

固定尺度存在两个潜在问题。第一，条件残差会随噪声水平和 Predictor/Corrector 阶段发生
变化，固定尺度无法响应当前条件信号的相对强弱。第二，cell、position 和 atomic number
score 的张量形状及数值尺度不同，直接拼接后计算统一范数会让维度较大或数值较大的字段
主导调节信号。因此，本章研究问题定义为：

> 在不训练 MatterGen 主干、不减少 Predictor/Corrector 步骤的前提下，能否利用三个字段
> 的在线条件残差，构造稳定、可追踪且有界的自适应 CFG？

为回答该问题，本文提出 Multi-field Residual-driven Online Adaptive CFG。该方法分别计算
三个字段的残差 RMS，再将有效字段的标量统计聚合为当前条件强度；随后使用按采样阶段隔离
的指数移动平均建立局部基准，依据当前值与基准的比值调整一个全局 CFG scale。方法不改变
模型参数、不改变 score 网络结构，也不删除任何采样步骤。

## 3.2 基线与符号定义

设时刻 \(t\) 的条件 score 和无条件 score 分别为
\(s_{t,k}^{\mathrm{cond}}\) 与 \(s_{t,k}^{\mathrm{uncond}}\)，其中字段

\[
k\in\mathcal K=\{\mathrm{cell},\mathrm{pos},\mathrm{atom}\}.
\]

字段条件残差定义为

\[
r_{t,k}=s_{t,k}^{\mathrm{cond}}-s_{t,k}^{\mathrm{uncond}}.
\]

原始固定 CFG 可写为

\[
\tilde s_{t,k}
=s_{t,k}^{\mathrm{uncond}}
+g_0\left(s_{t,k}^{\mathrm{cond}}-s_{t,k}^{\mathrm{uncond}}\right),
\]

其中冻结基线使用 \(g_0=2.0\)。实验中的 C0 指官方 `dft_mag_density` 条件 MatterGen、
固定 CFG、FP32、batch size 1 和完整 Predictor/Corrector。A0 指在相同 checkpoint 和
采样设置上启用本章自适应控制器的模型。

需要特别说明，本文并未为三个字段分别学习三个 guidance scale。三个字段分别归约的目的
是避免直接拼接张量产生形状偏置；归约后形成一个共同的、逐 score call 更新的 scale，
从而保持与原始 CFG 融合形式一致。

## 3.3 多字段残差统计

对每个有效字段，首先计算条件残差的均方根：

\[
\delta_{t,k}
=\operatorname{RMS}(r_{t,k})
=\sqrt{\frac{1}{N_k}\sum_{i=1}^{N_k}r_{t,k,i}^{2}},
\]

其中 \(N_k\) 为字段 \(k\) 的 score 元素数。RMS 将每个字段归约为一个与字段维数弱相关的
标量，避免 position 字段仅因元素数量更多而获得更大的统计权重。

当前 score call 的多字段残差强度为所有有效字段 RMS 的算术平均：

\[
\delta_t
=\frac{1}{|\mathcal K_t^{\mathrm{valid}}|}
\sum_{k\in\mathcal K_t^{\mathrm{valid}}}\delta_{t,k}.
\]

实现会检查字段是否存在、条件与无条件张量形状是否一致、张量是否为空以及数值是否有限。
若残差无效，控制器不更新自适应状态，并回退到当前 stage guidance。该回退策略的目标是
使日志或异常字段不会破坏原始采样路径。

## 3.4 分阶段 EMA 与有界在线更新

Predictor 和 Corrector 的 score 调用具有不同语义和残差分布。若二者共享同一 EMA，
阶段切换本身可能被误认为条件强度突变。因此，本文为两个 phase 维护相互独立的状态
\(m_{t,p}\)，其中 \(p\in\{\mathrm{predictor},\mathrm{corrector}\}\)。

对同一 phase 的后续观测，EMA 更新为

\[
m_{t,p}
=\beta m_{t-1,p}
+(1-\beta)\delta_t,
\qquad \beta=0.95.
\]

每个 phase 的第一次观测直接用 \(\delta_t\) 初始化，而不是与零值历史比较，以避免产生
人为的首步放大。随后计算当前残差相对局部基准的比值：

\[
q_t=\frac{\delta_t}{m_{t,p}+\epsilon},
\qquad \epsilon=10^{-6}.
\]

自适应 multiplier 为

\[
u_t
=\operatorname{clip}
\left(1+\alpha(q_t-1),\,0.25,\,4.0\right),
\qquad \alpha=0.50.
\]

最终 guidance scale 定义为

\[
g_t
=\operatorname{clip}(g_0u_t,\,0,\,5).
\]

当当前残差高于 phase 局部平均时，\(q_t>1\)，控制器提高 guidance；当残差低于局部平均
时，控制器降低 guidance。内部 multiplier 和最终 scale 的双重限幅分别防止相对比值突变
和绝对 guidance 越界。A0 使用 adaptive schedule，不启用额外的 warmup/decay 分段。

## 3.5 与 MatterGen Predictor–Corrector 的集成

图2给出了实现流程。每次需要 score 时，采样器将无条件输入和条件输入拼接成一个 joint
batch，通过一次联合模型调用得到两条 score。随后对三个被腐蚀字段分别计算残差 RMS，
更新控制器并使用 \(g_t\) 完成 CFG 融合：

\[
\tilde s_{t,k}
=s_{t,k}^{\mathrm{uncond}}
+g_t r_{t,k}.
\]

自适应控制器只改变 score 融合系数，不改变以下组件：

1. MatterGen checkpoint 和主干参数；
2. Predictor 与 Corrector 的执行次数；
3. 三字段 SDE/D3PM 腐蚀过程；
4. 初始随机状态和采样 seed；
5. 原始 joint conditional/unconditional forward 结构。

因此，本方法与 Corrector Gating、timestep skipping 或 score caching 不同。本文不把 A0
表述为推理加速方法，其目的在于改善条件控制的在线适应性。

## 3.6 算法流程

**算法3-1 多字段残差驱动在线 Adaptive CFG**

```text
输入：当前状态 x_t，时间 t，基础 guidance g0
状态：predictor EMA、corrector EMA

1. 构造 unconditional 和 conditional 输入，并联合计算两条 score
2. 对 cell、position、atom：
      r_k ← s_cond,k − s_uncond,k
      δ_k ← RMS(r_k)
3. δ ← 所有有效 δ_k 的平均
4. 读取当前 phase 对应的 EMA
5. 若为该 phase 首次观测：EMA ← δ
   否则：EMA ← β·EMA + (1−β)·δ
6. q ← δ / (EMA + ε)
7. u ← clip(1 + α(q−1), 0.25, 4)
8. g ← clip(g0·u, 0, 5)
9. 对每个字段：
      s_guided,k ← s_uncond,k + g·r_k
10. 将 s_guided 返回完整 Predictor/Corrector 流程

若字段缺失、形状错误或出现非有限值：使用 g0 并记录 fallback reason
```

控制器在每个 sampled batch 开始时重置，避免不同 seed 或 batch 之间发生状态泄漏。可选
trace 记录 sample seed、sampling step、phase、三个字段 RMS、EMA、ratio、multiplier、
最终 scale 和 fallback 原因；trace 本身不改变 RNG。

## 3.7 实验设置

正式实验使用 seeds 20000–20255，共 256 对样本。C0 与 A0 使用相同 seed、相同初始状态、
相同官方 `dft_mag_density` checkpoint、FP32、batch size 1 和完整 Predictor/Corrector。
每个方法均完成 256/256 generation 和 256/256 MatterSim relaxation，跨方法初始状态配对
检查通过，确定性达到 Level 1。

冻结参数如下：

| 参数 | 数值 |
|---|---:|
| 基础 guidance \(g_0\) | 2.0 |
| 自适应强度 \(\alpha\) | 0.50 |
| EMA 系数 \(\beta\) | 0.95 |
| 数值稳定项 \(\epsilon\) | \(10^{-6}\) |
| multiplier 范围 | [0.25, 4.0] |
| guidance 范围 | [0, 5] |
| 精度 | FP32 |
| batch size | 1 |
| Predictor/Corrector | 完整保留 |

评价统一使用 MatterSim-5M 代理势。主要指标为 E-hull、Stable 和 NUS，并以相同 seed 进行
配对统计。连续指标使用配对 bootstrap 95% CI 与 Wilcoxon signed-rank test；二元指标使用
精确配对 discordant-binomial/McNemar 检验，并对主指标 p 值进行 Holm 校正。

## 3.8 正式结果

表3-1概括主要结果。

| 指标 | C0 | A0 Adaptive CFG | A0−C0 | 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|
| E-hull (eV/atom) | 0.143667 | 0.140232 | −0.003435 | [−0.017926, 0.011030] | 1.00 |
| Stable | 41.016% | 46.875% | +5.859 pp | [−1.563, 13.281] pp | 1.00 |
| NUS | 22.266% | 25.781% | +3.516 pp | [−2.734, 9.766] pp | 1.00 |

从方向上看，三个主要指标均朝有利方向变化：平均 E-hull 降低约
0.003435 eV/atom，Stable 提高 5.859 个百分点，NUS 提高 3.516 个百分点。E-hull 的
逐 seed Win/Tie/Loss 为 137/1/118；Stable 为 54/163/39；NUS 为 40/185/31。

但是，三个 paired bootstrap 95% CI 均跨越零，Holm 校正 p 值均为 1.00。因此，正式结果
只能解释为“总体正向趋势”，不能写成统计显著的质量提升。图5使用原始逐 seed 配对点和
区间估计同时展示效应方向与不确定性，避免只报告均值。

补充质量指标显示，Novel 从 73.828% 变为 76.563%，Unique 从 95.703% 变为 98.047%；
composition validity 从 84.375% 变为 83.984%，变化为 −0.391 个百分点；structure
validity 均为 100%。这些指标用于描述整体行为，不替代三个冻结主要指标。

## 3.9 讨论

结果支持两个层面的结论。首先，在线残差信号能够在不训练主干和不删除采样步骤的情况下
稳定接入 MatterGen，并在 256 个正式配对样本中得到方向一致的 E-hull、Stable 和 NUS
变化。其次，效应方差相对于均值较大，当前样本下不足以排除零效应。这说明方法贡献主要
体现在多字段条件信息的在线归约、分阶段状态隔离、有界控制和完整采样兼容性，而不是已经
证明了显著的材料质量提升。

本章仍有三项限制。第一，实验只覆盖一个条件 checkpoint 和一套目标配置，尚未证明跨属性
泛化。第二，Stable、NUS 和 E-hull 均来自 MatterSim-5M 代理势，未进行 DFT 复核。第三，
项目没有验证生成结构的真实 `dft_mag_density` 是否命中目标。因此，后续工作应在独立
条件、外部材料域和 DFT 子集上复核趋势，同时研究分字段但受约束的 scale 是否能够在不
破坏采样稳定性的前提下提高效应强度。

## 3.10 本章小结

本章提出多字段残差驱动在线 Adaptive CFG。方法对 cell、position、atom 条件残差分别
计算 RMS，聚合为当前条件强度，使用 Predictor/Corrector 隔离的 EMA 和双重限幅在线调整
一个全局 CFG scale。正式 256-seed 配对实验显示 E-hull、Stable 和 NUS 均呈正向趋势，
但未达到统计显著性。该结果建立了完整生成链的上游条件控制模块，并为后续安全后生成精修
提供 A0 输出。

## 本章图表与引用位置

- Figure 2：放在 3.5 节，用于解释真实控制器数据流与公式。
- Figure 5：放在 3.8 节，正文先解释 A 面板，再解释 B 面板。
- Table 1：实验设置可放第2章末或 3.7 节。
- Table 2：放在 3.8 节，完整指标表放附录。

## 本章不能删除的限定语

- “总体正向趋势，但配对统计未达到显著性”；
- “完整 Predictor/Corrector，不是 Corrector Gating”；
- “MatterSim-5M 代理评价，未开展 DFT 与真实目标属性验证”。
