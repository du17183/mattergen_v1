# TCL Formal256 Geometry Degradation Root-Cause Report

`DFT_VERIFIED=False`

本报告分析冻结的 C0、FT0、TCL checkpoint 及 Formal256 已有生成/弛豫结果。没有重新训练、重新生成、替换 seed 或执行新的 Formal。新增动态计算仅包括真实 checkpoint 的同状态 forward，以及 4 个真实 validation batch 的无 optimizer backward 归因。Formal 的能量、稳定性、力与弛豫仍是 MatterSim-5M/项目评价管线的代理结果，不是 DFT。

## 执行范围、结论与证据边界

- 分支：`analysis/tcl-formal256-root-cause`；父提交为冻结 Formal256 final `6ed463cfcba4f8a4864b7c603ec40dcd075f5e72`。
- Formal seeds `83000–83255` 只用于事后诊断，已经永久失去未来调参/独立 formal 资格。
- 原 Formal 生成设置了 `record_trajectories=False`，所以本报告的 stage drift 是：把 256 个已有 TCL 弛豫前结构按真实 corruption 在 10 个固定噪声点重新加噪，再对完全相同的 `x_t,t,condition` 做 C0/FT0/TCL forward。它不是重放的真实 reverse trajectory。
- score 使用 Formal 的常数 CFG=2，即 `lerp(s_uncond, s_cond, 2)`；hidden drift 使用 conditional branch 的 GemNet block 输出。
- stage/outcome 相关性是 post-hoc、未做多重检验校正的机制证据；由于 probe state 来自 TCL 终态，相关不等同于因果。

核心结论是一个组合机制，而不是单一 head 故障：

1. 当前 TCL clean-cell consistency 在高噪声 clean-state 反演时除以很小的 `alpha_t`，形成严重重尾；它在训练目标和梯度上压倒 position consistency。
2. 全模型共同使用一个 global gradient clip，cell consistency 的梯度主要流入 input/shared 表征，而不是只流入 cell head；于是原始 position/base 梯度被共同缩小，四个 GemNet block 和三字段被耦合改写。
3. 最终最能标记坏结构的是 late reverse-stage position score drift：与 RMSD、atomic force、max force、relaxation steps 的 Spearman `rho` 分别为 `0.732/0.756/0.732/0.709`。
4. TCL 相对 FT0 的收益主要是少遗忘 C0 prior、恢复 atomic/composition 多样性，并在 late geometry score/深层 hidden 上更靠近 C0；不是因为现有 position consistency 很强。
5. 碰撞和 cell distortion 都存在于部分坏例，但均不是全局单因。TCL 全体最近距离反而比 C0 更长；真正的 tail 集中在 Mn/C-rich、较致密、部分高 cell-condition 的子分布。

---

## 1. TCL 真实实现总结

TCL 的实际名称为 **Cross-Timestep Consistency Learning**。它和 FT0 都从官方本地 `dft_mag_density` checkpoint 初始化，使用相同 1024 train/128 validation、1000 optimizer steps、batch 16、AdamW、`lr=1e-4`、`weight_decay=1e-4` 和每结构两个 timestep view。

FT0 的原始目标为：

```text
L_FT0 = 0.5 * sum_view(
    1.0 L_atomic + 0.1 L_pos + 1.0 L_cell
)
```

TCL 额外使用 low-noise view 作为 stop-gradient teacher：

```text
L_TCL = L_FT0 + lambda(step) *
        (L_atomic_cons + L_pos_cons + L_cell_cons) / 3
lambda: step 1 的 0 -> step 100 的 0.1，此后保持 0.1
```

- Atomic：`KL(stopgrad(p_low(x0|xt)) || p_high(x0|xt))`。
- Position：比较两个 view 恢复的 clean fractional positions，以 periodic minimum-image MSE 计算。
- Cell：将 `x0_hat=(xt+sigma*s-(1-alpha)mu)/alpha` 标准化后比较两个 view。

TCL 会改变 atomic、position、cell 和 shared representation。它不是 adapter-only：训练代码先禁用历史实验 adapter，然后对 Lightning module 的每个参数执行 `requires_grad_(True)`。FT0/TCL 均声明 48,760,443 个模型参数可训练；相同的 83 参数 scheduler namespace 也在 optimizer scope 中，但 TCL/FT0 的 scheduler 没有梯度。训练 budget 完全相同。

理论收益是用同一 clean crystal 的低/高噪声视图约束预测一致性，减少 ordinary FT0 的噪声级别过拟合和条件分布塌缩。它对 position/cell 的 clean estimate 有几何形式的约束，但没有 pair distance、collision、force、MatterSim 或 DFT 约束；更重要的是，它没有任何 frozen C0 teacher、C0 score anchor 或 C0 representation protection。因此“cross-time 一致”不等于“保留 pretrained geometry prior”。

## 2. C0/FT0/TCL parameter drift

下表只统计去重后的真实 `named_parameters`，不重复计算 state-dict alias/buffer。`position_head` 指 GemNet `out_blocks`（包含 position-force 输出路径），`cell_head` 指 `lattice_out_blocks + mlp_rbf_lattice`。

| Category | Params | FT0 rel-L2 | FT0 cosine | TCL rel-L2 | TCL cosine | TCL drift² share |
|---|---:|---:|---:|---:|---:|---:|
| Embedding/input | 1,693,184 | 1.553% | 0.999879 | 1.207% | 0.999927 | 5.00% |
| GemNet block 1 | 5,423,107 | 3.279% | 0.999463 | 2.599% | 0.999662 | 17.58% |
| GemNet block 2 | 5,423,107 | 3.350% | 0.999439 | 2.522% | 0.999682 | 16.01% |
| GemNet block 3 | 5,423,107 | 3.398% | 0.999423 | 2.375% | 0.999718 | 14.14% |
| GemNet block 4 | 5,423,107 | 2.802% | 0.999608 | 2.035% | 0.999793 | 14.89% |
| Atomic head | 51,813 | 2.782% | 0.999613 | 2.004% | 0.999799 | 0.21% |
| Position/output head | 18,437,130 | 2.214% | 0.999755 | 1.607% | 0.999871 | 20.70% |
| Cell head | 2,667,008 | 1.690% | 0.999857 | 1.521% | 0.999884 | 5.09% |
| Condition modules | 4,198,400 | 9.681% | 0.995305 | 7.233% | 0.997381 | 6.29% |
| Other shared | 20,480 | 0.192% | 0.999998 | 0.179% | 0.999998 | 0.07% |
| **All** | **48,760,443** | **2.559%** | **0.999673** | **1.912%** | **0.999817** | **100%** |

TCL 在每一类参数上都比 FT0 更接近 C0，说明它确实起到了 regularization/less-forgetting 作用；但绝对漂移并未局限于某一 head。TCL 的四个 GemNet block 合计承担 62.7% 的 drift energy，position/output head 20.7%。Condition modules 的相对漂移最大，但不是主要绝对漂移来源。高维参数 cosine 接近 1 不能推出函数保持不变；后面的 hidden/score 差异远大于参数 cosine 所暗示的程度。

完整数据：`parameter_drift.csv`。

## 3. Field-level score drift

在全部 256 个已有 TCL 初始结构、相同 corruption noise、相同条件和 CFG=2 上，TCL−C0 的 stage 平均为：

| Field | Reverse stage | Mean abs/RMS drift | Relative L2 | Mean cosine | Norm ratio TCL/C0 |
|---|---|---:|---:|---:|---:|
| Atomic logits | Early | 2.765 RMS | 0.222 | 0.976 | 0.971 |
| Atomic logits | Mid | 2.378 RMS | 0.196 | 0.983 | 0.997 |
| Atomic logits | Late | 1.749 RMS | 0.159 | 0.992 | 1.066 |
| Position | Early | 0.00583 RMS | 1.867 | 0.460 | 1.976 |
| Position | Mid | 0.14570 RMS | 0.651 | 0.926 | 1.381 |
| Position | Late | 0.18643 RMS | 0.197 | 0.973 | 0.959 |
| Cell | Early | 0.06670 RMS | 0.074 | 0.997 | 0.984 |
| Cell | Mid | 0.08211 RMS | 0.092 | 0.995 | 0.986 |
| Cell | Late | 0.10351 RMS | 0.111 | 0.989 | 0.999 |

Early position 的 relative L2 很大，是因为高噪声端 C0 position score norm 很小；其绝对 RMS 只有 0.00583，不能按 relative 数值单独判为最危险。到 late stage，position 的绝对 drift 增大到 0.186，且具有最强 outcome correlation。

相对 FT0，TCL 在 early atomic drift（0.222 vs 0.264）、late position（0.197 vs 0.215）、late cell（0.111 vs 0.128）和 late h3/h4 上更接近 C0；但 TCL 的 mid position relative drift（0.651）比 FT0（0.365）更大。这说明 TCL 的收益不是三字段在所有噪声段一致改善。

## 4. Stage-level score drift

forward diffusion 定义是 `t=0` 为 clean/低噪声，`t=1` 为 prior/高噪声。报告中的 reverse sampling progress 定义为 `p=1-t`：

- 0% sampling progress：高噪声，刚从 prior 开始。
- 30–70%：中段。
- 100%：低噪声，接近最终结构。

TCL−C0 十个 10% bin 的 relative L2 如下；括号是 position absolute RMS，防止高噪声小分母误导。

| Progress center | t | Atomic | Position | Cell |
|---:|---:|---:|---:|---:|
| 5% | .95 | .213 | .528 (.0071) | .075 |
| 15% | .85 | .222 | 2.317 (.0048) | .072 |
| 25% | .75 | .232 | 2.756 (.0056) | .075 |
| 35% | .65 | .233 | 1.695 (.0477) | .082 |
| 45% | .55 | .214 | .485 (.1840) | .108 |
| 55% | .45 | .176 | .267 (.2082) | .095 |
| 65% | .35 | .159 | .158 (.1429) | .082 |
| 75% | .25 | .156 | .152 (.1404) | .074 |
| 85% | .15 | .155 | .187 (.1711) | .088 |
| 95% | .05 | .165 | .251 (.2478) | .171 |

结论不是“early position 最差”，而是：early relative angle 已偏离但绝对 score 很小；真正同时具备较大绝对 drift、末端直接几何作用和坏结果预测力的是 late position。Cell 在最后 10% 也明显上升，需要中等强度保护。

## 5. GemNet hidden drift

Conditional hidden representation 的 TCL−C0 平均 relative L2/cosine：

| Block output | Early | Mid | Late |
|---|---:|---:|---:|
| h1 | .248 / .972 | .249 / .971 | .232 / .976 |
| h2 | .401 / .922 | .480 / .896 | .399 / .926 |
| h3 | .599 / .813 | .609 / .826 | .457 / .905 |
| h4 | .679 / .742 | .533 / .844 | .313 / .950 |

偏离从 h1 已经可见，在 high-noise early stage 沿 h1→h4 放大；mid stage 在 h3 达峰；late stage h4 有部分恢复，但 h2/h3 仍显著偏离。坏结果不是“只改了最终 head”。TCL 的 late h3/h4 比 FT0 更接近 C0（relative L2 分别低约 13%/14%），与 TCL 相对 FT0 的整体恢复一致。

## 6. Atomic / position / cell attribution

| Hypothesis | 证据等级 | 判断 |
|---|---|---|
| A. Position branch drift → local geometry failure | **strongly supported** | Late position drift 对 RMSD/max force/atomic force/steps 的 `rho=.732/.732/.756/.709`；坏例 late-position rel drift 0.355，正常 0.176。 |
| B. Cell branch drift → lattice/geometry failure | **moderately supported** | 当前 cell consistency 主导训练梯度；TCL 坏例 cell condition mean 3.43 vs 正常 2.27，5/30 坏例 condition>5。但全体 cell score drift 较小，部分高 RMSD 样本无新增 cell 异常。 |
| C. Atomic branch → composition/basin change | **moderately supported** | TCL composition 明显偏向 Mn/C；坏例 Mn/C 原子占比 49.3%/13.3%，正常为 24.0%/2.87%；early atomic drift 与 max force/RMSD/steps 的 `rho=.355/.292/.379`。但 composition change 几乎普遍发生，不能单独解释所有 tail。 |
| D. Shared representation coupling | **strongly supported** | 全模型训练；62.7% 参数 drift² 在四个 blocks；cell 梯度主要进入 embedding/shared；h2–h4 大幅偏离。 |

最合理的因果链是 `ill-conditioned cell TCL objective -> shared-gradient domination/global clipping -> hidden/field coupling -> late position drift`，再与 atomic composition shift 和特定 cell/local-coordination 状态相互作用。

## 7. Formal bad-seed case studies

TCL severe union 有 30/256：RMSD>0.5 为 10，max-force>1 为 24，max-force>2 为 5，steps>400 为 8。以下是六个代表性 paired cases；`V/atom` 单位 Å³，距离/RMSD 单位 Å，force 单位 eV/Å。

| Seed | Method | Formula | dmin | V/atom | Cell cond | MaxF | RMSD | Steps | E-hull |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 83043 | C0 | Au2Eu12Ir | 2.837 | 38.01 | 1.88 | .090 | .073 | 67 | .019 |
| 83043 | FT0 | C4Mn11 | 1.965 | 9.98 | 1.46 | .645 | .065 | 120 | .131 |
| 83043 | TCL | C8Eu7 | **1.232** | 18.40 | 1.69 | **5.598** | .089 | 90 | .162 |
| 83069 | C0 | Eu3Ge7Mn5 | 2.470 | 19.29 | 4.58 | .508 | .112 | 98 | .183 |
| 83069 | FT0 | C5Mn10 | 1.880 | 9.88 | 1.90 | .561 | .176 | 243 | .204 |
| 83069 | TCL | Eu4Ge5Mn6 | 2.269 | 20.17 | 4.54 | .762 | **1.367** | 490 | .320 |
| 83077 | C0 | HFe3GaMn2O7 | .998 | 12.18 | 3.06 | .611 | .055 | 95 | .092 |
| 83077 | FT0 | C3Mn11 | 1.919 | 9.76 | 2.59 | .597 | .049 | 94 | .158 |
| 83077 | TCL | EuFe2Mn3O8 | 1.898 | 11.69 | **7.51** | **3.307** | .070 | 81 | .276 |
| 83130 | C0 | Mn9N4 | 1.795 | 9.91 | 3.34 | .691 | .059 | 57 | .147 |
| 83130 | FT0 | C4Mn9 | 1.942 | 9.92 | 2.01 | .855 | .090 | 84 | .217 |
| 83130 | TCL | Mn13 | 2.270 | 11.52 | 1.95 | .505 | **1.182** | 494 | .165 |
| 83138 | C0 | CrGdLi3Ni2O11V2 | 1.725 | 11.27 | 2.05 | .716 | .075 | 125 | .269 |
| 83138 | FT0 | C7Mn13 | 1.832 | 9.33 | 1.27 | 1.367 | .106 | 181 | .208 |
| 83138 | TCL | CFe4Mn15 | 1.730 | 11.23 | 1.06 | **2.456** | **.585** | 655 | .144 |
| 83203 | C0 | B3Fe3Ni10 | 1.764 | 10.13 | 3.44 | .495 | .037 | 83 | .170 |
| 83203 | FT0 | C4Mn12 | 1.848 | 9.96 | 1.65 | .743 | .057 | 56 | .145 |
| 83203 | TCL | B2Gd4Pd10 | 2.042 | 18.51 | 2.02 | **2.379** | **.715** | **1321** | .470 |

结构性错误不是一种：83043 是直接 short-contact/high-force；83077 是高 cell condition 的 lattice/local-force 问题；83130 是 pure-Mn composition/basin 大位移；83069 在与 C0 类似的 elongated cell 下仍产生大 RMSD，说明 cell 不是充分解释；83138 是 Mn/C-rich dense coordination tail；83203 无短碰撞却 force、RMSD、relaxation 同时失败，属于 species/local basin mismatch。

全部 30 条 TCL bad cases 见 `tcl_bad_seed_case_studies.csv`；三方法全部描述量见 `structural_diagnostics.csv`。

## 8. Composition-change analysis

三方法每组都有相同的 2475 个总原子，且 paired seed 的 atom count 在 C0/TCL 间 256/256 相同；但是 atomic identity 基本都变了：

| Subset | N | Same atom count | Same element set | Same stoichiometry | Mean composition-fraction L1 | Element-set Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| All TCL vs C0 | 256 | 256 | 7 | 3 | 1.371 | .241 |
| TCL severe | 30 | 30 | 1 | 0 | 1.507 | .217 |
| TCL normal | 226 | 226 | 6 | 3 | 1.354 | .244 |

全体原子分布显示 ordinary FT0 严重塌缩，TCL 只部分恢复 C0：

| Method/group | Element entropy | Mn | C | O | Eu |
|---|---:|---:|---:|---:|---:|
| C0 all | 2.823 | 5.86% | .81% | 19.27% | 17.41% |
| FT0 all | 1.371 | 62.83% | 20.44% | .16% | 2.75% |
| TCL all | 2.670 | 28.28% | 4.65% | 2.55% | 19.88% |
| TCL severe | 1.964 | **49.29%** | **13.33%** | 5.71% | 3.10% |
| TCL normal | 2.716 | 23.99% | 2.87% | 1.90% | 23.31% |

这既解释了 TCL>FT0 的 NUS/Unique/quality recovery，也说明 atomic path 是 geometry tail 的间接来源：坏例明显向 Mn/C-rich dense basin 富集。不过“是否 change”本身没有区分力，因为 253/256 TCL 本来就与 C0 stoichiometry 不同；有区分力的是 change 的方向和程度。

## 9. Minimum-distance / collision analysis

弛豫前 minimum periodic distance：

| Method | P1 | P5 | Median | Mean | dmin<1.5 count |
|---|---:|---:|---:|---:|---:|
| C0 | 1.187 | 1.637 | 2.361 | 2.328 | 12 |
| FT0 | 1.513 | 1.781 | 1.967 | 2.169 | 3 |
| TCL | 1.588 | 1.786 | 2.460 | 2.534 | 3 |

TCL−C0 mean distance为 `+0.207 Å`，paired bootstrap 95% CI `[+0.136,+0.278]`。因此不支持“全局 TCL collision collapse”。但是在各方法内部，短距离都是强风险因子：dmin 对 max force 的 `rho` 为 C0 `-.654`、FT0 `-.709`、TCL `-.725`；对 TCL RMSD/steps 为 `-.511/-.596`。TCL severe 的 dmin mean 1.894，normal 为 2.619。结论是 collision 能解释一部分 high-force bad seeds，但不能解释 83130/83203 等无短接触的大 RMSD/长弛豫样本。

## 10. Cell geometry analysis

| Method/group | V/atom mean | V/atom median | Cell cond mean | cond P95/P99/max | Aspect ratio mean |
|---|---:|---:|---:|---:|---:|
| C0 all | 18.05 | 14.64 | 2.37 | 4.76/6.03/7.17 | 1.86 |
| FT0 all | 13.38 | 10.91 | 2.15 | 3.24/4.95/6.65 | 1.70 |
| TCL all | 20.52 | 19.14 | 2.41 | 5.11/8.77/12.10 | 1.92 |
| TCL severe | 12.41 | 11.36 | **3.43** | 11.26/12.08/12.10 | **2.62** |
| TCL normal | 21.60 | 22.34 | 2.27 | 4.63/6.67/6.90 | 1.83 |

TCL 全体并非过小 cell；它相对 C0 的 V/atom mean `+2.47 Å³`，95% CI `[+1.27,+3.70]`。但坏例落在 TCL 自身的低体积/高 condition 子群：30 个 severe 中 10 个 V/atom<10、5 个 condition>5、5 个 aspect>5。Cell distortion 是 tail amplifier，尤其是 83077 一类；它不是全局均值退化的唯一来源。

长度/角度/行列式也支持同一结论。C0/FT0/TCL 的 median `a/b/c` 分别为 `4.31/5.21/6.86`、`4.09/4.93/6.03`、`4.34/5.45/7.28 Å`；TCL 并非整体 cell collapse。三组 determinant 均为正，TCL minimum `36.09 Å³`，没有 near-singular/negative-volume 全局故障。但 TCL bad 的 `c` P99 达 `22.76 Å`，角度范围 `74.67–125.85°`；seed 83077 的 `a/b/c=3.30/3.31/17.42 Å`、`alpha/beta/gamma=90.12/89.97/120.44°`、condition=7.51，是明确的个例 cell distortion。Seed 83069 的 elongated cell 则与 C0 已有形态相近，说明异常 cell 不是大 RMSD 的必要条件。

## 11. Trajectory drift vs final RMSD/force correlation

由于 Formal 没有保存真实 trajectory，下表是 matched-state forward proxy。数值为 TCL−C0 relative score drift 与 TCL 最终结果的 Spearman `rho`：

| Field/stage | RMSD | Atomic force | Max force | Relax steps | Severe |
|---|---:|---:|---:|---:|---:|
| Atomic early | .292 | .354 | .355 | .379 | <.15 |
| Atomic late | <.15 | .208 | .204 | .176 | .211 |
| Position early | .197 | .182 | .192 | .196 | <.15 |
| Position mid | -.190 | -.213 | -.253 | -.369 | <.15 |
| **Position late** | **.732** | **.756** | **.732** | **.709** | **.442** |
| Cell early | -.415 | -.417 | -.437 | -.554 | -.267 |
| Cell late | <.15 | <.15 | <.15 | <.15 | .157 |

Early-cell 的负相关不是“cell drift 有保护作用”的因果证据；它与 probe 结构的体积/密度共同变化，且 state 是终态再加噪。可稳健使用的结论是相对排序：late position 远强于其它正向关联。真正证明 1000-step amplification 仍需要未来 development seeds 保存 trajectory；不能用本报告声称已经完成了真实轨迹因果验证。

## 12. Late-position-drift hypothesis: supported

**结论：supported，而且是当前最强的 field×stage outcome marker。**

- Severe TCL 的 late-position relative drift mean `0.355`，normal 为 `0.176`，约 2.02 倍。
- Late-position 与 RMSD/max force 的 `rho=.732/.732`，与 atomic force/steps 为 `.756/.709`。
- 对照项明显较弱：mid position 为负相关，late cell 与连续 geometry outcome 的 `|rho|<.15`，atomic 最大约 `.38`。

限制：终态 state 可能同时造成“模型分歧更大”和“最终指标更差”，所以这里证明的是强预测关联，不是单向因果。未来若实现 anchor，应在新 development trajectories 上确认，而不是继续复用 Formal seeds。

## 13. Geometry degradation 最可能机制

现有训练曲线给出了比参数 norm 更直接的根因证据：

| Quantity | All 1000-step mean | Last-50 mean | Tail/max |
|---|---:|---:|---:|
| Base loss | .2960 | .2958 | — |
| Weighted atomic TCL contribution | .05735 | .06038 | — |
| Weighted position TCL contribution | **.00191** | .00205 | — |
| Weighted cell TCL contribution | **.56094** | .30880 | cell-cons P99 213.34, max 1621.31 |
| Pre-clip gradient norm | 95.84 | 37.15 | max 16642.48 |

TCL 的 1000/1000 steps pre-clip gradient norm>1；FT0 为 94.4%，但 FT0 mean/median 仅 2.29/2.00，而 TCL 为 95.84/27.50。Cell consistency 与 `t_high_mean` 的 `rho=.379`，符合 `x0_hat` 除以小 `alpha_t` 在高噪声放大的预期。

4 个真实 validation batch、冻结 TCL、无 optimizer 的归因进一步显示：

| Weighted objective | Loss mean | Gradient L2 mean | Cosine with base gradient |
|---|---:|---:|---:|
| Original MatterGen base | .3724 | 2.405 | 1.000 |
| Atomic consistency | .0729 | .356 | .226 |
| Position consistency | .00217 | **.00465** | -.026 |
| Cell consistency | .4414 | **66.24** | .135 |
| Total | .8888 | 66.57 | .185 |

Cell gradient约为 position 的 14,260 倍、base 的 27.5 倍；其 gradient² 平均约 59% 在 embedding/input、26% 在 other shared basis、6.4% 在 cell head。Global clip 后，position head 的 base gradient 也被同一总 norm 缩小。最可能机制因此是：**量纲/反演条件不良的 cell consistency 主导共享更新，position consistency 实际没有足够梯度保护，最终在 low-noise reverse steps 暴露为 position drift 和结构 tail。**

## 14. TCL quality benefit 最可能来自哪里

证据更支持“保留 C0 prior”而非“学到更强 geometry”：

- TCL 的总参数 drift 比 FT0 小 25.3%（L2 11.54 vs 15.44），每个参数类别都更接近 C0。
- TCL 将 FT0 的 composition entropy 1.371 恢复到 2.670，接近 C0 的 2.823；Mn/C collapse 从 FT0 的 62.8%/20.4% 降到 28.3%/4.65%。
- 相对 FT0，TCL 的 early atomic、late position/cell、late h3/h4 更接近 C0。
- Formal 中 TCL 相对 FT0 的 E-hull/NUS/force 等优势复现，但 TCL 没有胜过 C0；这正符合“less forgetting”而不是“新 optimum 超过 teacher”。

最可能 benefit source 是 **atomic/composition diversity recovery + deep shared representation 更靠近 C0**。当前证据不支持把收益归因于 position TCL 分支；它的实际加权梯度太小。Cell-dominated clipping可能产生隐式 regularization，但它同时是训练不稳定和 field coupling 的来源，不能作为值得保留的设计机制。

## 15. Catastrophic forgetting 是否成立

结论：**部分成立，但不是经典的全面 catastrophic forgetting。**

- FT0 是强 forgetting：condition module rel drift 9.68%，composition 严重 Mn/C collapse，Formal quality/geometry 全面弱于 C0。
- TCL 明显减轻而没有消除 forgetting：参数更接近 C0，但 hidden、CFG score、composition 和 Formal geometry tail 仍偏离。
- Objective conflict：**strongly supported**，cell/position TCL 项相差多个数量级且 cell 与 base gradient cosine 仅 .135。
- Field coupling：**strongly supported**，cell gradient主要进入 shared/input，late position failure association 最强。
- Sampling amplification：**plausible/moderately supported**，1000-step sampler 可累积小 drift，但缺少原始 trajectory，不能直接建立。

最准确分类是 `partial forgetting + objective-scale conflict + shared-field coupling + likely sampling amplification`。

## 16. C0 score anchor 是否有依据

**有直接依据，且比先加 force loss 更匹配当前 failure。** C0 在 Formal 的 geometry/force prior 明显优于 TCL；TCL 的质量收益本身主要来自靠近 C0。Teacher anchor 不需要 MatterSim 标签，也不会把 evaluator 直接泄漏进训练。

建议按字段区分：

- Position：最强，尤其 reverse late stage；使用 dimensionless relative score anchor。
- Cell：中等；用 score/metric anchor 替代当前高噪声 `x0/alpha` consistency，不在高噪声端强行反演 clean cell。
- Atomic：弱 anchor 或 C0-logit KL，防止 FT0 式 composition collapse，但保留一定改变 species 的自由度。

不建议三字段未归一化等权 L2，也不建议把 `lambda_atomic=lambda_pos=lambda_cell` 当成公平。

## 17. Field-selective protection 是否有依据

**有依据。** GemNet 的显式 heads 可区分：`fc_atom`、`out_blocks`、`lattice_out_blocks`；可以冻结 position/cell heads 和 C0 backbone，只训练 condition modules、atomic head或新建 branch-specific residual。

但 existing `cond_adapt_layers/cond_mixin_layers` 在每个 message-passing block 前改变 `h`，即使 position/cell heads frozen，geometry score仍会变化。因此真正 isolation 有两种级别：

1. 软 isolation：冻结 backbone/geometry heads，只训练 condition modules+atomic head，并用 C0 pos/cell anchor约束其经 shared h 的间接影响。
2. 硬 isolation：冻结完整 C0 denoiser，在 final C0 node representation 后新增只连接 atomic logits 的 residual adapter；position/cell score在相同 state 上严格等于 C0。

未来实现会涉及 `mattergen/adapter.py`/`mattergen/denoiser.py` 的 branch output、一个新的 training objective 文件和实验配置；不应修改冻结 Formal 生成/评价脚本。

## 18. Stage-aware anchor 是否有依据

**有依据，但应基于 reverse progress，而不是把 t 方向写反。** 推荐：

```text
reverse progress 0–30% (t≈1→.7): position weak, cell weak/不做 clean-x0 anchor
30–70% (t≈.7→.3): position medium
70–100% (t≈.3→0): position strong, cell medium
```

依据是 late position 的 outcome correlation和最后 10% cell drift 上升。Early position 虽 relative drift 大，absolute score 极小；在这里强 anchor 可能浪费 capacity并抑制 atomic/global exploration。Stage-aware 不是独立充分修复：必须同时移除/归一化当前 cell consistency 的高噪声梯度尖峰。

## 19. Physics/force loss 是否值得

当前不应作为第一步。Loss 候选比较：

| Loss | Stability | Cost | Leakage/bias | 与本 failure 匹配 |
|---|---|---|---|---|
| C0 score anchor | 高（归一化后） | teacher forward约翻倍 | 无 MatterSim evaluator leakage | **最高**，直接保护已验证 prior |
| Pair-distance consistency | 中高 | 低–中 | 无 evaluator leakage | 适合局部几何，但全局 collision 没有恶化 |
| Cell metric `G=L^T L` consistency | 中 | 低 | 无 evaluator leakage | 可处理旋转不变 cell tail，次级 |
| Local collision penalty | 中高 | 低 | 阈值/元素半径偏置 | 只覆盖少数 short-contact cases |
| Force-aware surrogate | 中低 | 很高，需对坐标反传 | 受 surrogate domain bias | 可捕获局部物理，但不是首要根因 |

若 score-anchor P1 已恢复大部分 geometry，就没有理由增加 force loss。只有新方法仍出现“无短碰撞但高 force”的独立 dev tail，才进入 Route D；应使用 CHGNet、MACE 或其它独立于最终 MatterSim-5M 的 surrogate，并继续明确它不是 DFT。训练和最终评价都用 MatterSim 会构成 evaluator leakage/过拟合风险，因此不采用。

## 20. DML 是否匹配当前 failure

DML 的实际名称为 **Dynamic Multi-field Loss**。它用 `Linear(1,16)-SiLU-Linear(16,3)` 根据 normalized t 输出 `[0.5,1.5]` field multipliers，再在保持总 base weight sum 的条件下动态重加权原始 atomic/position/cell losses；它不含 cross-time consistency、C0 anchor、pair/force geometry loss。

学到的方向是噪声越高越降低 position（mean multiplier .871；高噪声 bin .847），越提高 cell（mean 1.123；高噪声 1.146），与本报告发现的“cell 已主导、position 未保护”不匹配。DML P0 只有 8 seeds 且为明确负结果：E-hull 0.1676 vs FT0 0.1100，Stable 37.5% vs 62.5%，NUS相同，force更差；offline fixed loss也差于 FT0。没有独立 positive evidence。

DML 没有 MatterSim training leakage；风险来自 end-to-end 学习同一训练 loss 权重、1024 样本上的 objective overfit 和错误 field allocation。TCL+DML 技术上可组合，但现有 DML 会进一步上调 cell/下调 position，最可能加重而不是修复 failure。**不推荐 DML alone，也不推荐 TCL+DML。**

## 21. Route A–G 比较表

评分 1–5，所有列均为“越高越有利”；Ease=实现容易，Stability=训练稳定，LowEng=工程量低，LowRisk=失败风险低。

| Rank/Route | Match | Geometry | Keep benefit | Ease | Stability | LowEng | Protect C0 | Explain | Thesis novelty | LowRisk | Total/50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. A C0 score-anchored TCL | 5 | 4 | 4 | 4 | 4 | 4 | 5 | 5 | 4 | 4 | **43** |
| 2. B Field-selective TCL | 5 | 5 | 3 | 3 | 5 | 3 | 5 | 5 | 4 | 4 | **42** |
| 3. C Stage-aware geometry anchor | 4 | 5 | 4 | 3 | 3 | 3 | 5 | 4 | 5 | 3 | **39** |
| 4=. D Physics/geometry-constrained TCL | 3 | 4 | 3 | 2 | 2 | 2 | 3 | 3 | 4 | 2 | **28** |
| 4=. E DML alone | 1 | 2 | 2 | 5 | 4 | 5 | 2 | 4 | 2 | 1 | **28** |
| 6. G Stop TCL/new architecture | 2 | 3 | 1 | 1 | 3 | 1 | 4 | 2 | 4 | 2 | **23** |
| 7. F TCL+DML | 2 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 3 | 1 | **21** |

Route C 的 stage schedule 是 Route A 很有价值的组件，但单独加 schedule不能消除 cell gradient pathology。Route E 得分来自易实现/工程量低，不代表科学推荐。

## 22. 第一推荐路线

第一推荐是 **Route A：Gradient-Balanced Stage-Aware C0-Anchored TCL（GBSA-TCL）**。核心不是在当前 loss 上机械再加一项，而是：

1. 去掉当前不稳定的 high-noise clean-cell `1/alpha` consistency，或改为有界/归一化的 cell score anchor。
2. 保留 atomic cross-timestep consistency，position consistency 必须按梯度/尺度归一化。
3. 用 frozen C0 teacher 对 position/cell做 field- and stage-aware anchor；atomic 使用较弱 KL anchor。
4. 冻结大部分 pretrained trunk，避免再次全量改写 48.8M 参数。

它最直接对应已观察到的根因，同时仍允许 TCL atomic/quality signal存在。

## 23. 第二推荐路线

如果第一推荐在全新 32-seed P1 没有同时恢复 geometry并保留 quality，第二推荐切到 **Route B：strict atomic-only TCL residual adapter**：完整 C0 backbone、position head、cell head全部冻结，只在 final node embedding 到 atomic logits 之间增加轻量 residual/gate，并只训练 atomic original loss、atomic TCL 和弱 C0 atomic KL。

该路线牺牲部分 geometry/quality协同的潜力，换取同状态下 position/cell score严格保持 C0。它是对“shared field coupling”最干净、最可解释的消融/备选，不与第一路线连续调参混在一起。

## 24. 明确不推荐路线

- 不推荐当前 TCL 原样继续、加训练步数或再做一个 Formal；它已有明确 Formal FAIL 和 objective pathology。
- 不推荐 DML alone；P0 已负，learned routing方向与 root cause相反。
- 不推荐 TCL+DML；没有 positive evidence，最可能把 cell dominance进一步强化。
- 不推荐先上 MatterSim force loss；会把评价器泄漏到训练，且没有先验证更直接的 C0 anchor。
- 不推荐立即切大型 Transformer/Self-Conditioning Route G；当前 failure 已有可定位、低成本的修复路径。若 A/B 都在独立 P1 失败，再停止 TCL 家族并重开模型创新路线。

## 25. 第一推荐的具体模型结构

```text
same real x_t, t, condition/mask
          │
          ├──────── Frozen C0 teacher (eval, no grad)
          │              ├─ atomic logits
          │              ├─ position score
          │              └─ cell score
          │
          └──────── Student initialized from C0
                         ├─ frozen input embedding/noise encoder
                         ├─ frozen GemNet blocks 1–4
                         ├─ trainable condition adapter/mixin (≈4.20M)
                         ├─ trainable atomic head (≈0.052M)
                         ├─ frozen position/output head
                         └─ frozen cell/lattice head
```

Condition adapter仍能通过 frozen blocks 改变 hidden，因此 frozen geometry heads不保证 score相等；pos/cell teacher anchors仍然必要。P0 可先把 trainable scope限制在约 4.25M（原模型 8.7%），不要直接解冻 blocks。若这个 scope完全没有 quality signal，才在下一条独立开发配置中考虑只解冻 block4，并保持 anchor；不能在 Formal seeds 上逐步试。

## 26. 第一推荐的 loss

建议以每结构、dimensionless reduction定义：

```text
L = L_MatterGen_original
  + lambda_TCL_atomic * KL(stopgrad(p_low) || p_high)
  + lambda_TCL_pos * normalized_periodic_position_consistency
  + lambda_atomic(t) * KL(p_student || p_C0)
  + lambda_pos(t) * ||s_pos,student-s_pos,C0||² / (||s_pos,C0||²+eps)
  + lambda_cell(t) * ||s_cell,student-s_cell,C0||² / (||s_cell,C0||²+eps)
```

约束：

- 不保留当前无界 clean-cell `1/alpha` consistency；cell teacher anchor直接在相同 noisy state/score parameterization比较。
- `lambda_pos(t)` 在 forward `t<=.3` 最强，`.3<t<.7` 中等，`t>=.7` 弱。
- `lambda_cell(t)` late中等、其它弱；`lambda_atomic(t)` 全程弱，避免消灭 composition flexibility。
- 每个 loss branch先做 EMA/RMS normalization，再合成；记录各 branch gradient norm，禁止任一辅助项长期超过 base gradient 3 倍。
- Teacher和 student使用相同 `x_t,t` 及相同 conditional/unconditional mask。无需 MatterSim/force标签。

初始 lambda只能在新的 development split/seeds上冻结，不能由 Formal256 tail调到“刚好通过”。

## 27. 参数冻结策略

P0 首选：冻结 atom/input embedding、noise encoder、GemNet blocks1–4、position/output heads、cell heads；只训练官方 condition adapter/mixin和 atomic head。这样阻断本次最危险的 cell-gradient→input/shared 权重更新路径，同时保留条件适配 capacity。

如果 condition adapter仍引起不可接受的 geometry score drift，直接切第二推荐的 hard atomic-only residual，不继续解冻。只有 P0 已显示 quality signal且 anchor稳定时，才允许一个预注册版本解冻 block4；不得同时解冻多个 block并做宽搜索。

## 28. 预计训练成本

- Teacher：每个 two-view batch增加两个 frozen C0 forward，无 teacher backward；预计 wall time约为当前 TCL 的 1.6–2.0 倍。
- Student optimizer state：trainable params从48.76M降到约4.25M，显著降低 optimizer/gradient memory；teacher activation无需保留反向图。
- 训练显存预计不高于当前 full-TCL太多，H20 可轻松承载；应以 P0 实测为准，不提前宣称精确比例。
- 推理/生成：不加载 teacher，只用 student；现有 condition adapter已经在模型内，几乎不增加 Formal 采样 forward数。若使用新小 residual，开销应远低于1%。

## 29. P0/P1/P2/Formal 计划

所有阶段必须使用从未用于 Formal256 的新 seeds；`83000–83255` 永久排除。

1. **P0（4–8 paired seeds）**：先做真实 checkpoint训练和生成 smoke。重点不是显著性，而是辅助梯度不再支配、C0 late pos/cell anchor生效、无 NaN、MatterSim tail不恶化。只允许一个预注册主配置和至多一个明确失败分析配置。
2. **P1（32 new paired seeds）**：必须同时生成/评价 `C0 / FT0 / frozen TCL / GBSA-TCL`，相同 seeds、采样和独立 MatterSim管线。主问题是是否保留 TCL>FT0 quality benefit并修复 TCL<C0 geometry gap。
3. **P2（64 independent paired seeds）**：仅在 P1 gate全部通过后，冻结 checkpoint/hyperparameters，在全新 seeds复现。继续保留全部 outliers。
4. **Formal256**：只有 P2 CLEAR GO、方法完全冻结后，才使用另一组从未观察的256 seeds；Formal开始后不再改模型、loss、阈值或替换样本。

未来 development run应保存低成本 trajectory diagnostics；正式评价仍报告 `DFT_VERIFIED=False`，除非真正执行独立 DFT。

## 30. 明确停止条件

P0 立即停止条件：

- NaN/Inf、checkpoint无法真实生成，或任一 auxiliary branch gradient持续超过 base 3倍；
- 又出现接近当前 TCL 的全步 global clipping/cell heavy tail；
- anchor loss下降但 student quality signal消失为机械复制 C0，且 atomic/TCL分支无有效梯度；
- 4–8 seeds中 geometry tail明显多于 C0/TCL。

P1 进入 P2 的最低冻结门槛：

- Paired mean RMSD 比 TCL 至少低 `0.01 Å` 且 bootstrap 95% upper bound `<0`；
- atomic-force mean ratio new/TCL `<=0.90` 且 95% upper bound `<1`；
- structure max-force mean ratio new/TCL `<=0.90`，同时 `max-force>1` 与 `>2` count均少于 TCL，不能新增 >400-step tail；
- E-hull：new−TCL upper CI `<+0.025 eV/atom`；Stable/NUS 的 new−TCL lower CI均 `>-10 pp`；三项 point estimate不得退回或差于 FT0；
- Novel/Unique无明显 collapse。

只要 RMSD、atomic force、max-force tail 任一没有明确恢复，或 quality回到 FT0 水平，**直接停止，不进入64 seeds**。P2若不独立复现、出现新的 robustness tail或需要再次改 lambda，也停止该方法并失去当前 formal资格。A与B都失败后停止 TCL 家族，才考虑 Route G；不通过追加审计、更多同分布 seeds或第二个 Formal挽救负结果。

## 机器可读结果

- `parameter_drift.csv`：去重参数的模块级 L2/relative/cosine。
- `score_hidden_drift_per_seed_bin.csv`、`score_hidden_drift_by_10pct_bin.csv`、`score_hidden_drift_by_stage.csv`：真实 checkpoint matched-state CFG score/hidden漂移。
- `score_hidden_outcome_correlations.csv`：field×stage与最终结果的探索性相关。
- `gradient_attribution.csv`、`gradient_attribution_summary.csv`：4个真实 validation batch 的无更新梯度归因。
- `structural_diagnostics.csv`、`structural_summary.csv`、`structure_outcome_correlations.csv`：已有 Formal结构的距离、邻居、晶胞、force和结果。
- `paired_composition_changes.csv`、`composition_summary.csv`、`element_frequencies.csv`：paired composition及 bad/normal 分层。
- `tcl_bad_seed_case_studies.csv`：全部30个TCL severe seed。

最终科研判定：当前 TCL 相对 FT0 的优点是真实且可复现，但其实现把一个高噪声、尺度失衡的 cell clean-state consistency 通过全量共享网络和 global clipping扩散到三字段；最终在 late position score 上形成与 geometry/force tail强相关的偏离。最值得验证的修复不是更多 full finetuning、DML或MatterSim force fitting，而是受控、归一化、field/stage-aware 的 frozen-C0 score protection；若它不能在全新32 seeds同时恢复 geometry并保留quality，停止该路线。
