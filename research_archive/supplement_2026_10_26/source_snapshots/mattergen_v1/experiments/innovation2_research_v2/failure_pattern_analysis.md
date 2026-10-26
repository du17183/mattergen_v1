# Innovation 2 历史失败模式审计

审计日期：2026-09-12。主仓库：`/mnt/datasets-livsyn/dxl/mattergen_v1`，审计起点 `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`。本轮只读代码、报告、远端跟踪分支、5 个 stash 以及既有轨迹；没有训练、生成或改动历史实验。

## 结论先行

历史结果不支持继续押注“更聪明的通用控制器”。失败路线的共同问题是：用弱代理信号覆盖成熟生成先验、在小样本上把均值改善误判为稳健改善，以及用共同的时间步趋势冒充字段依赖。唯一经 Formal256 确认的 E3-PCR 成功，是因为它解决了一个窄而可观测的问题：生成结束后直接降低预松弛力，并用门控减少有害干预。

剩余最可信的缺口不是 magnetic-density 再优化，而是 MatterGen 的原子 D3PM 反向转移没有全局价态/电中性可行性。E3-PCR Formal256 的 256 个 C0 样本全部通过几何有效性，但只有 202 个通过 SMACT 组成有效性，即 54/256（21.094%）是“geometry-valid but chemistry-invalid”。这是大样本、直接、尚未被 E3-PCR 修复的错误源。

## 证据层级

- Formal256：可作主要判断，包括 Innovation 1、E3-PCR 和 Corrector residual distillation。
- independent/prospective 32–64：可作方向判断，不能替代 Formal256。
- 4–16 样本 smoke：只用于杀死明显错误路线，不用于宣称成功。
- stash 中实验：只读取第三父提交中的报告/指标，不 apply、不 pop；它们仍是可复核的负证据。
- 64 条 shared-field trajectories：32 calibration + 32 validation，每条 55 个 observation，共 3,520 行；只做预先分开的相关诊断。

## 主要审计来源索引

- Innovation 1 Formal256：`remotes/origin/archive/thesis-analysis-package-v1:thesis_archive/reports/innovation1/formal_final_report.md`。
- E3-PCR Formal256：`remotes/origin/feature/q3-e3-pcr-formal256:reports/q3_e3_pcr/formal256/final_report.md` 与 `final_summary.json`。
- Corrector distillation：`experiments/corrector_residual_distillation_formal256/final_report.md` 及同族 v1/v2/v3 reports。
- Adapter/interaction：`experiments/distribution_constrained_quality_adapter_p0/`、`p1/`、`p1b/`、`distribution_balanced_quality_adapter/`、`global_transformer_adapter_p0/`、`cross_field_interaction_p0/`、`p1/`。
- GBSA/TCL：`experiments/gbsa_tcl_p0/`、`p1/`、`prospective32/`、`tcl_p1/`、`p2/`、`tcl_dml_p0/`、`tcl_formal256/` 下的 `final_report.md`。
- Adaptive neighborhood：`/mnt/datasets-livsyn/dxl/mattergen_v1_adaptive_neighbor/experiments/adaptive_neighborhood_p0/final_report.md`。
- Self-conditioning：`/mnt/datasets-livsyn/dxl/mattergen_v1_self_condition/experiments/multifield_self_conditioning_p0/final_report.md`。
- Hidden probe/SCS：`/mnt/datasets-livsyn/dxl/mattergen_v1_self_correcting_search/experiments/self_correcting_search_p0/final_report.md`。
- Field/site async 与本轮轨迹：`/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/{shared_field_dynamics,field_async_freeze_p0,site_level_convergence_diagnostic}/`。
- Cell–atom fusion：`/mnt/datasets-livsyn/dxl/mattergen_v1_cell_atom_fusion/experiments/cell_atom_fusion_p0/final_report.md`。
- 只读 stash 证据：`stash@{0}` 的 `experiments/raab_sc_p0/final_report.md`；`stash@{1}` 的 `experiments/mattergen_rl_p0/final_report.md`；`stash@{2}` 的 `experiments/alm_gen_p0/final_report.md`；`stash@{3}` 的 `experiments/mattergen_lora_p0/final_report.md`；`stash@{4}` 的 `experiments/composable_property_adapter_p0/final_report.md`。读取方式为 `git show`/`git stash show`，未 apply/pop。

## 已有正基线

### Innovation 1

归档 Formal256 报告给出 A0 相对 C0：E-hull −0.003435 eV/atom，Stable +5.859 pp，NUS +3.516 pp，`FORMAL_INNOVATION1_CONFIRMED=True`。因此任何新 Innovation 2 必须在论文叙事上与“多字段 residual-driven online adaptive CFG”清楚正交。

### E3-PCR

远端分支 `remotes/origin/feature/q3-e3-pcr-formal256` 的 Formal256 报告给出：

- C0 MaxF 0.342964 → E3-G 0.263107 eV/Å，下降 23.2844%。
- 配对 bootstrap 均值差 95% CI `[−0.144966, −0.032453]`，Holm 校正 Wilcoxon `p=4.19e-10`；163 胜、93 负。
- RMSD 0.049390 → 0.045937；E-hull 0.156136 → 0.156177 eV/atom，实质不变。
- Stable 44.531%、NUS 22.266%、Novel 71.875%、Unique 98.438% 均不变。
- 门控只 refine 170/256，平均 3.320 步；129 个 gate 参数；有害率 25.39% → 18.36%。
- 局限：CHGNet 是 refinement teacher，MatterSim-5M 是 evaluator；没有 DFT 验证。

这构成很高的替代门槛：新候选不应只在 16/32 样本上让一个均值略好，就声称优于 E3-PCR。

## 失败路线及其实际结果

| 家族 | 真实结果 | 失败的核心原因 | 对新候选的约束 |
|---|---|---|---|
| Corrector/加速 | residual distillation Formal256 达到 1.3246×，主体质量非劣，但 force tail 非劣性未成立，最终 BORDERLINE | 省 forward 容易把误差集中到尾部 | 不再以“更少 score call”作为 Innovation 2 主轴 |
| Quality adapter / LoRA | 小样本一度改善部分质量；independent 样本暴露 Novel/NUS 收缩和 RMSD/force tail；LoRA target error 约为 official 的 12×且 0/4 hit | 参数高效不等于分布安全；训练信号与生成目标错配 | 不再往 backbone/hidden state 加通用 adapter |
| Global Transformer / cross-field adapter | Transformer 出现 884-step relaxation tail；cross-field P1 的 Stable −12.5 pp、NUS −25 pp | 更强表达力覆盖了已有先验，几何局部改善换来分布崩坏 | 不把“更复杂耦合模块”当默认答案 |
| RL | 训练 reward 上升，但 held-out error 仍约 0.18895，official 约 0.01995，0/4 hit | reward hacking/代理泛化失败 | 不用 RL 或在线 reward 再包装 |
| ALM | Element/Stoich hit 0/16，文本 adherence 下降 | 语义先验没有落到数值晶体生成 | 不进入 semantic–numeric joint conditioning |
| GCPA / property control | 4 样本 official 3/4 hit、error 0.00414；GCPA 0/4、error 0.18222 | 新控制覆盖了原有条件能力 | 新方法必须保留官方 posterior/score 的相对概率结构 |
| RAAB dynamic refinement | oracle 能保留/改善 force，但比 E3-G 多 30.5% steps，伤害减少不足，controller 未进入训练 | 可控预算不是新质量机制，且弱于已有 gate | 不再做 dynamic refinement budget |
| Adaptive neighborhood | A0 Mag MAE 0.00899 vs B0 0.01008；hit 65.6% vs 71.9%；MaxF、E-hull、Novel/Unique/NUS 多项变差 | 图邻域随时间变化造成 OOD message passing | 不改变邻接规则/截断半径作为主线 |
| Multi-field self-conditioning | S-on 没有 target 优势，MaxF 恶化 26.85%，E-hull 变差 | clean estimate 回注产生训练–推理错配 | 不做普通 self-conditioning |
| Hidden probe / SCS | held-out Pearson 0.189、Spearman 0.173、R² −5.276；搜索按 gate 未运行 | hidden state 不能可靠预测目标 | 不用内部 probe 驱动搜索或 tail gate |
| Field time schedule / async freeze | 64 轨迹 validation 最大字段 median gap 0.06，阈值 0.10；有 ≥0.10 gap 的轨迹 40.62%，阈值 60%；`FIELD_CONVERGENCE_HETEROGENEITY=FALSE` | 所谓字段异步主要不是可复现的收敛差 | 停止 field-level async/schedule 家族 |
| Site-level async | validation `SiteSpread≥0.1` 为 0%；`STOP_ASYNC_FAMILY` | site 级异质性也不存在 | 不把字段异步缩小到 site 再试一次 |
| Cell–atom fusion | 新增参数 0.6775%；validation loss 改善为 −0.00318%（即略坏）；gate 约 1e−3、残差只占 hidden 0.108%；16 smoke 的 RMSD +47.8%，CI 跨零 | 模型学会忽略模块；smoke 混合且分布指标变差 | 不再做 learned hidden fusion |
| GBSA/TCL | prospective32 的 NUS 点估计 +18.75 pp，但 CI 跨零；组成有效性 93.75% → 81.25%，RMSD/relaxation tail 变坏；TCL Formal256 几何/force 更差 | 小样本正向点估计未复现，质量收益伴随尾部代价 | P0 必须预注册主指标与独立 guardrail |

## 新做的零训练诊断

数据源：`/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/shared_field_dynamics/`。没有写回历史目录。

### 1. Atomic–geometry cross-lag

对每个轨迹构造 `signal(t) → response(t+1)`。同时报告 pooled Spearman、按 observation ordinal 去中位数后的残差 Spearman，以及逐轨迹 Spearman 中位数。逐轨迹和 raw pooled 相关会被共同的去噪时间趋势放大，因此判断以 calibration/validation 的 timestep-residualized 结果为主。

| 关系 | Cal raw | Cal residualized | Val raw | Val residualized |
|---|---:|---:|---:|---:|
| atom argmax change → next position displacement | +0.295 | −0.008 | +0.326 | −0.065 |
| atom entropy → next position displacement | +0.659 | −0.134 | +0.641 | −0.168 |
| cell metric change → next position displacement | +0.771 | +0.126 | +0.721 | +0.059 |
| cell volume change → next position displacement | +0.712 | +0.004 | +0.681 | −0.005 |
| position displacement → next atom argmax change | +0.308 | −0.049 | +0.357 | −0.055 |
| position displacement → next atom entropy | +0.657 | −0.130 | +0.638 | −0.188 |
| position displacement → next cell metric change | +0.759 | −0.039 | +0.733 | −0.001 |

结论：没有稳定的正向、时间步以外的 cross-lag 依赖。`dependency-aware sequential update` 不通过 zero-training diagnostic；若继续做，本质会重演 async/coupling 家族。

### 2. Late residual/entropy 与 final bad tail

对 reverse progress ≥0.90 的 atomic entropy/change、position displacement/raw-to-clean、cell metric/volume change取 mean/max/last，再分别与 final MaxF、RMSD、E-hull、relaxation steps 做 Spearman。

- validation 中最后 atomic entropy 对 MaxF 看似很强（ρ=+0.702），但 calibration 未复现到 `|ρ|≥0.30`；不能据此造新 gate。
- cell late residual 对 E-hull 的 calibration 相关约 +0.47，validation 未复现。
- late position displacement 与 MaxF 的相关反而为负，且 calibration/validation 约 −0.43/−0.31；它不是“越抖越坏”的直接 tail indicator。
- 跨两个 split 同号且 `|ρ|≥0.30` 的稳定直接信号主要是 final/as-generated 几何：minimum periodic distance 对 MaxF 为 −0.538/−0.683，对 E-hull 为 −0.565/−0.587，对 RMSD 为 −0.605/−0.598。
- volume/atom 对 MaxF 为 −0.475/−0.457，对 RMSD 为 −0.341/−0.371；atoms/Å³ 符号相反、幅度相同。
- position raw-to-clean max 对 MaxF 为 −0.420/−0.407，但方向不支持简单“高 residual 是高风险”。

结论：Residual-Triggered Selective ReNoise 和 residual/entropy tail controller 应停止。直接几何可行性信号值得保留，因为它可在线精确观察、无需 probe，但应做成约束保持而不是拟合 tail 分类器。

### 3. 元素归一化局部几何

用 ASE covalent radii 对 64 个 as-generated 结构计算最小 `d_ij/(r_i+r_j)`、p10 和 soft overlap，并分 split 关联 final 指标。minimum normalized distance 对 E-hull/RMSD 在 calibration 为 −0.457/−0.401，但 validation 只有 −0.196/−0.192；p10 几乎为零相关，soft overlap 也未复现。

结论：现有数据不支持“按共价半径重权 atomic posterior”作为高优先级路线。稳定的是绝对最小距离和体积/原子，而非这套元素归一化启发式。

### 4. Composition-invalid gap

E3 Formal256 的 C0、E3-A、E3-G 都是：structure validity 256/256，composition validity 202/256。E3 只移动 position，不改 atomic numbers/cell，因此 54 个化学无效样本完全保留。其他 cohort 的 chemistry-invalid-but-geometry-valid 比例有波动：GBSA C0 2/32、adaptive-neighborhood B0 2/32、self-conditioning B0 6/32。这说明错误源真实但 cohort-dependent；P0 必须带 baseline-prevalence gate，不能保证每个 32 样本批次都有 21.1%。

## 横跨失败路线的六个规律

1. **代理信号错配比参数量更危险。** 129 参数可以成功，0.7% 参数也会失败；关键是信号是否直接对应可干预错误。
2. **小样本均值极易制造假阳性。** Adapter、GBSA/TCL 和若干 smoke 都在 independent/formal 阶段反转，因此 P0 只能作 gate。
3. **分布收缩是最常见的隐性成本。** target 或局部质量改善，经常由 Novel/Unique/NUS 换来。
4. **共同时间轴不能当因果耦合。** raw cross-lag ρ 可到 0.77，去掉 timestep 后接近零；这是本轮最重要的反例。
5. **改成熟 backbone/邻域/hidden state 的风险远高于改采样可行域。** 前者造成 OOD 或被模型忽略，后者可限制在小、可回退的更新内。
6. **成功 gate 需要直接可观测、机制对齐的量。** E3 的 pre-relax force 与位移干预同域；hidden probe、residual tail prediction 不满足这一点。

## 对后续候选的硬约束

- 不再训练通用 adapter、LoRA、RL、ALM、hidden probe 或异步控制器。
- 不用 raw time correlation 支持 sequential/async 更新。
- 不把 mean 改善当作 tail 安全；guardrail 至少包含 E-hull、NUS 和 MaxF/RMSD 中的相关项。
- 优先“不改变网络、最小改变 posterior/step、失败时精确回退官方采样”的算法。
- 若候选只复制 Scout-Matter 的 volume/coordination guidance、CrysVCD 的前置合法 composition，或把 E3 移进 sampler，则不算独立创新。
