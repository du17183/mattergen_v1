# Metric Framework Reassessment FINAL

日期：2026-09-08。项目 `/mnt/lis-wam-data/dxl/mattergen_v1`。读取时分支 `experiment/gbsa-tcl-p1`，HEAD `b12985130a7e7dfff490118d566fefd1e2b65af2`。

本报告是已有数据的回顾性分析和未来研究建议，不是新的实验结果或新的 preregistration 已经执行。没有训练、生成、模型 forward/backward、MatterSim relaxation、Atomic-only、P1/P2/Formal，也没有修改历史 checkpoint、采样代码、训练代码或历史 verdict。新计算只用项目 `.venv` 在 CPU 上读取 CSV、Git 归档和已保存的力数组，进行描述统计与重采样。

核心判断：**有条件支持 E-hull + NUS 作为材料发现的双主指标；GBSA 有值得独立验证的 NUS 信号，但目前不能宣布双主指标成功或物理安全。第一研究推荐 B，唯一 fallback C。**

## 1. 读取了哪些真实实验文件

主要证据不是聊天摘要：

| 实验 | 真实来源（项目内路径或本地 Git blob） | 使用范围 |
|---|---|---|
| Adaptive CFG Formal256 | 本地 `origin/archive/thesis-analysis-package-v1` 的 `thesis_archive/reports/innovation1/formal_final_report.md`、`thesis_archive/data/innovation1/per_seed_metrics.csv`、`source/*official_metrics_per_structure.csv`、`source_manifest.json` | 256 paired seeds，主指标及可恢复几何 |
| Corrector V1/V2/V3 | `experiments/corrector_residual_distillation_{v1,v2,formal256,v3_anchor}/` 的 final_report、quality_metrics/quality_per_seed、stage_c_results、speed metrics、paired statistics | 早期筛选、Formal256、后续独立64、真实单 H20 测速 |
| E3-PCR | 本地 `origin/feature/q3-e3-pcr-formal256` 的 `reports/q3_e3_pcr/formal256/final_report.md`、三臂 official_metrics_per_structure、gate_mechanism_summary；归档 per_seed_metrics | C0/E3-A/E3-G Formal256 与门控消融 |
| A0 + E3-G | 归档 `reports/a0_e3g_compat64/`、`reports/a0_e3g_independent64/` 的 final/quality/statistics reports | 兼容64、独立64；不冒充组合 Formal256 |
| TCL、FT0 | `experiments/tcl_dml_p0/`、`tcl_p1/`、`tcl_p2/`、`tcl_formal256/` 的 final_report、quality_results、paired statistics；Formal detailed JSON 与 initial_with_properties.extxyz | 从小样本到正式结果，FT0 必须与 pretrained C0 分开 |
| TCL 根因 | `experiments/tcl_formal256_root_cause/root_cause_report.md` 及既有 drift/gradient 统计 | 只引用过去完成的机制诊断，本轮不重做 forward |
| GBSA P0/P1 | `experiments/gbsa_tcl_{p0,p1}/final_report.md`、quality_results、P1 per_seed_results/paired_statistics、relaxation_summary、initial/relaxed.extxyz | 完整四臂、85014 初末态、所有留一对敏感性 |
| 其他本地质量路线 | distribution-constrained/balanced quality adapter、global Transformer、cross-field interaction 各阶段 final_report 与 quality_results | 纳入完整历史 CSV，不挑最好阶段 |
| 更早归档路线 | `docs/experiments/negative_results_summary.md`；归档 postgen_fastgate 的原始 Q5/new_eval reports 等 | Corrector Gating、Residual Reuse、FN-PRA、CrystalREPA、RP-QTFG、CG-TDR、Q1/Q2/Q4/Q5/Q6；缺失原始逐结构数据的路线明确限制证据等级 |

可复核交付：

- [统一历史全字段表](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/historical_table.md) / [CSV](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/historical_metrics.csv)：48 个方法×阶段比较行；对照臂标明 CONTROL，不能把所属实验的 FAIL 错当成该对照的独立历史判定。
- [主要实验绝对指标与尾部分布](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/major_absolute_metrics.csv)：六个 cohort、17 个臂。
- [回顾性配对区间](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/retrospective_paired_statistics.csv)、[GBSA 尾部与影响分析](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/gbsa_tail_diagnostics.json)、[基线抽样波动](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/baseline_sampling_variation.csv)。
- [CPU 分析脚本](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/analyze_existing.py)、[数据来源索引](/mnt/lis-wam-data/dxl/mattergen_v1/experiments/metric_framework_reassessment/sources.md)。本地 Git 来源固定到读取时 commit；没有 fetch/切分支。

证据边界：并非每条早期路线都恢复到了原始逐 seed 文件。不能为缺失字段伪造统一 CI 或补零。历史负面汇总中明确写 Not fully recovered 的项目，本报告也仅作为有限级别的归档证据。

## 2. E-hull / Stable / NUS / Novel / Unique 的关系

按本项目正式代码，而非指标名称猜测：

| 指标 | 含义、关系与局限 |
|---|---|
| E-hull | 弛豫后能量相对固定参考相图的每原子超额能，连续量；当前由 MatterSim-5M 提供能量，不是 DFT。`allow_negative=True` 允许低于固定参考 hull，不代表相对于包含所有候选的新完整 hull 仍有负的 energy-above-hull。 |
| Stable | 本流程为 `E-hull <= 0.1 eV/atom` 的阈值事件；与 E-hull 存在确定性的逐结构包含关系，但均值不能决定阈值以下比例。这里 Stable 包含一定亚稳窗口，不等于可合成保证。 |
| Novel | 相对冻结参考数据库与 matcher 的新颖性；不是相对于人类全部已知材料的绝对新颖性。 |
| Unique | 同一评价集合中重复结构的代表性标记；依赖集合规模、匹配容差和代表选取。 |
| NUS | `mean(Stable AND Novel AND Unique)`；必然不超过三个边际比例中的任何一个，但通常不等于三者边际比例相乘。 |

因此 E-hull 与 NUS **互补而不独立**：前者保留连续能量信息，后者加入“新、不同且达稳定窗口”的联合发现信息。Stable 是 E-hull 的阈值投影；Novel/Unique 提供参考库和集合多样性信息。NUS 不能单独说明变化来自何处，故三者仍须报告，但没有必要再与 NUS 一起成为四重同等级改善门槛。

平均 E-hull 降低可能只因少数极坏结构改善，Stable 不变；Stable 提升也可能来自边界附近结构跨线，而平均 E-hull 反向。NUS 提升时 Novel 边际比例下降并不自动失败：可能减少了大量“新但不稳定”的样本。

## 3. RMSD / Force / Relaxation 应该扮演什么角色

RMSD 描述生成结构到弛豫结果的结构变化；初始 force 描述该点的局部能量梯度。它们不等价：高曲率环境可有大力但小位移；较平缓势能面或晶胞变化可有较大结构变化但并不对应同样大的力。

原子力按每结构先取原子力范数均值，再对结构等权平均。MaxF 是每结构最大原子力，再在结构之间统计均值、分位数与事件率；**不是整个数据集单一最大值**。原子池加权会使大结构占更大权重，不与结构等权均值混用。

本轮纠正了一个重要跨实验口径：Adaptive 归档 `c0_max_force/a0_max_force` 映射自 `maximum_force_ev_ang`（弛豫输出），均值为 0.036616/0.039243；后期 E3、V2、TCL、GBSA 的主 force 是弛豫前力。统一表把前者单列为 `post_relax_max_force`，不拿它证明 C0 没有初始力尾部。Adaptive 原始 pre-force 尚未恢复。

另一个限制来自 `mattergen/evaluation/utils/utils.py:61`：结构匹配失败时返回 `MAX_RMSD / normalization` 的惩罚值。因此有些高 RMSD 是带体积尺度的匹配失败惩罚，不是直接测出的真实原子位移；不能把所有大 RMSD 解释成相同几何机制，未来应单列匹配失败率。

对“发现率改善”论文：RMSD/初始 force 为物理稳健性 guardrail；steps/time 主要是诊断和成本。对明确研究物理精修的 E3，则 force 可以是**该贡献本身的预注册主终点**，而 E-hull/NUS 是必须保护的发现质量。层级服务于科学问题，不是要求所有论文贡献都优化同一标量。

## 4. 是否支持 Primary = E-hull + NUS

**有条件支持，不支持作为“材料逆向生成成功”的全部充分条件。**

对固定任务、固定采样预算、固定 reference/matcher/relaxation 的代理材料发现比较，这两项覆盖连续稳定性质量与有效新发现率，是合理的主要终点。MatterGen 原论文也把稳定、独特、新颖的联合发现与条件属性设计区分讨论；本地代理评价不能冒充论文中的 DFT 级结论。[MatterGen 原论文](https://www.nature.com/articles/s41586-025-08628-5)

但是本项目多条路线条件是 magnetic density=0.1，主要正式结果没有独立的属性目标验证。更稳定、更新颖不等于更接近磁性目标。完整逆向设计评价还需要冻结的独立 target error/hit rate，以及 target-qualified NUS；项目代码虽有 property metrics，不代表本轮历史报告已经测过。Q5 的 CHGNet target proxy 也不能替代其他路线未做的独立验证。

MatterSim 是学习势，能量/力与对应相图仍存在代理误差、分布外误差；“来自真实 pretrained 模型生成”不等于“材料物理已由 DFT/实验确认”。[MatterSim 原论文](https://arxiv.org/abs/2405.04967)

## 5. 推荐完整评价层级

| 层级 | 固定角色 | 不应做什么 |
|---|---|---|
| Primary | E-hull + NUS；事先指定一项 superiority，另一项 non-inferiority（NI），或明确要求两项 superiority | 结果出来后选择任一有利指标便宣布成功 |
| Secondary | Stable、Novel、Unique，以及组成/元素/结构族分布 | 要求每项都显著改善；把 Unique=100% 当作无 mode collapse 的证明 |
| Physical guardrails | 结构有效性/不可恢复失败；RMSD 均值与匹配失败；初始原子力均值；预定义 MaxF 事件率 | 逐项必须优于 C0，或用单个 maximum 一票否决整个总体分布 |
| Diagnostics / cost | steps 中位数/P95、>200/>400、实际耗时、末态力、cell/距离异常、score drift | 用 steps 代替全部物理合理性；用 score drift 改善替代真实材料质量 |
| Inverse-design evidence | 独立 target error/hit 与 target-qualified NUS（当前多数缺失） | 无独立属性评估却宣称定向逆向设计成功 |

## 6. 历史实验统一对比表

以下全部为“方法减去**同一 cohort 的 C0**”。E 为 eV/atom，RMSD 为 Å，力为 eV/Å；比例变化统一用百分点 pp。完整单张全字段表见第1节链接。不同 cohort 的绝对值不可直接做方法排行榜。

| 方法 | 阶段 / n | 历史 verdict | ΔE | ΔNUS pp | ΔStable pp | ΔNovel pp | ΔUnique pp | ΔRMSD | ΔAtomicF | ΔMaxF | 回顾分类 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Adaptive CFG | Formal / 256 | CONFIRMED | -0.003435 | +3.516 | +5.859 | +2.734 | +2.344 | +0.009247 | NA | NA（pre） | U：正向但 superiority 不确定 |
| E3-G | Formal / 256 | E3_G_FORMAL_CONFIRMED | +0.000041 | 0 | 0 | 0 | 0 | -0.003452 | NA | -0.079857 | C |
| E3-A | Formal消融 / 256 | primary/quality pass | +0.000043 | 0 | 0 | 0 | 0 | -0.004332 | NA | -0.099008 | C；门控需与它比较 |
| Corrector V2 | Formal / 256 | BORDERLINE | +0.003738 | +1.563 | +0.391 | +2.344 | -1.172 | -0.027359 | -0.004641 | +0.006071 | C型加速候选，NI不完整 |
| FT0 | TCL Formal / 256 | 对照；非独立创新 verdict | +0.038687 | -19.531 | -20.703 | -0.391 | -9.375 | +0.066358 | +0.153211 | +0.263234 | E |
| TCL | Formal / 256 | FAIL | +0.002111 | -3.516 | -4.297 | +1.172 | +0.391 | +0.033581 | +0.033755 | +0.050695 | D/E：无主收益且几何恶化 |
| GBSA | P0 / 8 | GO | -0.008676 | +50.000 | +25.000 | +25.000 | 0 | +0.005675 | -0.112047 | -0.249918 | A型探索信号；未证实安全 |
| FT0 | GBSA P1 / 32 | 对照；P1方法判定FAIL | +0.008575 | -3.125 | -6.250 | +3.125 | 0 | +0.010936 | +0.141012 | +0.275429 | 当前样本不优于 C0 |
| TCL | GBSA P1 / 32 | 对照；不替换TCL Formal | -0.036536 | +9.375 | +6.250 | +3.125 | 0 | -0.021361 | +0.038143 | +0.108432 | 小样本主指标正向、力风险 |
| GBSA | P1 / 32 | FAIL under original P1 | -0.020924 | +25.000 | +15.625 | +9.375 | 0 | -0.043880 | +0.052120 | +0.144909 | B型 NUS信号；非确认性成功 |

物理尾部与成本补充（计数均为该行同 cohort C0→方法）：

| 方法/阶段 | MaxF P95 | MaxF P99 | MaxF>1 | MaxF>2 | RMSD>0.5 | 速度/成本 |
|---|---:|---:|---:|---:|---:|---|
| E3-G Formal256 | 0.936→0.741 | 3.046→1.531 | 11→9 | 5→2 | 3→2 | 增加 CHGNet 精修/门控；steps 40.082→40.000，不能宣称端到端加速 |
| V2 Formal256 | 0.933→0.983 | 1.654→2.046 | 13→12 | 2→3 | 12→5 | 单 H20 1.324642×，CI [1.267466,1.376734]，n=16；steps 53.695→42.699 |
| TCL Formal256 | 见绝对指标CSV | 见CSV | 16→24 | 3→5 | 1→10 | steps 47.082→62.750，>400：3→8 |
| FT0（TCL Formal） | 见CSV | 见CSV | 16→30 | 3→10 | 1→13 | steps 47.082→81.887 |
| GBSA P1 | 0.671→0.801 | 0.994→4.733 | 1→2 | 0→1 | 2→0 | steps 58.344→36.875；非受控端到端加速证据 |

其他可恢复路线也必须纳入，尤其不能删掉独立验证中的反例：

| 路线/阶段 | 真实信号与比较对象 | 历史结论及回顾解释 |
|---|---|---|
| V1 Stage-C32 | Fallback50：ΔE -0.00716、NUS +12.5pp、RMSD +0.02662、MaxF均值 +0.04879；约1.267×。Fallback75/90：ΔE +0.14334/+0.08706，NUS -12.5/-15.625pp | 继续开发但不进Formal；50有B型探索价值，不能据此美化更激进版本 |
| V1 Reuse / Skip | Reuse：ΔE +0.06137、NUS -9.375pp，虽约1.847×；Skip仅24/32生成，成功子集RMSD均值2.5948、单结构MaxF最大68.826 | 前者直接质量代价；后者明确失败/严重不稳健。Skip成功子集均值不是32次尝试总体质量 |
| V2 前期独立64 | ΔE +0.02000、NUS -4.6875pp、RMSD +0.01290、MaxF +0.03468；采样1.3396× | GO to Formal只表示晋级，不是已证明保质 |
| V2 后续独立64（69000–69063） | ΔE -0.00950、NUS -3.125pp、RMSD +0.04410；1.3451× | 速度复现、RMSD优势未复现；必须与Formal反向结果同时报告 |
| V3 K16 独立64 | ΔE -0.00120、NUS -7.8125pp、Stable -9.375pp、RMSD +0.03349、MaxF +0.06599；1.2992× | FAIL；相对V2的MaxF均值差CI还为正，修复假设没成立 |
| M2 quality adapter P1/32 | ΔE -0.04996、NUS 0、Stable +6.25、Novel -12.5pp、force明显下降；普通M1的NUS更高 | BORDERLINE；Novel单项门槛可能遮蔽能量收益，但后续16的M2、DB并未稳健复现，不能只保留P1 |
| M2-L1/L2 /16、M2-DB /16 | L1小NUS收益伴RMSD +0.05716；L2 E下降但NUS -6.25pp；DB E +0.01495、NUS -12.5pp、RMSD +0.11239 | FAIL保留；不是全部由过严guardrail造成 |
| Global Transformer P0/8 | ΔE -0.06251、NUS +50pp，但RMSD +0.08168、steps +104.375；普通MLP能量还更好 | FAIL；B型小样本信号而非结构创新已经成立 |
| CFI P0/8→P1/32 | P0 E -0.07203、NUS +25pp；P1 E -0.01216、NUS -6.25pp、Novel -18.75pp；P1 MLP NUS明显更好 | GO→FAIL；独立验证为主指标tradeoff，不该只讲P0 |
| TCL P1/32、P2/64 | vs C0：P1 E +0.01570/NUS +3.125pp；P2 E +0.00427/NUS -3.125pp | CLEAR GO、GO主要基于修复FT0；不等于优于pretrained C0 |
| DML P0/8 | E +0.06152、NUS -25pp，即使RMSD下降 | FAIL；Primary本身不佳 |
| Corrector Gating Formal256 | **vs A0**：1.506×，E +0.022423、NUS -9.375pp、Stable -9.7656pp | Formal No-Go；核心发现质量损失，不是单纯力门槛太严 |
| Budget-aware Gating /32 | **vs A0**：G1 1.183×、NUS -6.25pp；G2 1.234×、E +0.022292 | No-Go；质量与速度折中仍弱于优先路线 |
| Unconditional Residual Reuse | 无正式质量比较；逻辑NFE -16.8%但中位wall time +1.35%，最佳同并发吞吐仅+1.16% | No-Go；真实性能收益不足，改质量评价体系不能补出加速 |
| FN-PRA /32 | **vs A0**：E +0.003786、NUS +6.25pp、Novel +21.875pp、RMSD -28.68%，composition/Stable各-6.25pp | No-Go；有tradeoff信号，但组成保护与独立复现缺口仍在 |
| CrystalREPA /64 | **vs无条件MP20 U0**：E +0.094236、RMSD +0.03329，Stable均0 | No-Go；不可与条件C0混排，也不能重解释为成功 |
| RP-QTFG gate8 | **vs A0**：最近候选E -0.003353、NUS不变，RMSD +68.28%、MaxF仅-0.26%、延迟+30.19% | No-Go；无有用主收益，代价大 |
| CG-TDR gate8 | V2P近乎平坦，V2C median RMSD +18.29%；残差泛化弱 | No-Go；更换指标不会产生缺失的效应 |
| Q1/Q2/Q4 historical pools | **vs C0_FIRST**：E/NUS联合正向，但Novel下降；32 held-out rows反复抽8 pools，共1000 trials | No-Go可能过度受Novel单门槛影响；1000不是独立seeds，且需4候选预算 |
| Q5/Q6 new32 pools | **vs C0_FIRST**：E -0.03140/-0.03372、NUS +3.125/+9.375pp、Novel -15.625/-12.5pp；几何改善 | No-Go保留；可称所选样本质量信号，不能忽略四候选成本，不能声称等预算发现率提高 |

上表中非C0基线已明确标注，不能把这些差值填成 Δvs C0。旧归档不完整的路线未臆造 atomic force、CI 或 tail counts。工程/GPU并发优化不另包装为材料质量创新。

## 7. Adaptive CFG retrospective classification

C0→A0：E 0.143667→0.140232；NUS 22.2656%→25.7813%；Stable 41.0156%→46.875%；Novel 73.8281%→76.5625%；Unique 95.7031%→98.0469%。RMSD 0.064575→0.073822。

历史 `FORMAL_INNOVATION1_CONFIRMED=True` 完整保留。当时的确认规则允许方向性主指标改善与有限伤害保护，并非所有改善的CI都要排除零。

本轮20,000次paired bootstrap、seed=20260908：ΔE -0.003435，95% CI [-0.017934,+0.010998]；ΔNUS +3.5156pp，CI [-3.125,+10.1563]；ΔStable +5.8594pp，CI [-1.5625,+13.2813]。因此是**方向一致的正向历史结果，但强superiority尚不确定**，不能为了维护第一创新而要求第二创新更强的统计证据却把第一创新写成“全部显著”。新增U类（证据未定），不硬塞为已确认A类；缺失pre-force也不能断言全面SAFE。

## 8. E3-PCR retrospective classification

**C：quality-preserving physical robustness refinement。** C0/E3-G E分别0.1561357/0.1561767，差仅+0.000041 eV/atom，回顾CI约[-0.000959,+0.001010]；NUS、Stable、Novel、Unique在256个配对样本中的标签完全相同。均值MaxF降低23.2844%，原正式CI为Δ[-0.144966,-0.032453] eV/Å；RMSD平均降低0.003452 Å，但其CI跨零，不能说RMSD显著改善。

门控覆盖170/256=66.4063%，避免33.5938%干预；保留always-on约80.657%的平均力收益。E3-A力均值降低28.8684%，**平均降力比E3-G更强**；G的贡献是降低有害干预：按原1e-6力变化容差，harm rate由65/256=25.3906%降至47/256=18.3594%；只A有害22例、只G有害4例。低力子集harm由29.6875%降至17.9688%。不能把舍弃部分平均收益的gate描述为处处优于always-on。

它有学习：14→8→1的129参数gate；但MatterGen骨干没有因此学习，实际结构更新是CHGNet驱动的生成后小步精修，冻结元素与晶胞，有限步长/位移及backtracking。不是大型生成模型训练创新。用于精修的CHGNet与评价的MatterSim分开是优点，但不等于已经完成DFT或独立属性验证。

组合证据：A0+E3-G兼容64降力约27.10%；全新50000–50063独立64再次降力19.02%，CI [-0.102213,-0.010696]，主指标保持。**单独E3有Formal256，组合只有有效64+64，不是组合Formal256。** 与gate训练集重叠的历史泄漏诊断不算独立支持。

作为硕士第二创新可以成立，前提是题目诚实聚焦“学习式选择性物理精修/稳健性”，清楚展示learned gate相对固定规则和always-on的价值，不伪称提高NUS或修改MatterGen主干。学位充分性最终仍由导师与答辩要求决定。

## 9. Corrector V2 retrospective classification

Formal256：E 0.102141→0.105879，NUS31.6406%→33.2031%，RMSD0.080374→0.053015；AtomicF0.134200→0.129559，MaxF0.284241→0.290312。E差CI [-0.01149,+0.01980]、NUS差CI约[-5.47,+8.98]pp，不能把“差异不显著”称为严格保质已证实。

原预注册允许E +0.025、NUS/Stable -10pp及RMSD +0.02Å等NI门槛；部分通过，但均值MaxF/P95/P99相对保护的上界尚未足够小，故BORDERLINE不是已证明力崩溃。P99点值1.654→2.046和>2的2→3仍需保留。

速度是可信的独立价值：n=16单H20、batch1，paired平均speedup1.324642× [1.267466,1.376734]，物理forward减少约29.65%；是计时区间内采样墙钟，不含加载/训练/完整评价，不能直接说有效材料发现端到端吞吐提升32.5%。模型是2661参数的训练残差adapter，模型/训练层程度高于E3门控但低于完整生成模型微调。

**可以重定位为保质加速候选，不能回顾性宣布已经保质。** 前期独立64的E/NUS点估计更差；后续69000–69063独立64虽速度1.345×再现，RMSD却+0.04410 Å。必须解释这种跨seed不稳定。若采用本报告建议的更严E/NUS NI，它也不是自动通过。故本轮不把V2列在唯一first/fallback中。

## 10. TCL Formal retrospective classification

| n=256 | E-hull | NUS | Stable | Novel | Unique | RMSD | AtomicF | MaxF | steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.112332 | 32.0313% | 55.4688% | 75.3906% | 98.4375% | 0.049300 | 0.148648 | 0.337212 | 47.082 |
| FT0 | 0.151019 | 12.5000% | 34.7656% | 75.0000% | 89.0625% | 0.115658 | 0.301859 | 0.600446 | 81.887 |
| TCL | 0.114443 | 28.5156% | 51.1719% | 76.5625% | 98.8281% | 0.082882 | 0.182402 | 0.387907 | 62.750 |

TCL显著修复FT0的遗忘，不等于优于C0。vs C0无主质量增益，RMSD均值增加0.03358 Å，回顾CI [0.00760,0.06033]；AtomicF增加0.03375，CI [0.00382,0.06430]；RMSD>0.5从1增到10。故没有理由仅因新层级而恢复优先级，历史FAIL不变。

严格区分：RMSD“恶化超过零”得到支持，不等于已证明“超过未来+0.02Å的material margin”（CI下界尚未超过0.02）。因此回顾类D/E，而不是偷换成按未来新margin完成的确认性guardrail FAIL。FT0则主指标和物理表现更明确地整体受损。

已有根因支持clean-cell反演小alpha放大、全局clip挤压其它梯度、shared表征漂移；late position drift与坏结构相关。但根因报告明确其probe来自终态重新加噪，不是保存的反向生成轨迹重放，不能用相关性冒充因果证明。本轮没有重跑这些probe。

## 11. GBSA P0 retrospective classification

84000–84007，n=8。C0→GBSA：E0.077954→0.069278，NUS25%→75%，Stable62.5%→87.5%，Novel62.5%→87.5%，Unique均100%；RMSD0.028622→0.034297，AtomicF0.172185→0.060137，MaxF0.348243→0.098325，steps42→21.625。

这是A型探索信号，足以解释当时GO到P1；不是n8证明总体安全，也不能把P0与P1看到的结果合并当作一套盲测Formal。P1没有重训冻结GBSA，提供了一次更有价值但仍小样本的独立检验。

## 12. GBSA P1 retrospective classification

分支/HEAD与用户指定一致，seeds85000–85031。四臂全部32/32真实生成和已完成MatterSim评价。历史结论仍为 **FAIL under its preregistered P1 criteria**，原因是未复现要求的atomic/max-force repair，而不只是实现没通过检查。

| n=32 | E-hull | NUS | Stable | Novel | Unique | RMSD | AtomicF | MaxF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.139579 | 21.875% | 43.750% | 75.000% | 100% | 0.087994 | 0.127401 | 0.246207 |
| FT0 | 0.148154 | 18.750% | 37.500% | 78.125% | 100% | 0.098930 | 0.268413 | 0.521636 |
| TCL | 0.103043 | 31.250% | 50.000% | 78.125% | 100% | 0.066632 | 0.165544 | 0.354640 |
| GBSA | 0.118655 | 46.875% | 59.375% | 84.375% | 100% | 0.044114 | 0.179521 | 0.391117 |

vs C0，NUS由7增到15，+25pp，约2.14倍；配对为10胜/20平/2负，原95%CI [+6.25,+43.75]pp。E下降0.020924 eV/atom，约15.0%，20胜/12负，但原CI [-0.080819,+0.037636]，**既没证明E superiority，也没证明+0.01的E NI**。不能把“E点值下降且NUS显著”缩写成双主指标已成功。

Stable 14→19，stable中的novel且unique比例从7/14=50%升至15/19=78.95%，共同解释NUS提升。Novel 24→27、Unique均32支持“没有明显精确重复塌缩”，但它们非独立复制的三份证据，n32也排除不了元素/化学系统集中化或属性偏离。

vs TCL，NUS +15.625pp但CI [-3.125,+34.375]，E反而+0.015613，CI [-0.019134,+0.055169]；因此GBSA并非对所有baseline都双主方向更好。RMSD虽均值减半，vs C0的中位数却0.02625→0.03308，逐seed14胜/18负；是上尾改善多于普遍逐样本改善。

回顾归类为 **B型：较强NUS发现信号 + E不确定 + force guardrail BORDERLINE**，不是字面意义“两个Primary SUCCESS已经确认”。选择风险来自此前多路线探索，效应可能有winner's curse。

## 13. 哪些属于 Primary SUCCESS / Guardrail BORDERLINE / Guardrail FAIL / Primary FAIL

- 确认性“E-hull与NUS均改善且已证明SAFE”：现有数据不应强行指定一个赢家。Adaptive方向正向但CI宽；GBSA NUS强而E NI未证实；E3主指标保持而非提升。
- Primary较强探索信号、guardrail未定：GBSA P1最值得验证；Global Transformer P0、V1 Fallback50仅小样本且缺少稳健复现。
- Primary保持、物理/效率获益：E3-G证据最完整；V2是加速确定、保质部分不确定的候选，不与E3等同证据等级。
- 真实明显的物理/执行失败：V1 Skip（8/32失败和成功子集严重RMSD/力异常）。TCL有真实几何恶化证据及历史FAIL，但不能把“显著>0”偷换成“超过任何新material margin均已证明”。
- Primary本身不佳：FT0 Formal、DML P0、激进V1、CrystalREPA；CFI P1、Corrector Gating等是主指标tradeoff/质量代价，不是全被guardrail误杀。

A/B/C/D/E是回顾解释，不是新的正式判定；小样本只写“A型/B型信号”，允许U（未定），不把证据不足硬判成功或物理灾难。

## 14. GBSA seed85014 详细解释

| 同seed | 化学式（均11原子） | E-hull | RMSD | AtomicF | 初始MaxF | steps | 末态MaxF |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | MgMnFe3O5F | 0.197711 | 0.831157 | 0.491502 | 1.089802 | 263 | 0.058070 |
| FT0 | Mn7As4 | 0.047220 | 0.038600 | 0.438290 | 1.038500 | 40 | 0.042753 |
| TCL | HfMn9Zn | 0.184623 | 0.628265 | 0.470502 | 1.162431 | 273 | 0.050419 |
| GBSA | Gd2Mn7Al2 | 0.580920 | 0.105228 | 3.046884 | 6.342232 | 63 | 0.049558 |

同seed配对控制随机起点，不意味着同化学式的受控局部干预。GBSA初始最近原子距离2.113574Å，体积/原子16.8234→17.3165Å³；没有仅凭保存数据就能指认的极短距离数值爆炸，且已有弛豫63步降到末态MaxF0.04956。最近距离无极端值也不证明化学键合理。

这是一个**很差的初始力样本，而且弛豫后E-hull仍0.58092、不Stable、不NUS**；不是被主指标偷偷算成有效发现。不能称它完全可接受，也不能把它说成不可恢复的NaN/碰撞/弛豫崩溃。

“评价success”与严格的最终原子力达阈值不完全等价：同seed C0/TCL保存的末态MaxF略高于fmax0.05。应分开报告程序成功、optimizer/filter收敛与末态原子力，不在本轮重新弛豫或改变历史标签。

全部32对逐一leave-one-pair-out，而不是只删85014：

| GBSA−C0指标 | 全样本差值 | 遍历32个留一对后的差值范围 | 排除85014，仅敏感性分析 |
|---|---:|---:|---:|
| E-hull | -0.020924 | [-0.036409,-0.003721] | -0.033960 |
| NUS（pp） | +25.000 | [+22.581,+29.032] | +25.806 |
| RMSD | -0.043880 | [-0.054997,-0.021878] | -0.021878 |
| AtomicF | +0.052120 | [-0.028630,+0.062232] | -0.028630 |
| MaxF | +0.144909 | [-0.019849,+0.171434] | -0.019849 |

85014占GBSA原子力总和53.04%、结构MaxF总和50.67%，足以改变两个力均值差的符号；**同一seed也贡献C0的大RMSD，因此删它会同时削弱GBSA的RMSD优势**。主结论始终保留它，不使用去尾均值替代主均值，也不把有利的留一结果当新n31独立实验。

## 15. GBSA force 风险三选一：BORDERLINE

| MaxF分布 | 中位数 | P95 | P99 | 最大值 | >1 | >2 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.182426 | 0.671498 | 0.994339 | 1.089802 | 1/32 | 0/32 |
| FT0 | 0.437125 | 1.087043 | 1.341152 | 1.428661 | 5/32 | 0/32 |
| TCL | 0.171174 | 1.080231 | 1.663700 | 1.888907 | 3/32 | 0/32 |
| GBSA | 0.146603 | 0.800739 | 4.732783 | 6.342232 | 2/32 | 1/32 |

GBSA中心更好、极端上尾更差，不是整个分布一致恶化。P95比C0差但比TCL好；n32线性插值P99有69%权重来自最大样本，不能将“最大值大、P99大、均值大”当三份独立灾难证据。

本轮paired bootstrap：AtomicF均值比1.409，95%CI [0.571,2.660]；MaxF均值比1.589，CI [0.649,2.853]。既包括改善，也包括很大的总体伤害。不能用“不显著”证明安全，也不能只看+41%/+59%的点值证明总体已崩溃。

MaxF>2为1/32，二侧Clopper–Pearson95%CI [0.079%,16.217%]；C0的0/32上界仍10.888%。这些区间不是差值区间，但直接显示n32无法精确识别稀有风险。更多历史C0也有重尾：E3 Formal max8.0824、>2为5/256；V2 Formal max6.1895、>2为2/256；TCL Formal max4.2431、>2为3/256。V3开发32的C0甚至max9.4597。故6.342不是只有GBSA能产生的独特“灾难指纹”。

若把各历史>2频率当固定真实值（仅示意），32次至少出现一次的概率约22%–47%；不能将此粗略估计当成GBSA与C0同分布的证明，cohort、模型版本与化学组成未必可交换。

结论：**BORDERLINE**。存在严重个体高力事件；尚无足够证据判定为method-specific总体catastrophic failure；更没有足够证据称acceptable/SAFE。按原“必须修复力”的P1规则FAIL合理，按未来“允许有限非劣化”的问题值得重新检验。

## 16. 之前评价是否过于接近 Pareto domination

部分路线是：如果E/NUS、三个边际比例、RMSD、两种力均值、多个分位数、steps都要求改善/不得有方向性下降，会变成近似逐指标Pareto dominance，且多指标相关、小样本重尾，会频繁出现随机反向项。

但不能说所有旧实验都如此。V2正式已经采用明确NI margins；GBSA P1的中心科学问题正是“修复TCL几何力”，要求力修复不属于无理由审计门槛。它没回答成功原问题，改成“提升发现率且物理不过度恶化”是**换了研究问题**，只能前瞻验证。

可能过早淘汰/低估：GBSA的NUS收益、M2 P1能量信号、Q1/Q2/Q4/Q5/Q6 selected-quality信号、FN-PRA部分折中；但各自还受独立复现、普通MLP对照、组成有效性或四候选成本限制。TCL Formal、DML、激进Corrector、Residual Reuse等不能这样“救回”。

## 17. Risk of metric redefinition / cherry-picking

存在明显post-hoc风险：已经看过GBSA +41%/+59%与85014后讨论新体系，即使理由合理，本报告仍是结果知情的评价重构，绝不能伪装成原预注册。

论文应写：“原方案在预注册的几何修复标准下未通过；回顾性分析发现材料发现率信号，因此提出区分发现主终点与物理保护的新前瞻研究。”附录完整保留原FAIL/BORDERLINE、旧规则、失败样本与新旧比较对象。

不能重标PASS：TCL Formal、V2 Formal、GBSA P1、V3独立64，以及旧Gating、质量adapter/reranker等原No-Go；也不能把Adaptive的旧CONFIRMED解释成按新严格superiority得到的确认。

## 18. 如何避免 cherry-picking

在看任何新候选结果前冻结：研究问题、唯一候选checkpoint、baseline、CFG/条件/样本预算、primary方向、全部NI margins、有限guardrails、统计单位、缺失失败处理、停止规则与N上限。历史数据可用于形成假设和保守估计波动，但失去再次充当独立确认集的资格。

不改变Stable阈值、reference/matcher、去重集合、力的初末态或加权法来“过线”；不删85014，不按有利结果挑trim比例，不只报告最优32。预先排除所有训练、dev、validation、formal、A0兼容以及根因probe用过的seed；只改名字/重新编号旧结构不算新seed。

统计以配对随机seed为主；同seed生成不同化学组成要诚实说明。多个timestep、重复forward、bootstrap重复、四候选pool不能冒充独立seed。seed85014不得从未来风险定义中专门排除，也不能把6.342设置成刚好例外。

本轮CI为描述性、未经跨路线选择校正。20,000次重采样固定seed20260908；报告引用“原CI”时保持原报告的bootstrap配置。NUS/Unique的逐seed标签依赖整批去重：对已赋标签bootstrap只是条件于当前集合的近似不确定性，不是新生成集合Unique/NUS的完整采样分布；全零差CI [0,0]更不是总体完美等价证明。未来固定评价panel规模及确定性去重，预定义panel/重复簇敏感性与配对离散区间；严禁把bootstrap抽到同一结构两次当新的生成重复。

## 19. 推荐未来 Primary success criteria

针对**冻结GBSA的“提高发现率”问题**，建议下一次事先选择以下唯一方向，不能结果后在E或NUS之间切换：

| 主终点 | 正式确认要求 |
|---|---|
| NUS superiority | ΔNUS点估计至少+5pp，且双侧95%CI下界>0 |
| E-hull NI | ΔE-hull双侧95%CI上界<=+0.010 eV/atom |
| 联合要求 | 两项都满足，并通过已冻结的物理NI保护；不是任选一项满足 |

+5pp相当于每100次尝试至少多5个有效发现；10meV/atom是相对100meV稳定窗口的保守工程/研究容忍量，**是建议的practical margin，不是MatterSim误差界、DFT可分辨极限或领域公认常数**。应按论文实际发现成本和可接受损失在新数据揭盲前确认。这里没有为适配GBSA放宽到其CI上界+0.038；按此建议现有GBSA仍不能宣称双主通过。

能量型方法可在另一独立预注册中设E superiority（例如至少下降10meV/atom）+NUS NI -5pp，但本轮GBSA不采用事后OR逻辑。加速型方法的主目标应是实际speed superiority，E/NUS均NI；不能靠极宽-10pp发现损失容忍线轻易命名“保质”。

报告效应量、CI、逐seed方向、分布及实际收益，而不是p-value排行榜。Primary/guardrails是预定义交集要求；探索性额外比较/多臂比较另标并适当控制多重性。n不足则INCONCLUSIVE，不把无显著差异当NI成功。

## 20. 推荐 RMSD guardrail

首选**绝对均值差margin**，不以相对倍数为主要门槛。C0 RMSD很小时倍数不稳定；不同匹配失败惩罚也会扭曲纯比例。

建议沿用早于GBSA的项目尺度：Δmean RMSD容忍上限+0.020Å；报告RMSD>0.5Å事件、匹配失败率以及正常匹配子集的敏感性，但不删除匹配失败的主评价惩罚。0.5Å仅是跨路线既有严重变化告警尺度，不是所有晶体的统一物理相变边界。

统一三态：伤害方向CI上界<=margin→该项NI SAFE；CI下界>margin→该项material degradation得到支持；其余BORDERLINE。另有重复真实无效结构/不可恢复数值失败时走技术有效性停止规则。SAFE不要求均值改善，允许例如小幅正Δ且上界仍在容忍内。

## 21. 推荐 Atomic-force guardrail

使用**结构等权的原子力均值 + 结构MaxF尾部**，而不是只比原子池均值或中位数。所有主要均值保留重尾；中位数、全LOO和预先声明的稳健均值仅做辅助解释。

可将均值比上限1.20作为既有工程尺度的候选NI保护，同时报告绝对均值差。来源是早期V2对mean maximum force的1.20保护；将其用于atomic-force mean属于本报告提出的迁移建议，不是该指标已经被验证的容忍界。这个20%不是物理常数，现有文献并未提供适用于所有元素/模型的统一容忍比；不能声称已经科学认证。若未来C0均值接近零，比例不可辨识，应在揭盲前切换为经物理需求确定的绝对尺度，而不是当场调阈值。

本轮历史C0抽样波动（同cohort经验分布独立抽两组，不是真实新增实验）：n32时V2/TCL原子力均值差95%范围约±0.105/±0.089 eV/Å，n256约±0.036/±0.031。它说明估计困难、要规划精度，**不能据此把允许伤害扩到这些数值**。不同cohort不盲目合池；历史差异同时含化学分布、实现和评价差异。

1.409及CI[0.571,2.660]既不满足1.20上界保护，也没有下界>1.20，因此GBSA是BORDERLINE；没有为它设1.45或1.60“刚好通过”。

## 22. 推荐 Max-force tail guardrail

优先预定义可解释事件率，而不是n32的P99比或单一最大值。建议保留早于GBSA的项目事件：

| 保护对象 | 候选NI容忍伤害（方法−C0） | 角色 |
|---|---:|---|
| P(初始MaxF>1 eV/Å) | +5pp | 常见高力事件保护 |
| P(初始MaxF>2 eV/Å) | +2pp | 更严重尾部保护 |
| Mean MaxF、P95 | 报完整值/CI，作分布补充 | 避免漏掉尾部形状变化，但不重复设多个近似同义“必须改善”门槛 |
| P99、dataset max | 定位个案及影响 | n32不作单独总体灾难判定 |

1/2eV/Å远高于0.05的目标末态力，是初始高力负担指示器，不是通用不可弛豫边界。+5/+2pp是既有项目容忍尺度的候选方案，不是由85014拟合；若材料用途要求更严，应在新结果之前收紧。不能把各类元素不同的力分布解释为同样的合成风险。

事件率使用适合稀疏配对数据的score/exact/保守区间；不要在两臂零事件时用普通bootstrap得到[0,0]并宣布安全。零事件n64的单侧95%上界仍约4.6%，n256约1.16%；所以64不能凭零事件证明真实风险<2%。这不是自动要求扩到任意大样本：提前做精度规划，若预定Nmax仍不能界定风险，就诚实以BORDERLINE结束。

## 23. 推荐 Relaxation guardrail / diagnostic

steps、>200、>400、P95和耗时主要作诊断/效率；不要求GBSA比C0都更少。steps受优化器、cell filter、batch并行与初态影响，不能直接等同生产GPU时间。

真正需要硬保护的是结构无效、NaN/Inf、无法完成的模型评价，以及在固定预算下的非收敛率；未来把末态原子力、广义cell优化收敛和“程序完成”分开报告。>200/>400不充分定义非收敛；反过来一次耗400步但正常完成也不自动判方法物理崩溃。

固定budget/optimizer/fmax，所有失败保留在尝试次数分母内，NUS按失败0处理；不可用丢弃失败后的平均E声称总体质量改善。E仅在可评价样本统计时，必须同时报告覆盖率和预注册的失败敏感性/保守分析，不临时补一个有利的有限能量。

## 24. 三组硕士论文方案评分

1–5分；高分均代表更有利。风险分高=风险低，工作量分高=剩余工作少。分数是基于本项目证据和论文定位的判断，不是实验测量，也不使用事后加权总分制造“客观冠军”。

| 第一创新均为Adaptive CFG | 创新性 | 模型/训练层 | 实验完整性 | 论文叙事 | 低风险 | Formal成熟度 | 指标证据强度 | 答辩可解释性 | 剩余工作少 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +GBSA-TCL | 4 | 5 | 2 | 5 | 2 | 1 | 3 | 3 | 2 |
| +E3-PCR | 3 | 2 | 5 | 4 | 5 | 5 | 4 | 5 | 5 |
| +Corrector V2 | 4 | 4 | 4 | 4 | 3 | 3 | 3 | 4 | 3 |

GBSA最符合“训练层提升发现率”的愿望，但尚无新框架Formal与稳定尾部证据；Adaptive+GBSA叠加收益也未验证。E3是现有证据最成熟、交付风险最低的组合，短板是后处理定位；它的5分Formal指E3单独正式确认，不是组合也已有Formal256。V2有真实模型蒸馏和速度贡献，但Formal BORDERLINE及后续RMSD不一致降低“保质”可信度。

若问题是“今天就定稿，哪条最稳”，E3应领先；本轮问的是“下一步值得投入哪一条研究”，考虑模型/训练层偏好，值得对冻结GBSA进行一次有限、严格的新验证。这两个判断不矛盾。

## 25. Recommendation — 第一推荐第二创新

**选择 B：重新把冻结GBSA-TCL列为第二创新的主要研究候选，用新预注册和全新seeds验证。**

理由不是“原FAIL判错”，而是独立P1已经出现+25pp NUS、E点值下降约21meV/atom的实质发现信号，并且P0同方向；它不像TCL Formal那样没有对C0的主要收益。严重力事件当前更支持风险未定而非总体灾难；GBSA也确实有条件adapter/atomic-head训练改变生成过程，不仅是后处理。

限制为一次冻结方法的前瞻验证，不恢复无尽TCL变体搜索，不按新seeds重调force margin；若收益不复现或保护无法确认，停止该主候选。**目前还不能将GBSA写成已成立的第二创新。**

## 26. Fallback

**唯一fallback：C，停止TCL系新增探索，使用E3-PCR作为quality-preserving physical robustness refinement。**

已有正式256、always-on消融、门控有害干预减少和A0组合独立64证据，科研不确定性明显低于重启另一条新结构路线。论文明确贡献边界：保留材料发现质量并改善生成结构的初始物理稳健性；不宣称NUS提升，也不把129参数gate说成MatterGen主干创新。

V2保留为有价值的加速研究/负面边界章节，但不是这次的第二fallback。也不同时推荐新结构E，以免决策再次发散。

## 27. 是否还值得运行 Atomic-only

**当前不值得作为下一优先动作，本轮不运行。** 当前最有价值的问题是冻结GBSA的发现率信号能否跨新seed复现；先开Atomic-only会引入新的训练方法、选择自由度和成本，无法替代这项验证。

Atomic-only未来可作为有明确机制问题的受控消融，但应在GBSA确认有效或有证据定位条件adapter导致风险之后另行预注册，不能因单个85014直接推断“只训atomic必然解决”。冻结几何head也不意味着几何score不变，共享条件输入仍会影响其输出。

## 28. 是否值得重新验证 GBSA

**值得，但值得的是验证假设，不是给已知结果补一张PASS。** P1的晚期position drift较TCL减少74.76%、cell减少66.35%、atomic减少64.35%，说明机制部分复现；它没有自动带来力均值修复，说明score一致性与终态物理风险并不等价。

GBSA约4.25M可训练参数（条件模块+atomic head），不是完全atomic-only；其余主干/几何head冻结仍不能保证采样分布不变。未来风险分析优先考察初态局部环境、元素/化学系统分布和独立目标偏离，不通过继续增加静态测试解决科学不确定性。

## 29. 如果重新验证：阶段、全新seeds、指标和GO/FAIL

以下仅为**建议协议，未执行也未启动任何队列**。每阶段seeds按全项目历史并集排重；85000–85031、84000–84007、83000–83255及所有早期训练/调参/组合/失败实验seed均永久不可再充当独立确认集。

| 阶段 | 样本/臂 | 目的与冻结条件 | 决策 |
|---|---:|---|---|
| 新development/诊断 | 32，全新D集合 | 冻结现有checkpoint与采样，不训练；仅观察风险和可运行性。事前确定第19–23节所有尺度；不能用这32个结果调margin | 点估计ΔNUS>=+5pp、ΔE<=+0.010；未见支持超margin的伤害或不可恢复结构异常→进入独立验证；否则停，不能改候选后沿用这批作为独立证据 |
| 独立validation | 64，全新V集合，与D及全部历史隔离 | 同一冻结checkpoint，C0/GBSA paired；真实正式pipeline，全样本报告 | 同样满足主效应方向/实用筛选线，且没有确认material伤害→允许进入正式确认；guardrail CI宽可标BORDERLINE晋级，但不叫SAFE。主信号未复现→No-Go主候选 |
| 正式确认 | 256，全新F集合 | 在揭盲前固定Nmax=256，不按中途结果扩样、换seed或调参 | ΔNUS>=+5pp且CI下界>0；E差CI上界<=+0.010；全部预注册保护NI成立→未来新protocol成功。否则FAIL to confirm；区分明确伤害与INCONCLUSIVE |

两臂计划总共2×(32+64+256)=704次生成与对应评价，都是**尚未执行的计划数量**，非本轮进度。老P0/P1不纳入新Formal的分母或CI。冻结方法已有真实smoke，不需要为了“完整性”再重复一轮低价值mock/GPU smoke；若未来环境/实现真的改变，仅补4–8必要smoke，且与确认seeds分离。

停止规则：没有主信号时不因为机制drift好就晋级；均值或事件率的CI下界超过伤害margin，或出现真实数值/结构无效性，先停止确认并定位，不将坏样本删掉继续过门槛。CI宽但未证实安全，最多按上表到预定Formal，最终仍宽就BORDERLINE/未确认，不能无限增加N直到PASS。

256不是保证充分：历史C0 E/NUS重尾和离散性足以使区间仍宽。基线经验null两组n256的NUS差范围约±7–8pp；n32约±19–22pp。该抽样仅用于认识精度，不包含候选的真实pair相关，也不是替代正式power分析。E3 cohort含高能极值，波动比其他cohort更大，不能把所有C0粗暴合池后调整容忍线。

若要在论文中声称“Adaptive CFG与GBSA可叠加”，还必须另行冻结组合检验：至少比较C0、A0、GBSA、A0+GBSA同一批全新seed，组合主要对A0比较，并控制多个主比较；不能把两次独立章节收益相加成组合收益。这个可选四臂问题会增加预算，当前两臂计划及旧P1均不提供组合成功证据。本轮没有安排或执行组合实验。

## 30. 最终一句话结论

**把材料发现主指标与物理保护分层是合理的，但不能用新标准重写旧FAIL：给冻结GBSA一次有停止条件的全新seed前瞻验证机会，若不能确认则转用已具正式与独立组合证据的E3-PCR。**
