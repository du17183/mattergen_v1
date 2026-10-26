# Innovation 2 候选长名单

## 排序口径

所有分数为 1–5。`ImplementationRisk` 与 `ComputeCost` 越高越危险/昂贵，其余越高越好。Rank 不按简单总分机械排序：已通过的本地动机证据、一月内能否完成、与公开工作是否直接撞车是硬 gate。完整机器可读分数见 `candidate_ranking.csv`。

| Rank | Candidate | Core Problem | Training | Novelty | Risk | P0 Cost | Recommendation |
|---:|---|---|---|---:|---:|---|---|
| 1 | KVF-D3PM | atomic reverse posterior 无全局价态/电中性可行性 | TRAINING_FREE | 4 | 3 | 32 paired | Tier 1：唯一建议立即跑 |
| 2 | FP-PC | pos/cell proposal 不保持 d_min、V/N 可行域 | TRAINING_FREE | 3 | 3 | 16 paired | Tier 1：Top1 失败后的下一项 |
| 3 | MC²-Guidance | 多 property residual 可能冲突 | TRAINING_FREE | 3 | 4 | 8-state diagnostic → 16 | Tier 2：只准先诊断 |
| 4 | DCC | noise scaling 未显式使用 V/N | HEAVY_RETRAIN | 3 | 5 | diagnostic → ≤2000 steps | Tier 3：本月不做 |
| 5 | LGCP | cell update 非 intrinsic SPD/Gram step | TRAINING_FREE | 3 | 5 | 8 → 16 | Tier 3：高风险 |
| 6 | HFG | cheap constraints 的层级/冲突 | TRAINING_FREE | 1 | 3 | 8/16 | Tier 3：与 Scout-Matter 撞车 |
| 7 | AGCP | atomic posterior 与局部元素几何不一致 | TRAINING_FREE | 3 | 4 | 既有 diagnostic 已失败 | Reject current evidence |
| 8 | SSPR | requested symmetry 不被 sampler 按构造保持 | LIGHT_FINETUNE | 2 | 5 | 8 | Tier 3：文献拥挤 |
| 9 | CETG | 单 surrogate correction 可被 teacher exploit | TRAINING_FREE | 3 | 5 | 8 endpoint | Tier 3：过贵 |
| 10 | RTSR | late field outlier 可能需局部 re-noise | TRAINING_FREE | 3 | 5 | 既有 64 轨迹 | Reject diagnostic |
| 11 | DOTR | residual/entropy 可能识别质量长尾 | TRAINING_FREE | 2 | 5 | 既有 64 轨迹 | Reject diagnostic |
| 12 | DASU | 字段更新可能有稳定先后依赖 | TRAINING_FREE | 3 | 5 | 既有 64 轨迹 | Reject diagnostic |

## Tier 1

### 1. KVF-D3PM — KL-Valence-Feasible Atomic Reverse Diffusion

**与已有工作的差异。** CrysVCD 在 diffusion 之前生成 valence-balanced composition；generic CDD/Primal-Dual decoding 已做 constrained discrete diffusion。KVF-D3PM 的增量限定为：在 MatterGen 已有 joint denoiser 内，对 variable-site `pθ(A0|At,Xt,Lt,c)` 的全局 composition event 作条件化，再走原始 D3PM posterior；不新增 composition model，不做 endpoint rejection。

- **Hypothesis：** 在 late atomic reverse diffusion 中，将 factorized x0 posterior 条件化到至少一个可行 oxidation-state/charge-neutral count vector，可把 composition validity 提高 ≥12.5 pp，而不显著损害 E-hull/NUS/property target。
- **Zero-training diagnostic：** 已通过动机 gate：E3 Formal256 有 54/256（21.094%）chemistry-invalid-but-geometry-valid；其他 32-cohort 为 6.25%–18.75%，说明 prevalence 会波动。先缓存 Formal256 formula/count 的 SMACT 失败类型；无需新模型 forward。
- **Implementation：** 修改/包裹 `mattergen/diffusion/d3pm/d3pm_predictors_correctors.py::D3PMAncestralSamplingPredictor.update_given_score`。在 `class_probs` 与 `sample_and_compute_posterior_q` 之间，对每个结构构造 `q*(A0) ∝ ∏_i p_i(A_i)·1[F(count(A0))]`。小 chemical system 用 count-vector forward/backward DP 得 conditional site marginals；一般系统用固定-width、确定性 top-k count beam。只在预注册 late window 激活，feasible support 空或数值异常时逐结构 exact fallback 到官方 probs。SMACT/oxidation feasibility 按 count tuple 缓存；不改 cell/pos、不额外调用 GemNet。
- **Baseline/control：** official C0；matched endpoint rejection（同样 SMACT，但不改 reverse process）；entropy-matched logit-temperature control。三者共享 seeds/initial noise。
- **P0 sample size：** 32 paired；若 official baseline 中 chemistry-invalid <4/32，则记 `UNDERPOWERED_PREVALENCE`，不调参、不宣称 FAIL/GO，改用预冻结的 invalid-rich historical seed block只做 replay feasibility audit。
- **Primary metric：** composition validity（唯一 headline）；同时报告 joint structure+composition validity。
- **Guardrails：** E-hull 增量 ≤+0.01 eV/atom；NUS drop ≤5 pp；MaxF mean worsening ≤10%；mag/property MAE worsening ≤15%；Novel/Unique 只作描述但不得出现明显 collapse。
- **GO / BORDERLINE / FAIL：** GO = composition validity +≥12.5 pp（至少少 4 个 invalid）且全部 guardrail 通过，并优于 matched endpoint rejection 的“有效且可用结构数”；BORDERLINE = +6.25 pp 或 guardrail 仅一项轻微越界；FAIL = ≤0 pp、posterior collapse 或任一主要 guardrail 明显越界。
- **Kill condition：** constrained beam 退化为最终 rejection；>10% structures 无 feasible support；sampling wall time >2×；或收益完全可由 entropy lowering/endpoint rejection解释，立即停止。

### 2. FP-PC — Feasibility-Preserving Joint Predictor/Corrector

**与 E3-PCR 的差异。** E3-PCR 在生成结束后，用 CHGNet force 对 position 做 learned-gated refinement；FP-PC 不调用 energy/force teacher，不学习 gate，不设 target，在每个官方 pos/cell proposal 后只防止跨越一个预注册几何 barrier，并保持 pos/cell 联合缩步。

- **Hypothesis：** 对 late pos/cell proposal 采用 joint barrier line search，阻止 `d_min` 和 `V/N` violation 恶化，可降低 as-generated MaxF/RMSD，而不改变 score-call 数和分布指标。
- **Zero-training diagnostic：** 已通过必要但非因果 gate：minimum periodic distance 对 final MaxF 为 cal/val −0.538/−0.683，对 E-hull −0.565/−0.587，对 RMSD −0.605/−0.598；V/N 对 MaxF −0.475/−0.457。元素半径归一化指标未复现，因此 P0 只用绝对 d_min 与 V/N，不引入 covalent-radius heuristic。
- **Implementation：** 在 `mattergen/diffusion/sampling/pc_sampler.py` 的 corrector/predictor `samples_means` 生成后、`_mask_replace` 前加 experiment-local proposal filter。对 pos+cell 使用共同 `α∈{1,1/2,1/4,0}` 插值（fractional minimum image + polar lattice）；只有 proposal 新进入 hard boundary 或增加 barrier 时缩步。early phase 不激活；`α=0` 精确回退旧 joint state；atomic field 始终走官方 D3PM。
- **Baseline/control：** official C0；只裁 position 的 control；只裁 cell 的 control；简单 fixed max-step clipping。joint α 是证明“constraint-preserving coupling”而非普通 clipping 的关键 ablation。
- **P0 sample size：** 16 paired；只有方向明确且无 guardrail 事故才扩到 independent 32。
- **Primary metric：** pre-relax MaxF mean；RMSD 为第二 primary。
- **Guardrails：** E-hull ≤+0.01 eV/atom；NUS drop ≤5 pp；composition validity 不降；property MAE worsening ≤15%；score calls 完全相同，wall time overhead ≤15%。
- **GO / BORDERLINE / FAIL：** GO = MaxF mean 下降 ≥10% 且 RMSD 不恶化 >5%，或 RMSD 下降 ≥10% 且 MaxF 不恶化 >5%，guardrails 全过；BORDERLINE = 主指标改善 5%–10%；FAIL = 两主指标无一 ≥5% 或任何分布 guardrail 越界。
- **Kill condition：** barrier activation >50%、大量 `α=0`、效果来自全局缩小步长 control 同样达到，或 FP-PC 不优于 E3-G 的 force reduction 方向，停止作为替代创新。

## Tier 2

### 3. MC²-Guidance — Multi-Condition Field-wise Conflict Resolution

- **Hypothesis：** multi-property sampling 中，按 property 分解的 CFG residual 在至少一个 field/time band 存在可复现负 cosine，而 field-wise constrained projection 能提高 joint target hit without additional collapse。
- **Zero-training diagnostic：** 当前没有所需 tensor，不能从 CSV 推断。先取 8 个固定 noisy states × 3 time bands，分别保持单个 property condition，计算每条件在 atomic logits/pos/cell score 的 residual cosine、norm ratio、与 total denoising score cosine；calibration/validation seed 各半。只有同一 pair/field 在两半负 cosine rate ≥60%、median cosine ≤−0.2 才通过。
- **Implementation：** 仅 diagnostic 通过后，在 `mattergen/diffusion/sampling/classifier_free_guidance.py::_score_fn` 暴露 per-condition residual；对发生冲突的 field 做对称 PCGrad/min-norm projection，再加回 unconditional score。不要改 Innovation 1 的时间/幅度 controller。
- **Baseline/control：** official multi-condition CFG；相同 forward 数的 residual average；全 field PCGrad；仅 diagnostic 指定 field PCGrad。
- **P0 sample size：** diagnostic 8 states；sampling P0 16 paired。
- **Primary metric：** joint target hit rate；第二 primary 为 NUS-within-target。
- **Guardrails：** 单目标 hit 各自不得下降 >10 pp；E-hull ≤+0.01；NUS overall drop ≤5 pp；wall time 必须如实报告（预期 2–4×）。
- **GO / BORDERLINE / FAIL：** GO = joint hit +≥12.5 pp 且无单目标/质量 guardrail 失败；BORDERLINE = +6.25 pp；FAIL = conflict diagnostic 不过或 sampling 无 gain。
- **Kill condition：** 若 residual 主要共线、冲突只出现在一个 split，立即停止；不得靠改 adaptive CFG scale 救活。与 Innovation 1 的论文边界说不清也视为 kill。

## Tier 3

### 4. DCC — Density-Calibrated Corruption

- **Hypothesis：** 在相同 N 下，用 V/N 校准 position/cell noise magnitude，可减少结构规模导致的 denoising mismatch。
- **Zero-training diagnostic：** bivariate V/N–MaxF/RMSD 在两 split 稳定，但 partial-rank 对控制变量敏感：控制 N 后 MaxF 为 cal −0.194、val −0.408；再控制 d_min 后变为 +0.073/+0.211。该证据不足以直接实现。下一步只能在训练数据上做 noise-normalized score residual 分层，不能用最终质量相关代替。
- **Implementation：** 若新 diagnostic 通过，在 `NumAtomsVarianceAdjustedWrappedVESDE.std_scaling` 与 `LatticeVPSDE` 增加冻结的 density bin calibration，并从官方 checkpoint 做 ≤2000-step light feasibility finetune；直接零训练改 corruption 概率解释不成立。
- **Baseline/control：** official N-only scaling；density scaling without finetune；density scaling + same-step finetune。
- **P0 sample size：** 先全历史只读分层；训练 smoke ≤2000 steps；generation 16。
- **Primary metric：** 分 density quartile 的 MaxF gap；第二 primary RMSD gap。
- **Guardrails：** overall E-hull ≤+0.01；NUS drop ≤5 pp；各 quartile composition validity 不降 >5 pp。
- **GO / BORDERLINE / FAIL：** GO = worst–best density quartile MaxF gap 降 ≥25% 且 overall MaxF 不坏；BORDERLINE = 10%–25%；FAIL = diagnostic 不稳定或需 full retrain。
- **Kill condition：** score residual 不随 V/N 稳定变化、2000 steps 不收敛，或收益由 d_min 完全解释，停止。

### 5. LGCP — Log-Gram Cell Predictor

- **Hypothesis：** 把 lattice proposal 映射到 `G=LLᵀ` 的 log-Euclidean tangent space并限制 geodesic step，可减少 abnormal cell/relaxation displacement。
- **Zero-training diagnostic：** 当前 late cell residual 没有跨 split 关联 final tail；仅有表示层面的动机，未过 empirical gate。
- **Implementation：** experiment-local `LatticeAncestralSamplingPredictor` wrapper：polar gauge 后构造 SPD Gram，做 `logm(G)` step clipping，再由 symmetric square root 回到 lattice；不改变 atomic/pos。
- **Baseline/control：** official polar projection；Euclidean Frobenius clipping；Log-Gram clipping。
- **P0 sample size：** 8 deterministic numerical replay，确认 equivariance/finite/determinant；再 16 paired。
- **Primary metric：** RMSD；abnormal-cell rate 只在 baseline 非零时作为第二 primary。
- **Guardrails：** MaxF worsening ≤10%；E-hull ≤+0.01；NUS drop ≤5 pp；condition target MAE ≤+15%。
- **GO / BORDERLINE / FAIL：** GO = RMSD −≥10% 且 controls 不等效；BORDERLINE = −5%–10%；FAIL = score mismatch/quality退化。
- **Kill condition：** numerical replay 改变 rotation-equivalent structure、determinant异常，或需要 retrain 才稳定，停止。

### 6. HFG — Hierarchical Feasibility Guidance

- **Hypothesis：** safety > chemistry > user objective 的 lexicographic constraint hierarchy，能比加权求和更少发生 constraint conflict。
- **Zero-training diagnostic：** 可在既有 final structures 统计 violation 共现；但本地只有 d_min/V/N 稳定，coordination/element-radius descriptor 未复现，尚不支持多层 hierarchy。
- **Implementation：** sampler 外部 differentiable losses，逐层投影低优先级 gradient 到高优先级可行切空间；late activation。目标函数接口与 Scout-Matter 相邻。
- **Baseline/control：** official；Scout-style weighted sum；同 constraint 的 lexicographic projection。
- **P0 sample size：** 8 smoke，若无明显 novelty/效果再不扩；最多 16。
- **Primary metric：** high-priority violation rate；第二为 user target hit。
- **Guardrails：** E-hull/NUS/MaxF 与对应 baseline 的预注册非劣界。
- **GO / BORDERLINE / FAIL：** 只有同时优于 matched weighted sum 且 overhead 可接受才 GO；否则 FAIL。
- **Kill condition：** 若实现退化为调 loss weights/activation schedule，或无法与 2026 adaptive constraint guidance 区分，立即停止。

### 7. AGCP — Atomic–Geometry Compatibility Posterior

- **Hypothesis：** 在 late D3PM 中按当前局部几何对 element logits 做小幅 likelihood-ratio-capped reweight，可减少 element–environment mismatch。
- **Zero-training diagnostic：** 已失败。最小 covalent-radius normalized distance 对 E-hull/RMSD 的 calibration ρ=−0.457/−0.401，validation 仅 −0.196/−0.192；p10 接近零相关，soft overlap 未复现。
- **Implementation：** 若未来出现独立、冻结且复现的 bond-valence descriptor，才在 `class_probs` 前加 `|Δlogit|≤δ` 的局部 compatibility bias；当前不实现。
- **Baseline/control：** official；atomic-logit temperature；不含元素类型的 d_min penalty。
- **P0 sample size：** 当前 P0 在 diagnostic 阶段结束；若新 descriptor 通过，只准 8 paired。
- **Primary metric：** descriptor violation 与 MaxF 二选一，不可事后挑选。
- **Guardrails：** composition validity、E-hull、NUS、property target。
- **GO / BORDERLINE / FAIL：** 当前状态 `FAIL_DIAGNOSTIC`。
- **Kill condition：** 任一 split 不复现即停止；不得重新命名 Cell–Atom Fusion。

### 8. SSPR — Symmetry-Subspace Predictor Retrofit

- **Hypothesis：** 对 space-group-conditioned checkpoint，把 pos/cell proposal 投影回请求的 Wyckoff/log-lattice subspace，可提高 symmetry adherence。
- **Zero-training diagnostic：** 先在既有 space-group-conditioned outputs 比较 requested vs detected SG，并量化一次 projection displacement；当前项目没有证明这是一项主要失败源。
- **Implementation：** 需要 spglib/Wyckoff parameterization、site correspondence 和 possibly ≤2000-step finetune；主要触及 pos/cell predictors。
- **Baseline/control：** official space-group checkpoint；endpoint symmetrization；during-sampling projection。
- **P0 sample size：** 每个 2–3 个 SG，共 8 structures。
- **Primary metric：** requested-SG adherence；第二为 projection RMSD。
- **Guardrails：** MaxF/E-hull、NUS、relaxation success。
- **GO / BORDERLINE / FAIL：** 只有 during-sampling 显著优于 endpoint symmetrization且质量守住才 GO。
- **Kill condition：** site mapping 不唯一、projection 大位移，或必须重训完整模型。由于 DiffCSP++/WyckoffDiff/SymmCD/DynaCrys，novelty 默认低。

### 9. CETG — Cross-Evaluator Consensus Tail Guard

- **Hypothesis：** 只有两个独立 UIP 对改善方向一致时才接受局部 correction，可降低单 teacher exploit。
- **Zero-training diagnostic：** 对既有 C0/E3 proposal，用 CHGNet 与 MatterSim/Equiformer family 比较 per-structure improvement sign；若 agreement 与实际 relaxation outcome 无关则停止。
- **Implementation：** 仅 late/endpoint 少量候选步，同时计算两 evaluator 的 energy/force direction，agreement 时做 trust-region step；不训练 probe。
- **Baseline/control：** E3-G；CHGNet-only；第二 evaluator-only；consensus。
- **P0 sample size：** 8 endpoint paired，不能直接跑 1000-step online evaluator。
- **Primary metric：** cross-evaluator win rate；第二为 worst-case MaxF。
- **Guardrails：** E-hull/NUS/RMSD；wall time ≤3× E3-G 才有继续价值。
- **GO / BORDERLINE / FAIL：** 两 evaluator improvement agreement ≥75% 且 consensus 优于单 teacher 才 GO。
- **Kill condition：** evaluator 不独立、显存/时间过高、或变成 E3-PCR 的昂贵 ensemble，停止。

## Reject：zero-training diagnostic 已经给出停止结论

### 10. RTSR — Residual-Triggered Selective ReNoise

- **Hypothesis：** late 单字段 residual 异常能定位 bad trajectory，局部 re-noise 后可重新进入高质量 basin。
- **Zero-training diagnostic：** 已失败：late residual 与 MaxF/RMSD/E-hull 没有跨 split、同方向稳健关系；validation atomic entropy 的强相关未在 calibration 复现。
- **Implementation：** 原设想是在 `_denoise` late window 对单字段调用对应 marginal noising 后短程重去噪；当前禁止实现。
- **Baseline/control：** official；matched extra joint denoise；random-trigger single-field re-noise。
- **P0 sample size：** 64 historical trajectories 已完成 gate，0 new generation。
- **Primary metric：** tail detector AUROC/ρ，而不是重采样后的事后最优指标。
- **Guardrails：** score calls、NUS、E-hull。
- **GO / BORDERLINE / FAIL：** `FAIL_DIAGNOSTIC`。
- **Kill condition：** 已触发；不再调 threshold。

### 11. DOTR — Direct-Observable Tail-Risk Controller

- **Hypothesis：** score norm、corrector displacement、posterior entropy 等无需 probe 的在线量可识别 final bad tail。
- **Zero-training diagnostic：** residual/entropy 不跨 split；唯一稳定信号是 d_min/V/N，已被更简单、机制更清楚的 FP-PC 吸收。
- **Implementation：** 原设想是 risk gate 控制额外 correction；当前禁止实现新的 learned/heuristic controller。
- **Baseline/control：** official；random gate；direct hard geometry barrier。
- **P0 sample size：** 64 historical trajectories 已足够否定当前特征集。
- **Primary metric：** held-out tail AUROC；现有证据不足。
- **Guardrails：** calibration/validation 一致性、额外计算。
- **GO / BORDERLINE / FAIL：** `FAIL_DIAGNOSTIC`。
- **Kill condition：** 已触发；不得用 validation-only atomic entropy 追参。

### 12. DASU — Dependency-Aware Sequential Updates

- **Hypothesis：** atomic/cell change 对下一步 position instability 有 timestep 之外的稳定先行作用，因而 sequential re-score 优于 simultaneous update。
- **Zero-training diagnostic：** 已失败。raw ρ 为 0.30–0.77，但 timestep-residualized validation 只有 −0.188 到 +0.059，说明共同 schedule 是主要来源。
- **Implementation：** 原设想是 atomic→re-score→pos/cell 或 cell→re-score→pos；会额外增加 forward，当前禁止实现。
- **Baseline/control：** official simultaneous；same-extra-forward no-op；随机 update order。
- **P0 sample size：** 64 trajectories ×55 observations 已完成 gate。
- **Primary metric：** residualized cross-lag / held-out next-step prediction。
- **Guardrails：** score calls 与全质量指标。
- **GO / BORDERLINE / FAIL：** `FAIL_DIAGNOSTIC`。
- **Kill condition：** 已触发；不以逐轨迹 raw correlation 代替去时间趋势结果。

## 分层结论

- **Tier 1：** KVF-D3PM；FP-PC。前者修复 E3 完全不触及的 chemistry gap，后者有直接两 split geometry evidence，但 target 与 E3 更重叠。
- **Tier 2：** MC²-Guidance，仅能先做 per-condition residual diagnostic；不通过就不能实现。
- **Tier 3：** DCC、LGCP、HFG、SSPR、CETG。它们分别受重训、score mismatch、公开工作撞车、文献拥挤或 evaluator 成本限制。
- **Reject：** AGCP（当前 descriptor 未复现）、RTSR、DOTR、DASU。后三项已被本轮 64 轨迹诊断直接杀死。

