# Top 3 最小 P0 设计

## 共同规则

- 本文件是设计，不代表已授权执行；本轮没有训练或生成。
- 每个 P0 在写代码前冻结 seeds、activation window、阈值、primary metric 和 kill condition。
- 实验单位是完整生成结构；使用 paired seeds/common initial noise，但统计不能把 1000 个 diffusion steps 当独立样本。
- P0 只负责 `GO/BORDERLINE/FAIL`，不作论文最终显著性声明；GO 后才设计 independent 64/128，不能直接 Formal256。
- 不复用 P0 outcome 调参数再在同一 seeds 上报告。
- MatterSim/CHGNet 仍是 surrogate；`DFT_VERIFIED=False` 必须保留。

## TOP1 — KVF-D3PM

### 科学问题

MatterGen 的 chemical-system mask 保证“元素属于允许集合”，却不保证这些元素计数存在满足 oxidation-state balance 与 Pauling electronegativity test 的组合。Formal256 的 54/256 chemistry-invalid-but-geometry-valid 是 E3-PCR 无法改变的正交缺口。

### 算法定义

对一个包含 `n≤20` sites 的结构，denoiser 给出 factorized clean-type posterior：

`pθ(A0|At,Xt,Lt,c)=∏ᵢ pᵢ(aᵢ)`。

定义 `F(n(A0))∈{0,1}` 表示 composition count vector 是否通过冻结的 SMACT/oxidation feasibility。理想约束后验为：

`q*(A0) = pθ(A0) F(n(A0)) / Ppθ(F=1)`。

这等价于在 feasible set 上对原 posterior 作 KL 最小条件化：可行 assignment 之间的相对概率不变。实际只需每 site marginal `q*(aᵢ)`，然后把它作为 x0 probability 送入原有 `sample_and_compute_posterior_q()`。

实现分两档，P0 只能选定一档：

1. chemical-system 条件且元素数小：按 count vector 做 exact forward/backward DP，枚举缓存过的 feasibility table。
2. unrestricted/top-k union 较大：固定 `K` 与 beam width `B` 的 deterministic count beam；对 beam 内 joint assignments 归一化并求 site marginals。若 feasible mass/coverage 不足，逐结构回退 official posterior，不允许动态加 B 追结果。

只在 late window `t≤t_on` 激活，因为高噪声时 mask 多、全局 assignment 不稳定。`t_on` 必须由历史 atomic entropy curve 冻结而非 P0 outcome；建议起点 `t_on=0.10`，并以 `0.05` 作一次预注册 sensitivity，不做网格搜索。

### 代码落点

- 核心 wrapper：`mattergen/diffusion/d3pm/d3pm_predictors_correctors.py`，位于 clean `class_probs` 与 D3PM posterior 调用之间。
- feasibility/cache 可放 experiment-local module，调用现有 `mattergen/evaluation/metrics/structure.py::smact_validity` 的同口径规则；不修改 metric 定义。
- sampling config 只加 `enabled/t_on/mode/top_k/beam_width/fallback`；默认全部关闭，保证官方路径不变。
- 单元测试必须覆盖 disabled exact equivalence、all-feasible identity、no-support exact fallback、site permutation invariance、chemical-system mask 不被重新打开。

### P0 arms

| Arm | 作用 |
|---|---|
| C0 | official sampler |
| R-end | 生成后只做 SMACT rejection；用相同总 proposal budget，证明“有效产出率”差异 |
| T-entropy | 调低 atomic temperature，使 endpoint entropy/unique count 与 KVF 接近；排除普通收缩 |
| KVF | late reverse posterior feasibility conditioning |

四 arms 用 32 paired seeds。若算力紧，先 C0/KVF 32；只有通过 primary gate 再补两个 controls，但最终论文前 controls 不能缺。

### 冻结判据

- **Primary：** composition validity。GO 目标 `Δ≥+12.5 pp`。
- **Secondary descriptive：** joint structure+composition validity、feasible posterior mass、fallback rate、count-vector entropy。
- **Guardrails：** `ΔE-hull≤+0.01 eV/atom`；NUS drop ≤5 pp；MaxF mean worsening ≤10%；property MAE worsening ≤15%；runtime ≤2×。
- **GO：** baseline invalid ≥4/32，KVF 至少多 4 个 valid，guardrails 全过，且 usable-valid count 优于 R-end。
- **BORDERLINE：** 多 2–3 个 valid，或只有一个轻微 guardrail 越界；只允许一次 independent 32，不允许在原 seeds 调参。
- **FAIL：** ≤1 个改善、NUS/quality collapse、>10% no-support 或作用等同 temperature/rejection。
- **Kill：** 无法形成 D3PM transition 内的 conditional posterior，只能做最终筛选时，研究贡献不成立。

### 为什么它可在一个月完成

它不训练、不改 GemNet、不增加 score calls。主要风险是 global count feasibility 的组合复杂度，而不是 GPU 训练。第一周可用 synthetic logits 和历史 formulas 验证 DP/beam；第二周实现 disabled-equivalence；第三周 32 paired；第四周独立确认或停止。

## TOP2 — FP-PC

### 科学问题

64 trajectory audit 中，absolute minimum periodic distance 是唯一对 MaxF、E-hull、RMSD 都跨 split 稳定的直接信号；V/N 对 MaxF/RMSD 也稳定。当前 wrap/polar 操作只保证周期和 lattice gauge，不保证 official proposal 不跨越物理可行 barrier。

### 算法定义

令 current joint geometry 为 `s=(X,L)`，official proposal 为 `ŝ=(X̂,L̂)`。定义只读 violation：

- `v_d(s)=max(0,d_floor(t)-d_min(s))`；
- `v_V(s)=distance(log(V/N), frozen_N-conditioned corridor)`。

若 official proposal 没有增加 lexicographic violation，则 `α=1`。否则按 `α∈{1/2,1/4,0}` 检查 `s_α`，选择最大的可行 α。position 用 minimum-image interpolation，cell 插值后走官方 polar gauge；pos 与 cell 使用同一个 α，避免只修一个字段造成新不一致。`α=0` 是旧状态 exact rollback。

这不是能量 guidance：没有 CHGNet/MatterSim gradient，也不增加 denoiser forward；它只改变违反 barrier 的 transition。

### 代码落点

- 在 `mattergen/diffusion/sampling/pc_sampler.py` 中两处 `samples_means → _mask_replace` 之间增加 disabled-by-default hook。
- experiment-local `joint_feasibility_filter.py` 计算 periodic d_min、V/N、interpolation 与 audit trace。
- 先只在 predictor phase + late 10% 激活；corrector phase 是预注册 ablation，不能同时上线后无法归因。
- 单元测试：disabled equivalence、rotation/translation/permutation invariance、α monotonicity、atomic numbers unchanged、score calls unchanged。

### P0 arms

| Arm | 作用 |
|---|---|
| C0 | official |
| Clip-X | 只限制 position step norm |
| Clip-L | 只限制 cell step norm |
| FP-PC | d_min/V/N barrier 下 joint α |

16 paired seeds。先用历史 calibration 冻结 `d_floor(t)` 和 V/N corridor；validation 32 不可用于选阈值。必须记录 activation、α、触发 constraint、被回退 proposal 的 official step norm。

### 冻结判据

- **Primary：** pre-relax MaxF mean；第二 primary RMSD。
- **GO：** 任一 primary 改善 ≥10%，另一项不恶化 >5%；E-hull ≤+0.01、NUS drop ≤5 pp、composition validity 不降、wall overhead ≤15%。
- **BORDERLINE：** 5%–10% primary gain 且 guardrail 全过。
- **FAIL：** 没有 ≥5% gain；或 joint 方法不优于单纯 Clip-X/Clip-L。
- **Kill：** activation >50%、`α=0` 频繁、分布收缩，或需要用 evaluator force 来决定 α——后一情形会退化成 E3-PCR during sampling。

### 与 E3-PCR 的比较门槛

FP-PC 即使对 C0 有改善，也未必比 E3-G 值得。若 32 样本扩展后 MaxF reduction 明显低于 E3 的 23.3%，且没有 composition/validity/速度上的独立收益，它只能作为辅助安全机制，不能替代 Innovation 2。

## TOP3 — MC²-Guidance

### 定位

这是高风险候选，不是低风险保底。它的意义在于研究“条件之间的冲突”，而 Innovation 1 研究“组合 CFG residual 在三个输出字段上的在线幅度”。只有按 property 分解并证明 conflict，二者才有清楚边界。

### Phase 0A：只做 score diagnostic

选择确实兼容的 multi-property checkpoint 和一组同时存在标签/条件的任务。若仓库/检查点无法给同一模型独立切换这些 property conditions，直接 `NOT_APPLICABLE`。

在 8 个固定 noisy states、`t∈{0.8,0.4,0.1}` 上，对每个 condition `c_j` 计算：

`r_j^f = score_f(c_j)-score_f(uncond)`，`f∈{atomic,pos,cell}`。

每对条件报告 weighted cosine、norm ratio、negative-cosine site/structure fraction，以及投影后与 unconditional denoising direction 的夹角。4 calibration / 4 held-out validation states 分开。

通过 gate 必须同时满足：同一个 `(condition pair, field, time band)` 在两半中 negative cosine rate ≥60%，median cosine ≤−0.2，且不是一个 residual norm 低于另一项 1%的数值伪冲突。

### Phase 0B：通过才采样

- 在 `classifier_free_guidance.py::_score_fn` 返回 per-condition residual。
- 只对 diagnostic 指定的 field/time band 做对称 conflict projection；其他字段使用 official composition。
- 禁止复用 Innovation 1 controller 的 EMA/动态强度作为主要机制。

Arms：official multi-CFG、相同 extra forwards 的 arithmetic mean、all-field PCGrad、diagnostic-selected MC²。16 paired seeds。

### 冻结判据

- **Primary：** joint target hit；第二 primary target-constrained NUS。
- **GO：** joint hit +≥12.5 pp，单目标各自不降 >10 pp，E-hull ≤+0.01，overall NUS drop ≤5 pp。
- **BORDERLINE：** +6.25 pp 且 guardrails 全过。
- **FAIL：** score diagnostic 不通过；或 projection 与 average 等效；或 sampling gain 只来自更强总 residual norm。
- **Kill：** 额外 forward 使 wall time >3×、没有真正 multi-property checkpoint、或论文描述无法用一句话区别 Innovation 1。

## 一个月串行决策

| 周 | 工作 | 停止点 |
|---|---|---|
| 1 | 冻结 KVF feasibility 定义；DP/beam synthetic + historical formula audit | 不能形成 transition 内 posterior → 直接 NO |
| 2 | KVF disabled-equivalence、fallback、32 paired generation | baseline prevalence <4/32 → 不追参 |
| 3 | KVF relaxation/evaluation + controls | FAIL → 才转 FP-PC 16；GO → 不分散算力 |
| 4 | KVF independent confirmation 或 FP-PC P0 | 不启动 MC² sampling；最多做 8-state diagnostic |

这个顺序确保“只剩一个月”时不会同时开三条半成品路线。

