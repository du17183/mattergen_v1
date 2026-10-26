# 相关工作审计（截至 2026-09-12）

## 检索边界

本轮联网核对了用户点名的 MatterGen derivatives、XtalPaint、Scout-Matter、Goodfire SCS、FlowMM、CrystalFlow、CrystalGRW、DiffCSP++、WyckoffDiff、CrystalDiT、EquiformerV2/V3、CrysVCD，以及 2025–2026 constrained discrete diffusion、symmetry-aware generation 和 evaluation 工作。优先引用论文主页、OpenReview、期刊页面和官方代码仓库；预印本、软件和研究博客均明确标注，不能和 peer-reviewed 证据混写。

## 直接相关工作

| 工作 | 状态 | 已做到什么 | 对本项目 novelty 的含义 |
|---|---|---|---|
| [MatterGen: A generative model for inorganic materials design](https://www.nature.com/articles/s41586-025-08628-5) | Nature 2025 | 联合生成 atom types、fractional coordinates、lattice；支持 chemistry、symmetry、scalar property 和 multi-property conditioning | 新工作必须是具体 sampler/corruption 缺口，不能把“联合晶体 diffusion”本身当创新 |
| [FlowMM](https://arxiv.org/abs/2406.04713) | ICML-era arXiv 2024 | 在晶体对称性对应的 Riemannian geometry 上做 flow matching，强调更少 integration steps | 直接把 MatterGen 换成 flow matching 是大模型重训，不适合当前一月窗口 |
| [CrystalGRW](https://arxiv.org/abs/2501.08998) | arXiv 2025 | 为 fractional coordinates、atomic types、lattice 使用相应 manifold/geodesic random walks，并报告 DFT 验证 | “用 Riemannian manifold”不是空白；可保留的增量只能是 pretrained MatterGen 上的小型 sampler projection |
| [CrystalFlow](https://www.nature.com/articles/s41467-025-64364-4) | Nature Communications 2025 | flow-based crystal generation；文章也显示稀有 target 上 conditional error 仍明显 | 支持 conditional fidelity 仍是难题，但不支持在本项目重换 backbone |
| [DiffCSP++](https://arxiv.org/abs/2402.03992) | ICLR-era arXiv 2024 | 把 space-group constraint 化为 invariant logarithmic lattice subspace 与 Wyckoff coordinate constraint | log-lattice/symmetry projection 已有强先例；简单移植会是复现，且当前 MatterGen score 未在该子空间训练 |
| [WyckoffDiff](https://arxiv.org/abs/2502.06485) | arXiv 2025 | 用离散、symmetry-based Wyckoff representation 生成结构，按构造保持 symmetry | 直接改 Wyckoff 表示需要重训；不列入一月首选 |
| [SymmCD](https://arxiv.org/abs/2502.03638) | arXiv 2025 | 联合生成 asymmetric unit 与 symmetry transformations | symmetry-aware generation 已拥挤；普通 symmetry regularizer novelty 低 |
| [DynaCrys](https://arxiv.org/abs/2608.07401) | arXiv 2026 | space group、Wyckoff occupations、elements 共演化，并用 legality-constrained decoder | 截至 2026-09，动态 symmetry/legality 已不是无人区；本项目不应追这条重训路线 |
| [CrystalDiT](https://arxiv.org/abs/2508.16614) | arXiv 2025 | 用统一 Transformer 把 lattice 与 atomic properties 作为一个 interdependent system | 说明统一 backbone 是竞争路线，也削弱“再加 hidden fusion”的独创性；本项目 Cell–Atom Fusion 负结果进一步排除它 |
| [EquiformerV2](https://arxiv.org/abs/2306.12059) / [EquiformerV3](https://arxiv.org/abs/2604.09130) | ICLR 2024 / arXiv 2026 | 更高阶 equivariant Transformer；V3 强调效率、smooth cutoff 与 PES 高阶导数 | 它们适合作 evaluator/MLIP 讨论，不是无需重训的 MatterGen sampler 增量；替换 GemNet-T 超出范围 |

## 化学可行性与 discrete constraints

### CrysVCD 已经占据的空间

[CrysVCD](https://arxiv.org/abs/2507.19799)（arXiv 2025）明确指出 oxidation-state balance 被许多 crystal generators 忽略，并使用 transformer elemental language model **先生成 valence-balanced composition**，之后再由 diffusion 生成结构。它还把系统定位为可插拔 valence-constrained design。

因此以下说法不能成立：

- “首次把价态约束用于晶体生成”；
- “SMACT 筛选本身是创新”；
- “先采 composition、后采 geometry”是本项目独有。

本项目仍可能有的增量是更窄的：不新增 composition generator，不做最终 rejection，而是在 MatterGen 已有 D3PM 的每个 late reverse transition 内，把 factorized atom-type posterior 做 KL-regularized/global-feasibility projection，再送入原始 D3PM posterior。这研究的是**预训练联合模型内的 constrained categorical posterior**，不是 CrysVCD 的两阶段 pipeline。

### Generic constrained discrete diffusion 已经占据的空间

- [Constrained Discrete Diffusion](https://openreview.net/forum?id=Es4s9dtCjR)（NeurIPS 2025）把每个 denoising step 写成 constrained optimization，训练自由地施加逻辑/分子等 hard constraints。
- [Search-Augmented Masked Diffusion](https://arxiv.org/abs/2602.02727)（arXiv 2026）在每个 reverse step 从模型 proposal set 内做 informed search，以提升 constraint satisfaction。
- [Primal-Dual Guided Decoding for Constrained Discrete Diffusion](https://arxiv.org/abs/2605.09749)（arXiv 2026）通过 KL-regularized optimization、在线 Lagrange multipliers 和 logit bias 处理多个约束，不需额外 model evaluation。

这些工作意味着“给 D3PM logits 加 constraint penalty”本身 novelty 只有中等。真正可写的贡献必须同时具备：

1. 把 crystal oxidation-state feasibility 表达为 variable-site global count event；
2. 用 forward–backward count DP/受限 beam 近似其 conditional site marginals；
3. 明确保留 MatterGen posterior 的相对似然与 exact fallback；
4. 用 paired experiment 证明不是 endpoint rejection 或温度收缩带来的假提升。

所以 Top1 的 novelty 评分为 4/5 而不是 5/5；若最后只能实现“采几个 composition、SMACT 过滤”，应降为复现并停止。

## 训练自由几何/物理 guidance

[Finetuning-Free Diffusion Model with Adaptive Constraint Guidance for Inorganic Crystal Structure Generation](https://arxiv.org/abs/2604.13354)（arXiv 2026）直接在预训练 crystal diffusion sampling 中加入 differentiable constraints，覆盖 atomic volume、local coordination 与 neighbor motifs。[scout-matter 官方仓库](https://github.com/link-lab3629/scout-matter)进一步显示它是 MatterGen 修改版，提供 training-free coordination、volume 和 volume-per-atom guidance，以及 self-recurrence/back-step。

因此下列候选应判定为低 novelty 或 reject：

- 普通 `minimum_distance loss + gradient guidance`；
- 普通 coordination target；
- 普通 volume/V-per-atom target；
- 再调 guidance weight 或 recurrence steps。

本项目可区分的增量只能是**feasibility-preserving transition**：不把结构拉向某个 target，不反传新 loss，而是对官方 P/C proposal 做最短 line-search/proximal rollback，使其不跨越安全 barrier，并保留 exact official step 作为对照。它仍与 projected diffusion 文献相邻，所以 novelty 仅 3/5；价值来自本地两 split 的直接几何证据与很低实现成本。

[XtalPaint](https://www.nature.com/articles/s41524-026-02090-1)（npj Computational Materials 2026；[官方仓库](https://github.com/psi-lms/XtalPaint)）在 MatterGen 派生模型上做 crystal inpainting，包含 retrained pos-only/time-dependent predictor-corrector。它说明部分字段重建是有效任务，但不能支持本项目的 selective re-noise：本项目 residual-to-tail diagnostic 没有通过，而且 XtalPaint 使用专门重训模型，不是把原 MatterGen 某字段临时重置。

## Self-correcting search 与本地反证

[Goodfire: Using Self-Correcting Search to Accelerate Materials Discovery](https://www.goodfire.com/research/self-correcting-search)（官方研究博客，2026）报告 probe-guided MatterGen 搜索可增加约 27–30% on-target viable candidates。它是相关的外部正结果，但本项目按相同核心前提做 held-out probe gate 时得到 Pearson 0.189、Spearman 0.173、R² −5.276，按预注册规则停止搜索。

结论不是 Goodfire 方法无效，而是**本项目的 checkpoint/target/样本中 probe 不可用**。继续做 hidden probe、MALA/search 或把它改称 tail-risk controller，都属于忽视本地反证。

## Multi-property 与 conflict

- MatterGen 论文已展示 multi-property conditioning。
- [Property-Guided Diffusion for Inverse Design of Crystalline Materials](https://arxiv.org/abs/2607.21849)（arXiv 2026）研究 formation energy、magnetization、hardness 的 property guidance 和 multi-property generation。
- [Multi-property directed generative design through Wyckoff-augmented transfer learning](https://arxiv.org/abs/2503.16784)（arXiv 2025）显式面对 multi-objective 与 sparse labels。
- [Adaptive Constraint Integration / SMOACS](https://arxiv.org/abs/2410.08562)（arXiv 2024）用 property-model gradients 同时优化多目标，并包含 electrical neutrality 等约束。
- generic diffusion 中已有 [PixelAsParam](https://openreview.net/forum?id=2q1Whv1kXL) 对 guidance gradient conflict 做投影，以及 [Composition and Alignment of Diffusion Models using Constrained Learning](https://openreview.net/pdf/4d20b264918657a1d750f822c477dfc409b1af84.pdf) 对冲突要求作 constrained/primal-dual formulation。

因此 PCGrad-like multi-condition guidance 不是算法空白。本项目可能的增量是 MatterGen-specific、按 atomic/pos/cell 三个 output field 分解每个 property residual 的 conflict map；但它与 Innovation 1 的 multi-field adaptive CFG 容易混淆，又需要每条件额外 forward。没有 cosine diagnostic 前只能 Tier 2。

## Manifold、symmetry 与重训成本

FlowMM、CrystalGRW、DiffCSP++、WyckoffDiff、SymmCD、DynaCrys 已覆盖 manifold、log-lattice、Wyckoff/symmetry-aware generation 的主要叙事。pretrained MatterGen 的 score 是在当前 Euclidean lattice / wrapped coordinate corruption 上训练的；推理时把 update 换到 log-Gram/SPD geodesic 并不自动保持 score 的概率解释。

因此：

- `Log-Gram Cell Predictor` 可作为高风险 sampler ablation，但不能声称首次 manifold crystal generation。
- `Symmetry-subspace projection` 只有在不重训且能证明 posterior distortion 可控时才有增量；当前没有这样的证据。
- 重训 FlowMM/CrystalGRW/CrystalDiT/Equiformer backbone 不满足一个月和 `<5% params / <=2000-step finetune` 优先规则。

## Evaluation 文献对 GO 标准的影响

- [Matbench Discovery](https://www.nature.com/articles/s42256-025-01055-1)（Nature Machine Intelligence 2025）强调 prospective/task-based evaluation、convex-hull stability 和 UIP relaxation，说明单一 surrogate 或 formation-energy MAE 不能覆盖 discovery quality。
- [PhononBench](https://arxiv.org/abs/2512.21227)（arXiv 2025）报告多种 generator 的 dynamical-stability gap，MatterGen 也并非按此指标完美；但 phonon evaluation 成本高，不适合本轮 P0 primary。
- [LeMat-GenBench](https://arxiv.org/abs/2512.04562)（arXiv 2025）报告 stability 与 novelty/diversity 平均存在 trade-off，支持把 NUS/Novelty 设为 guardrail，而不是要求所有指标共同提高。
- 2025 年的 [Generative AI for Crystal Structures review](https://www.nature.com/articles/s41524-025-01881-2) 强调 validity、stability、novelty、效率需要多指标评估，并指出简单 validity 不保证 stability。

本项目据此采用“1 个主指标 + 少数 guardrails”的 P0，而不是把 Mag MAE、E-hull、NUS、MaxF、RMSD、Validity 全设为同时必过。

## 文献审计后的 novelty 边界

| 候选族 | 会不会只是复现 | 可主张的本项目增量 |
|---|---|---|
| KL-Valence-Feasible D3PM | 中等风险：CrysVCD + constrained discrete diffusion 已存在 | MatterGen joint D3PM 内的 variable-site global count feasibility posterior；不是前置 composition model/最终过滤 |
| Feasibility-Preserving P/C | 中高风险：Scout-Matter 已有 volume/coordination guidance | 不设 target、不反传 constraint loss；对官方 joint P/C proposal 做最小 barrier rollback，并量化 posterior/step distortion |
| Multi-condition conflict map | 中等风险：generic gradient projection、多属性生成已存在 | MatterGen 三 field × 多 condition 的 residual conflict 诊断和 field-wise constrained composition |
| Density-aware corruption | 较低复现风险，但重训风险高 | 用 V/N 而不只是 N 校准 pos/cell corruption；必须证明确有因果收益 |
| Log-Gram/symmetry | 高复现风险 | 仅能主张 pretrained-sampler retrofit；不能主张 manifold/symmetry generation 首创 |
| Selective re-noise / SCS | 本地证据已杀死 | 无；不继续 |

