# MatterGen 架构与可干预缺口审计

## 审计结论

MatterGen 已经在一个 GemNet-T 中联合读取原子类型、分数坐标和晶格，并共享 node/edge hidden state；因此“模型完全没有跨字段耦合”是不成立的。真正相对独立的是三种 corruption、三个输出头和 sampler 中同一时刻的字段更新。当前最明确的空缺是：atomic D3PM 的逐 site categorical posterior 只受元素集合 mask，不受全局化学可行性约束；position/cell Predictor–Corrector 也只做几何表示合法化，没有最小距离或体积/原子可行域保持。

## A. Atomic diffusion

### 当前实现

- `D3PMCorruption` 是离散时间、离散类别 corruption；其 `N` 必须和 sampling noise levels 一致（`mattergen/diffusion/corruption/d3pm_corruption.py:15-34`）。
- 默认 atomic predictor 是 `D3PMAncestralSamplingPredictor(predict_x0=True)`（`sampling_conf/default.yaml:45-55`）。
- denoiser 输出 `p(x0|xt)` logits；代码先 softmax 得到 per-site `class_probs`，再用 D3PM 的 `q(x_{t-1}|x_t,x_0)` 公式得到 reverse logits，并对每个 site 做 categorical sample（`mattergen/diffusion/d3pm/d3pm_predictors_correctors.py:67-103`）。
- `mask_disallowed_elements` 会屏蔽全局禁用元素；若条件中有 `chemical_system`，还会屏蔽系统之外的元素（`mattergen/denoiser.py:49-125`）。

### 已有约束与缺口

已有的是**允许元素集合约束**，不是**组成可行性约束**。不同 site 的最终联合组成没有在 reverse transition 中检查 oxidation-state balance、charge neutrality 或 SMACT feasibility。Formal256 的 54/256 chemistry-invalid-but-geometry-valid 正好落在此缺口。

最自然的干预点不是最终 rejection，也不是改 backbone，而是 `D3PMAncestralSamplingPredictor.update_given_score()` 中 `class_probs` 形成之后、调用 `sample_and_compute_posterior_q()` 之前：把 factorized `pθ(x0|xt)` 做一个受限、KL 最小的全局组成后验投影，再保持官方 D3PM posterior 公式不变。

## B. Position diffusion

### 当前实现

- position 使用 `NumAtomsVarianceAdjustedWrappedVESDE`；分数坐标在单位周期上 wrapped，噪声标准差乘 `N^{-1/3}`，目的是降低 cell size 对 fractional-coordinate variance 的影响（`mattergen/common/diffusion/corruption.py:215-270`）。
- GemNet 在当前 lattice 下把 fractional coordinates 转为 Cartesian，在线构图并计算距离/方向；position score 最终来自累积 edge force-like head（`mattergen/common/gemnet/gemnet_ctrl.py:122-180, 196-235, 288-299`）。
- denoiser 返回前，再用 inverse-transpose lattice 把 Cartesian position output 映射回 fractional score（`mattergen/denoiser.py:128-167`）。

### 缺口

- wrapped representation 保证周期等价，但 Predictor/Corrector update 不保证更新后 `d_min` 不跌破一个合理的、随时间收紧的下界。
- `N^{-1/3}` 只按原子数缩放，不显式使用当前 `V/N`。现有 64 样本中 `V/N` 与 final MaxF、RMSD 在 calibration/validation 都稳定相关，说明这是结构归一化风险因子；但相关不等于更改 forward corruption 会有效。
- 默认 geometry validity 的 0.5 Å cutoff 很宽；64 轨迹和 E3 Formal256 都可达到 100% structure validity，同时 MaxF/RMSD 仍有长尾。因此 P0 不应只优化二值 structure validity。

## C. Cell diffusion

### 当前实现

- cell 使用 `LatticeVPSDE`。limit mean 是只由原子数和固定 `limit_density=0.05 atoms/Å³` 得到的各向同性晶格；limit variance 随 `N^{2/3}` 缩放（`mattergen/common/diffusion/corruption.py:41-154`）。
- forward marginal 对 lattice matrix 元素加入对称 Gaussian noise（同文件 `156-182`）。
- lattice Predictor/Corrector 在更新后调用 polar-decomposition 规范化，避免任意旋转 gauge；这保持的是表示/规范，不是物理密度、最短距离或给定对称性约束。
- GemNet 的 lattice output 是由各 interaction block 的 edge embedding、distance vector 和 lattice 汇聚的 stress-like head（`mattergen/common/gemnet/gemnet_ctrl.py:171-181, 226-235, 301-304`）。

### 缺口

- Euclidean lattice update + polar gauge 没有在 SPD metric/Gram manifold 上显式限制 geodesic step。
- 固定先验密度只用于高噪声 limit，不保证 reverse trajectory 的 `V/N` 落在化学/规模合理区间。
- 64 轨迹中 late cell residual 对 final 指标没有跨 split 稳健复现，因此不能据此做 cell residual gate；若研究 cell，应直接针对表示/可行域，而非 residual 异常检测。

## D. 三字段 coupling

### 在哪里真正共享

1. `GemNetTDenoiser.forward()` 把 time embedding 与 property embeddings 拼成每晶体 global `z`，再把当前 atomic numbers、fractional coordinates、lattice 和 num_atoms 一起送入同一个 GemNet（`mattergen/denoiser.py:228-260`）。
2. GemNet 用 atomic embedding 与 `z` 合成 node hidden `h`，用当前 cell/positions 构造 periodic edge `m`；四个 interaction blocks 共享并更新这些表示（`mattergen/common/gemnet/gemnet_ctrl.py:122-168, 196-235`）。
3. position 和 cell heads 在各 block 累积，atomic logits 从最后的 node embeddings 经 `fc_atom` 输出（`mattergen/denoiser.py:208, 262-267`）。所以 atomic、position、cell 并非三个互不交流的网络。

### 在哪里相对独立

- corruption families 分开：categorical D3PM、wrapped VE-SDE、lattice VP-SDE。
- output parameterization 分开：atomic logits、force-like position score、stress-like cell score。
- 给定一次共享 score evaluation，三个 predictors 是一个 `apply()` 调用中的并列字段更新；它们看见的是同一个更新前状态，而不是 atomic 更新后立刻重算 position score。
- 默认 corrector 只有 pos 和 cell，没有 atomic corrector（`sampling_conf/default.yaml:57-69`）。

### 是否需要 sequential update

64 轨迹的 raw cross-lag 很高，但 timestep-residualized correlation 在 validation 为 −0.188 到 +0.059，没有稳定的有方向依赖。因此当前没有证据把 simultaneous predictor 改成 sequential predictor。一个额外 score call 所产生的成本也会直接触碰 Corrector 加速家族已经暴露的速度–尾部质量权衡。

## E. Conditioning

- base property embeddings 与 time encoding 在进入 GemNet 前拼入 global `z`（`mattergen/denoiser.py:237-247`）。
- conditional adapter 路径在每个 GemNet interaction block 前计算 property-specific MLP/mixin，mixin 零初始化（`mattergen/common/gemnet/gemnet_ctrl.py:23-67, 188-205`）。
- CFG sampler 分别计算 conditional/unconditional output 后组合，但默认是对已组合条件的整体 guidance；它不显式拆出每个 property residual，也不诊断条件间 cosine conflict。
- Innovation 1 已经对多字段 CFG 强度/阶段作在线自适应。因此新 multi-constraint 方法若仍只是“另一个动态 guidance scale”，论文边界不清；只有**按条件分解、解决条件间冲突**才是不同问题，但会增加多次 forward。

## F. Sampling 的确切顺序

默认 `N=1000`，每个 time step：

1. 对当前 joint batch 做一次 exact score evaluation。
2. position 与 cell corrector 使用同一个 score 并列更新；atomic 没有 corrector。
3. 对 corrector 后的 joint batch 再做一次 exact score evaluation。
4. position、cell、atomic predictors 使用这一个 score 并列更新。

实现见 `mattergen/diffusion/sampling/pc_sampler.py:277-344`。正常配置因此约 2,000 次 score calls。历史 residual-distillation 路径能替换第 3 步的一部分 exact call，但 Formal256 的 force-tail 非劣性未完成，所以本轮不把它重新列为新候选。

## 架构 gap map

| Gap | 代码证据 | 历史数据证据 | 是否适合一月 P0 |
|---|---|---|---|
| G1：atomic posterior 无全局价态/电中性约束 | D3PM per-site categorical；只有 allowed-element mask | Formal256 54/256 chemistry-invalid、geometry-valid | 是；Top1 |
| G2：P/C update 不保持 d_min 与 V/N 可行域 | polar/wrap 只保证表示合法 | d_min 对 MaxF/E-hull/RMSD 两 split 稳定；V/N 对 MaxF/RMSD 稳定 | 是；Top2，但可能与 E3 target 重叠 |
| G3：多条件 guidance 不显式处理条件冲突 | condition 先合成，CFG 是整体 residual | 尚无 per-condition cosine 数据 | 先 diagnostic；与 Innov1 边界风险高 |
| G4：cell reverse update 非 intrinsic metric-space step | Euclidean matrix SDE + polar gauge | late cell residual 未跨 split复现 | 理论成立，当前高风险 |
| G5：noise scaling 不含当前 density | pos 仅 N scaling；cell prior 是固定 density | V/N 与质量相关，但只有 observational evidence | 需重训/至少重标定，当前高成本 |
| G6：在线 tail observability 不足 | 可记录 residual/entropy | residual/entropy 对 bad tail 不跨 split | 否；杀死 re-noise/tail gate |

## 哪些看似 gap、实际不是

- **“缺乏原子–几何 coupling”不是 gap。** 一个共享 GemNet 已联合建图和消息传递；Cell–Atom Fusion 的失败也说明再塞 hidden fusion 没有证据。
- **“字段收敛时间不同”不是 gap。** field/site diagnostics 均未过预注册 gate。
- **“模型不懂 periodic geometry”不是 gap。** wrapped coordinates、periodic graph、lattice gauge 已存在；可研究的是更窄的 physical feasibility 或 metric-space update。
- **“backbone 不够强”不是当前可行动 gap。** CrystalDiT/EquiformerV3 可能更强，但替换 backbone 要重训，且无法在一月内形成公平、独立的 Innovation 2。

## 审计后的方法空间

剩余方法空间被压缩为三类：

1. 对 discrete atomic posterior 做全局可行、最小分布偏移的约束投影。
2. 对 continuous P/C proposal 做局部、可回退、模型无关的几何可行域保持。
3. 只在通过独立 diagnostic 后，研究条件间 conflict 或 density-aware corruption；这两类当前都没有资格直接实现。

