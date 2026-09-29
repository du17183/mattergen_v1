# RC-NFGD 与 Fixed-K2 采样状态审计

结论：RC-NFGD Formal256 保存**终态与末 20 步代理力观测**，不保存可恢复的扩散检查点。Fixed-K2 保存一个用于同运行分支搜索的前缀 batch/RNG，但也不是跨环境自包含快照。

| 必要状态 | RC-NFGD Formal256 历史制品 | Fixed-K2 C1 前缀制品 |
|---|---|---|
| `x_t`：原子种类、分数坐标、晶胞 | 无；仅最终 extxyz/CIF | `batch.clone()` 含三字段及 batch 元信息，step 400 |
| timestep/当前步 | F0 trace 仅记录 980–999 步 `t_norm`；C0 无 | `branch_point=400`；数值 `t` 由代码重建 |
| scheduler 状态与 `dt` | 无序列化；`pc_sampler.py` 用 `torch.linspace(max_t,eps_t,N)` 与公式重建 | 无序列化，依赖冻结代码/配置 |
| noise schedule / corruption 配置 | 从模型 `config.yaml` 和采样器配置加载；制品未记录完整哈希 | 依赖代码/配置；prefix artifact 未独立保存完整配置哈希 |
| condition tensor（目标、原子数等） | summary 有目标 0.2 和最终原子数；没有原始 condition tensor | batch 理论上保留条件；目标未单独哈希 |
| atom-type / position / cell prior | 无 | step 400 batch 已包含其演化结果，不单列初始 prior |
| RNG | 只记录整数 seed | 当前设备的 Python/NumPy/Torch CPU/CUDA 状态快照 |

RC-NFGD 代码路径是 `condition_factory.get_number_of_atoms_condition_loader()` → `pc_sampler._sample_prior()` → `pc_sampler._denoise()`。condition loader 先用 NumPy 抽原子数；prior 按排序后的 corruption 字段抽原子种类、位置和晶胞；每个 predictor/corrector step 会继续计算 score 并抽噪。`late_force_sampler.py` 仅在 predictor 且 `t≤0.02` 时修改**位置 score**，并把力、接受/回退写入 CSV；它没有在采样早期或结束时序列化完整 `x_t`/RNG。

已有可比锚点：历史与重建均为 1000 步、CFG=2、每样本 2000 MatterGen score calls；F0 均在 step 980–999 触发 20 次且 20 次接受。历史 step 980 的 `t_norm=0.020000001415610313` 与重建相同，但首次力评价能量分别为 -45.071213 与 -56.238422 eV、MaxF 分别为 0.053565 与 0.005932 eV/Å。此行记录发生于本次力修正之前，说明当时输入 clean estimate 已不同；不能把不一致归咎于之后的力修正或 extxyz 格式化。C0 本身的终态也不同。

缺少历史 step 0 的 condition/prior、step 1 的 score 与 step 1–979 的 `x_t`，所以**无法指出首次分叉的精确步号**。Fixed-K2 的状态保存机制不能填补这些 RC-NFGD 历史缺口。

## 本轮最小化状态对照

在相同历史源码入口和 seed 740000 下，重建环境与现存 ALM 环境的 condition、prior `x_T` 和各 RNG 摘要完全一致；加载后的 304 个模型 state tensor 汇总 SHA256 均为 `2d505e7fc6d1ae3102e4d8881dc493852f8690b4db7dafaf86901250d2d99520`。首次 denoiser score 的原子种类、位置、晶胞三个字段哈希均不同，且 forward 前后 RNG 摘要仍相同。这说明**两现存运行时**从相同初始状态和相同权重进入首次 score 时已出现数值差异；历史 2026-09-13 的首次 score 未保存，不能反推它与哪一边相同，也不能声称历史恰在 step 0 分叉。固定噪声单步 scheduler 对比应在首次 score 差异解释清楚后再做，否则混合了上游误差。
