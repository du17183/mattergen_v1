# Cross-field Feature Interaction — P0 最终报告

## 结论

**最终判定：谨慎 GO。建议冻结当前 CFI 配置和 checkpoint，等待批准后进入 32 个全新 paired seeds 的 P1；本轮没有自动启动 P1。**

CFI 在 8 个新 seeds 上相对近似等参数 MLP 同时改善了 E-hull、Stable、NUS、Novel 和 RMSD tail，满足预设“至少两个重要维度额外受益”的 GO 条件。CFI 存在一个 0.393 Å / 495-step 的显著 outlier，但没有复现上一轮 Transformer 的 0.5–1 Å、800+ steps 崩坏；去掉这个 paired seed 后，CFI 的平均 relaxation steps 为 39.4，优于 MLP 在相同剩余 seeds 上的 58.0，说明 CFI 的 relaxation 均值劣势由单个尾部样本主导。

这是 8-seed P0 signal，不是显著性结论。新的 32-seed P1 仍需验证收益是否可复现以及 495-step tail 的真实发生率。

## 1. 分支、环境与实验边界

- Branch：`experiment/cross-field-interaction-p0`
- Parent HEAD：`499dba7df79de8e82539243d4273ee3b66ba81dd`
- Final HEAD 在提交后的交付信息中报告；commit 无法在自身内容中嵌入自己的 hash。
- 项目、虚拟环境、模型、数据、checkpoint、缓存和运行产物全部位于 `/mnt/lis-wam-data/dxl/mattergen_v1`。
- 未使用 CHGNet/MatterSim training reward、force/E-hull loss、novelty reward、composition balancing、Adaptive CFG 或 Q3 refinement。
- 未改变 LR、hidden、gate、插入位置或训练步数；未做 architecture sweep。

## 2. 真实插入位置

复用真实 `GemNetTAdapter -> GemNetTCtrl` 路径。共享 atom hidden 为 `[N_total_atoms,512]`，timestep embedding 为 `[B,512]`，共有四个 interaction blocks。

CFI 在真实 forward loop 的零基 `block_index=1` 后调用，即第二个 interaction block 及其原输出头之后、第三个 interaction block 之前。CFI 只修改 shared scalar hidden `h`；后续两个冻结 GemNet blocks 继续把融合后的表示传播到 atomic、position 和 cell 输出路径。

## 3. 三个字段分支

### Atomic branch

- 输入：中间 hidden `h:[N,512]`。
- 编码：`LayerNorm(512) -> Linear(512,128) -> SiLU`。
- 输出：`z_A:[N,128]`。
- 不使用 atomic logits argmax 或其他不可导硬离散输入。

### Position branch

- hidden 路径：`LayerNorm(512) -> Linear(512,128)`。
- geometry 路径：直接复用 MatterGen 真实 periodic graph 的 `edge_index` 和 `D_st`；距离经过 32 个 Gaussian RBF（0–7 Å）和 `Linear(32,128)`，再按目标原子 `edge_index[1]` 做 mean scatter。
- 两路相加后经过 SiLU，输出 `z_P:[N,128]`。
- 不重新计算 fractional round 距离，不把 Cartesian xyz 或方向向量作为普通 scalar 输入。

### Cell branch

- 每个 crystal 对 `h` 做 mean pool，得到 `[B,512]`。
- 当前行晶格 convention 下使用 `L @ L.T` 的 6 个独立 Gram 分量，加 volume、volume/num_atoms、num_atoms，共 9 个描述符。
- 描述符只宣称对整体 Cartesian rotation 不敏感，不宣称对任意 lattice basis 变换完全不变。
- `[pooled_h,cell_descriptors]:[B,521] -> LayerNorm -> 521→256→128`，输出 `z_C:[B,128]`，再由 `batch` broadcast，绝不跨 crystal 混合。
- H20 上 volume 使用等价、可微的 scalar triple product，避免 batched 3x3 determinant 驱动问题。

### Timestep context

复用原模型 `z:[B,512]`，通过 `LayerNorm(512) -> Linear(512,128) -> SiLU` 得到 `z_T:[B,128]`，仅作为本次单层 interaction context。

## 4. 双向 gated interaction 与残差

本轮只执行一次交互，不使用 Transformer、cross-attention、循环更新或多层 experts。

Atomic 和 Position atom-level context 分别为：

`c_A=[z_A,z_P,z_C[batch],z_T[batch]]`

`c_P=[z_P,z_A,z_C[batch],z_T[batch]]`

Cell context 先对 `z_A,z_P` 按 crystal mean-pool：

`c_C=[z_C,mean(z_A),mean(z_P),z_T]`

每个字段使用相同形式但不共享参数：

`m_F = SiLU(W_message,F c_F + b_message,F)`

`g_F = sigmoid(W_gate,F c_F + b_gate,F)`

`z_tilde,F = z_F + g_F ⊙ m_F`

最后把 atom-level `z_tilde,A`、`z_tilde,P` 和 broadcast 的 `z_tilde,C` 拼接为 384 维，通过 `384→256→512`。最终 `256→512` projection 的 weight/bias 严格 zero-init，得到 `h' = h + Δh`。

## 5. 参数量与公平对照

- CFI：995,986 trainable parameters。
- Matched MLP：1,090,936 trainable parameters。
- CFI 比 MLP 少 94,950，按 MLP 计差异 **8.7035%**，满足预设 ±10% 预算。
- MLP 直接复用上一轮冻结 checkpoint `f21bcd...1ee78`，没有重新训练或挑选新超参数。
- CFI checkpoint SHA256：`5573b71a6bd6a8b57beb36ff27ce309f81bc3f82fb7b4759c3a7aa1b248b5d1b`。

## 6. 数据和训练正确性

- 完全复用上一轮已冻结的 official MatterGen Alex-MP split。
- Source：`alex_mp_20/train.csv`。
- Train：1024；Validation：128；selection seed：`20260907`。
- 不包含历史 generated/formal samples。
- AdamW，LR `1e-4`，weight decay `1e-4`，batch 16，1000 steps，原始 mixed-field diffusion corruption/loss 和原 field weights。

真实 checkpoint zero-init 正式检查通过。CFI-on 相对 baseline 的最大输出差异为：Atomic 0.017879、Position 0.003417、Cell 0.000275；均处于同一次 GPU scatter 重复计算波动容差内。额外 batch-1 smoke 差异为 0.006693、0.000631、0.000133。

36/36 个 CFI 参数张量在真实 backward 中获得 finite gradient。由于最终 projection 为零，step 1 只有 output projection 的 gradient 非零，符合 zero-init residual 预期；训练结束后 36/36 参数张量全部变化。冻结主干没有 gradient/optimizer update，训练前后 digest 相同。

## 7. 训练结果

| Method | Final train total | Final train A/P/C | Final online val total | Final val A/P/C | Time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| MLP（复用） | 0.188760 | 0.105824 / 0.813471 / 0.001588 | 0.290381 | 0.101900 / 1.784611 / 0.010020 | 54.27 s | 1223.85 MiB |
| CFI | 0.189124 | 0.106101 / 0.814657 / 0.001557 | 0.289794 | 0.101744 / 1.780906 / 0.009959 | 58.58 s | 1229.07 MiB |

CFI first/last-50 train total means为 0.277260/0.297589；所有 loss 有限。diffusion timestep sampling 导致 train loss 高方差，因此不把窗口变化解释为单调收敛。

## 8. 完全同噪声 offline validation

128 个 validation structures 使用同一个 corruption seed `20260927`。

| Method | Total | Atomic | Position | Cell |
|---|---:|---:|---:|---:|
| C0 | 0.308686 | 0.103765 | 1.922188 | 0.012702 |
| MLP | 0.309589 | 0.103624 | 1.930229 | 0.012942 |
| CFI | 0.309382 | 0.103677 | 1.928123 | 0.012893 |

CFI 相对 MLP total `-0.000207`、Position `-0.002106`、Cell `-0.000049`，Atomic `+0.000053`。差距很小，仅作辅助，不用于代替生成结论。

## 9. 真实生成与 MatterSim-5M

- 全新 paired seeds：77000–77007。
- C0、MLP、CFI 各 8 个，共 24 次真实生成。
- `dft_mag_density=0.1`、constant original CFG=2.0、original Predictor、original Corrector、1000 diffusion steps、batch size 1。
- 三轮使用 GPU 0–7 并行生成；CFI 轮墙钟约 206.9 秒。
- MatterSim-5M 在 GPU 0–2 上 relaxation，`fmax=0.05`；随后 CPU 仅执行 frozen Alex-MP reference 的 convex-hull/novelty/structure matching。
- 三方法 generation success 均 8/8，MatterSim success 均 8/8。
- 全部是 surrogate evaluation：`DFT_VERIFIED=False`。

## 10. P0 指标

| Metric | C0 | Matched MLP | CFI |
|---|---:|---:|---:|
| Generation success | 1.000 | 1.000 | 1.000 |
| MatterSim success | 1.000 | 1.000 | 1.000 |
| E-hull mean (eV/atom) ↓ | 0.154769 | 0.148692 | **0.082740** |
| E-hull median ↓ | 0.088172 | 0.140134 | **0.054186** |
| E-hull max ↓ | 0.659058 | 0.306365 | **0.217067** |
| Stable ↑ | 0.500 | 0.375 | **0.625** |
| NUS ↑ | 0.250 | 0.125 | **0.500** |
| Novel ↑ | 0.750 | 0.750 | **0.875** |
| Unique ↑ | 1.000 | 1.000 | 1.000 |
| RMSD mean (Å) ↓ | 0.310122 | 0.204315 | **0.081884** |
| RMSD median ↓ | **0.027949** | 0.032888 | 0.048916 |
| RMSD P95 ↓ | 1.221084 | 0.929022 | **0.278790** |
| RMSD max ↓ | 1.563577 | 1.378458 | **0.393339** |
| Atomic force mean (eV/Å) ↓ | 0.252592 | **0.243591** | 0.263773 |
| Structure max-force mean ↓ | 0.502571 | 0.561260 | **0.455691** |
| Atomic force P95 ↓ | **0.474013** | 1.139466 | 0.708645 |
| Force max ↓ | 2.108766 | **1.292300** | 1.310078 |
| Relaxation steps mean ↓ | 159.250 | **56.750** | 96.375 |
| Relaxation steps P95 ↓ | 590.95 | **129.20** | 345.55 |
| Relaxation steps max ↓ | 788 | **167** | 495 |

## 11. Outliers 与鲁棒性解释

CFI 的主要 outlier 是 seed 77003（`ScFe2HO4`）：E-hull 0.05295 eV/atom、RMSD 0.39334 Å、initial max force 0.71342 eV/Å、495 relaxation steps。它显著拉高了 CFI relaxation mean/P95，但没有达到预设的 RMSD≥0.5 Å 或 800+ steps 崩坏。CFI seed 77005 的 force max 为 1.31008 eV/Å，但 RMSD 0.04224 Å、56 steps，未形成同步几何崩坏。

对照方法也有更严重 tails：C0 seed 77002 RMSD 1.56358 Å，C0 seed 77003 RMSD 0.58503 Å/788 steps；MLP seed 77007 RMSD 1.37846 Å。CFI 把本批次最大 RMSD 限制在 0.39334 Å。

排除 paired seed 77003 后，CFI/MLP relaxation steps mean 为 39.43/58.00。分别排除各方法自己的最大 RMSD outlier 后，CFI/MLP trimmed RMSD mean 约 0.03739/0.03658 Å，说明典型样本接近，而 CFI 当前均值优势主要来自更小的极端 RMSD tail。

## 12. 科学比较

### CFI vs C0

正向：E-hull mean/median/max 全面降低；Stable +0.125、NUS +0.250、Novel +0.125；RMSD mean/P95/max 大幅降低；structure max-force mean 和 force max 改善；relaxation mean/P95/max 均改善。

负向：RMSD median 增加约 0.0210 Å；atomic force mean 增加约 0.0112 eV/Å；force P95 增加约 0.2346 eV/Å。因此不是所有局部力指标一致改善。

### MLP vs C0：上一轮信号是否复现

只部分复现。MLP 在本批 seeds 上仅小幅降低 E-hull mean，并降低 RMSD mean/max、atomic force mean、force max 和 relaxation steps；但 E-hull median、Stable、NUS、structure max-force mean 与 force P95 反而更差，Novel/Unique 不变。上一轮 MLP 的强 Stable/NUS signal 没有在新 8 seeds 上复现，说明 plain residual PEFT 的小样本效果本身具有明显 seed 方差。

### CFI vs Matched MLP（核心）

CFI 的额外正向收益：E-hull mean `-0.06595 eV/atom`，median `-0.08595`，max `-0.08930`；Stable `+0.250`，NUS `+0.375`，Novel `+0.125`；RMSD mean `-0.12243 Å`，P95 `-0.65023 Å`，max `-0.98512 Å`；structure max-force mean `-0.10557 eV/Å`，force P95 `-0.43082 eV/Å`。

CFI 的负向：RMSD median `+0.01603 Å`；atomic force mean `+0.02018 eV/Å`；force max `+0.01778 eV/Å`；由于 seed 77003，relaxation mean/P95/max 分别比 MLP 高 39.625/216.35/328 steps。

因此 CFI 并非所有指标获胜，但其优势不能仅由“增加相同参数”解释：相对同一 MLP checkpoint，它在热力学质量、Stable/NUS/Novel 和 RMSD extreme tail 三个核心维度同时改善。当前证据为 Cross-field Interaction 提供了初步机制支持。

## 13. 判定与下一步

1. 训练、gradient、冻结行为：PASS。
2. 24/24 generation 和 24/24 MatterSim：PASS。
3. CFI 相对 C0：有多项真实正向 signal。
4. CFI 相对 MLP：至少 E-hull/Stable-NUS 和 RMSD tail 两个以上重要维度明确更好。
5. 鲁棒性：有一个 495-step tail，需要 P1 验证，但没有触发预设 catastrophic FAIL 阈值。
6. Cross-field mechanism：获得初步而非最终支持。

最终为 **谨慎 GO**，不是正式成功结论。建议进入 32-seed P1，但必须冻结当前 data、architecture、hook、checkpoint、LR、1000-step training 和 sampling/evaluation config，并预先把 relaxation tail 作为关键风险指标。

**唯一下一步建议：经用户批准后，用当前冻结 CFI/MLP checkpoints 在 32 个全新 paired seeds 上执行 P1，重点验证 E-hull/Stable/NUS 优势和 495-step relaxation tail 是否可复现；在此之前不修改 CFI。**
