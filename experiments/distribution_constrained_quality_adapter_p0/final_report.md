# MatterGen Distribution-Constrained Quality Adapter P0 最终报告

## 一句话结论

基于官方 Alex-MP clean-x0、真实 MatterGen mixed-field diffusion loss、冻结 CHGNet 独立质量标签和真实 MatterSim-5M 评价，静态 M2 Quality Adapter 能稳定训练并在 8 个全新配对 seeds 上给出足够明确的正向 signal，**建议冻结当前 M2 配置并进入 P1 32-seed 验证**。这是 P0 小样本结论，不是显著性或正式效果声明。

## 1–16：分支、数据、模块与训练

1. **Branch / HEAD**：分支 `experiment/distribution-constrained-quality-adapter-p0`；P0 实现与真实结果提交 `b0c22b5`；开始基线 HEAD `f983e1285de0913d5bb2c2cdef4b5488057f58ad`。文档提交后的最终 HEAD 以 `git rev-parse HEAD` 为准，避免在提交内容中自引用。

2. **clean-x0 来源**：官方下载的 MatterGen Alex-MP 数据包 `.cache/datasets/alex_mp_20.zip` 中 `alex_mp_20/train.csv`。zip 为 186,506,244 字节，SHA256 `6688e652e45fac4ac8b016dbc4a3280afde9ba687bdacfd8c676775ed19b6228`。

3. **数据量**：597,339 条满足 `dft_mag_density` 有限且 1–20 atoms 的候选；固定 seed `20260906` reservoir sampling 128 条，96 train / 32 validation，split seed `20260907`。

4. **是否原训练分布**：是。数据直接来自官方 Alex-MP train split；没有使用历史 formal/test/generated output，没有 self-training pool。

5. **CHGNet quality signal**：冻结 CHGNet 0.3.0 在 GPU 上为 128/128 结构成功打标。force-max（eV/Å）：min `1.82e-7`、P10 `0.00239`、P30 `0.07550`、median `0.12979`、mean `0.18823`、P70 `0.20597`、P90 `0.43693`、P95 `0.57233`、max `1.47247`。训练集分段阈值为 P30 `0.07690`、P70 `0.19853`。

6. **force-max 是否能区分样本**：能。P90/P10=`183.0`，分布非恒定；96 个训练结构来自 96 个不同 chemical systems，低/高 force 组各含 29 个不同 systems，单一 system 最大占比仅 `3.45%`，不是组成集中造成的假信号。

7. **Adapter 插入位置**：GemNet interaction block index 1、2 的原输出 heads 之后、下一 interaction block 之前。C0 不挂载该属性，保持原 checkpoint 路径。P0 使用 static Adapter，stage gate 未启用。

8. **参数量**：每个 bottleneck 为 `LayerNorm(512) → Linear(512,64) → SiLU → Linear(64,512)`，两个 block 共 `134,272` 个可训练参数。挂载后的总模型参数 `48,894,715`，Adapter 占约 `0.275%`；checkpoint 各约 530 KiB。

9. **trainable 列表**：正确，共 12 个张量：两个 block 各自的 `norm.weight/bias`、`down.weight/bias`、`up.weight/bias`。所有 MatterGen 原参数 `requires_grad=False`；第一步 12/12 Adapter 参数得到梯度，冻结参数梯度数为 0。

10. **zero-init baseline 恢复**：通过。四个最终 `up.weight/up.bias` 最大绝对值严格为 0。GPU scatter 本身非 bitwise deterministic，因此同时记录 disabled-vs-disabled repeat：atomic/pos/cell 最大抖动约 `0.0116–0.0170 / 0.00081–0.00122 / 0.00016–0.00020`；enabled-zero 与基线差异处于同一数量级并通过保守绝对阈值。该检查验证解析零残差，而不是错误要求 GPU 两次 forward bitwise 相等。

11. **M1 是否稳定**：是。200 steps 全部 finite，无 NaN/Inf；冻结主干训练前后 SHA256 均为 `01e439133f0c7ca98e0bcd7ca9217bca7978aa0b588806bacf6752f06580ac8a`。前/后 20-step train loss 均值 `0.28003 / 0.28215`；记录的最低 validation loss `0.23263`。训练后 up-projection L2=`1.47737`。

12. **M2 是否稳定**：是。200 steps 全部 finite，无 NaN/Inf；冻结主干哈希同样不变。前/后 20-step train loss 均值 `0.27612 / 0.28447`；记录的最低 validation loss `0.23290`。训练后 up-projection L2=`1.48016`。小数据与随机 diffusion timestep 导致曲线波动，不将“非单调下降”误判为失败。

13. **Replay**：已实现。M2 每 batch 用 `randperm` 随机选精确 50% 样本将质量权重恢复为 1；batch=16 时 replay=8。M1 的所有权重原本就是 1，因此 replay 操作对 M1 数值无影响。

14. **Output anchor**：已实现。teacher 与 student 使用完全相同 noisy state；teacher 在 Adapter disabled 且 `no_grad` 下计算。anchor=`atomic KL + 0.1 × normalized position MSE + normalized cell MSE`，总 loss 加权 `lambda_A=0.05`。

15. **训练步数与配置**：M1/M2 均为真实 pretrained checkpoint、200 steps、batch 16、AdamW、lr `3e-4`、weight decay `1e-4`、同 seed `20260906`；单组训练约 17 秒，峰值 CUDA 显存约 1,200 MiB。

16. **Loss 曲线**：validation total loss（step 1/20/40/60/80/100/120/140/160/180/200）：

   - M1：`0.3187, 0.3177, 0.2517, 0.2805, 0.2601, 0.4060, 0.3130, 0.2326, 0.3929, 0.3001, 0.3411`
   - M2：`0.3205, 0.3029, 0.2453, 0.2859, 0.2641, 0.4024, 0.3142, 0.2329, 0.3707, 0.2945, 0.3146`

   完整逐 step 曲线保存在忽略的大型运行目录 `checkpoints/M1|M2/training_curve.csv`，压缩汇总见 `training_summary.csv`。

## 17–25：8-seed 真实生成与 MatterSim 效果

17. **动态生成协议**：C0/M1/M2 各 8 个全新 seeds `78000–78007`，共 24 个结构；`dft_mag_density=0.1`、1000 diffusion steps、原 predictor/corrector、每 timestep 1 次 corrector、constant CFG=`2.0`、batch=1。三组 generation success 均为 8/8，MatterSim relaxation success 均为 8/8。8 张 H20 每轮并发；平均纯采样耗时 C0/M1/M2=`116.63/117.17/118.57 s`，没有明显 Adapter 吞吐惩罚。

18–25. **正式指标**：E-hull、Stable、NUS、Novel、Unique、RMSD 来自项目正式 MatterGen evaluator；能量/力/松弛来自 MatterSim-5M，`fmax=0.05 eV/Å`。

| 指标 | C0 | M1 | M2 | M2 − C0 |
|---|---:|---:|---:|---:|
| E-hull (eV/atom, ↓) | 0.16073 | 0.13328 | **0.08495** | **-0.07578 (-47.1%)** |
| Stable | 3/8 (37.5%) | 3/8 (37.5%) | **6/8 (75.0%)** | **+37.5 pp** |
| NUS | 3/8 (37.5%) | 2/8 (25.0%) | **5/8 (62.5%)** | **+25.0 pp** |
| Novel | 7/8 (87.5%) | 7/8 (87.5%) | 6/8 (75.0%) | -12.5 pp |
| Unique | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 0 pp |
| RMSD from relaxation (↓) | 0.16798 | **0.04186** | 0.05973 | -0.10825 (-64.4%) |
| 初始全原子 force norm mean (eV/Å, ↓) | 0.22297 | 0.17137 | **0.14615** | **-34.5%** |
| 初始全原子 force norm P95 (eV/Å, 描述) | 0.75922 | **0.59608** | 0.69005 | -9.1% |
| 初始 max-force（全部原子最大值，eV/Å, ↓） | 1.33473 | **0.82768** | 1.08824 | -18.5% |
| 每结构 max-force 的均值 (eV/Å, ↓) | 0.38738 | 0.31395 | **0.25886** | **-33.2%** |
| 每结构 max-force P95 (eV/Å, 描述) | 1.13090 | **0.80000** | 0.84309 | -25.4% |
| 松弛 steps mean / max (↓) | 98.25 / 601 | **48.75 / 159** | 50.88 / 208 | -48.2% / -65.4% |

M2 对 C0 的逐 seed 描述：E-hull 在 7/8 seeds 更低，RMSD 在 6/8 更低；Stable 有 3 个净改善、0 个变差。固定 seed 的 paired bootstrap 仅作不确定性描述：E-hull 平均差 95% interval `[-0.1656, -0.0005] eV/atom`；RMSD、force interval 仍跨 0。M2 对 M1：E-hull 再降 `0.04832 eV/atom`（6/8 更低），Stable 与 NUS 均增加 3/8；Novel 少 1/8。n=8 不支持强统计结论。

## 26–29：科学判断

26. **是否看到真实正向 signal**：是，而且主要来自 M2 而非仅仅“加一个 Adapter”。相对 C0，M1 改善 E-hull/RMSD/force，但 Stable 不变且 NUS 下降；M2 同时在 P0 描述值上改善 E-hull、Stable、NUS、RMSD、force 与松弛步数。相对 M1，M2 的 E-hull、Stable、NUS、force 均向好，满足“Quality Adapter 相对 Unweighted Adapter 至少一点正向 signal”的晋级条件。

27. **是否值得进入 P1 32 seeds**：**值得，GO**。六条 P0 条件全部满足：真实训练稳定、Adapter 确实更新、zero-init 可恢复、24/24 真实生成成功、24/24 MatterSim 成功、没有质量灾难且 M2 对 M1 有正向 signal。P1 应冻结当前 M2 架构、权重分段、replay 与 anchor，不在扩大样本前继续调参。

28. **当前最大问题**：样本仅 8/方法，比例指标一次样本就是 12.5 pp；C0 RMSD 均值被 seed 78007 的 `1.166` 明显拉高，因而 RMSD 相对提升不够稳健。同时 M2 Novel 从 7/8 降到 6/8，需在更大独立样本确认这是随机波动还是质量偏置的代价。

29. **下一步只建议一个最值得做的修改**：不改模型与超参，只把冻结 M2 与 C0/M1 的独立验证规模从 8 扩到 **P1 32 seeds**，用同一正式 MatterSim pipeline 检查 E-hull/Stable/NUS 改善能否复现，并重点观察 Novel trade-off。除此之外本轮不建议加入 stage gate、A0、RL、physics/symmetry loss 或继续小样本调权重。

## 最终回答

> 基于真实 clean-x0 数据、真实 MatterGen mixed-field diffusion loss 和独立 CHGNet 质量标签，小型 Distribution-Constrained Quality Adapter 能够稳定训练；M2 在真实生成中出现了超过 P0 晋级要求的正向 signal，值得进入冻结配置的 32-seed P1。当前证据仍是 n=8 的筛选性证据，不能替代 P1 独立验证。
