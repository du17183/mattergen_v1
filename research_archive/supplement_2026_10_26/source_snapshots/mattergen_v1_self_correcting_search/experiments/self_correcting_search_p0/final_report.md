# Hidden-State Probe + Self-Correcting Search P0 最终报告

## 结论

```text
PROBE_P0=FAIL
SELF_CORRECTING_SEARCH_P0=NOT_RUN_PROBE_FAIL
NEXT=STOP
```

- Q1：隐藏状态能否预测最终 CHGNet 磁密度？**NO**
- Q2：概率 verification 能否改善独立 held-out 结果？**NOT RUN**
- 继续条件为 YES/YES；本次决定：`STOP`。

所有 property 数字均为冻结 CHGNet surrogate；`SURROGATE_PROPERTY_EVAL=True`，
`DFT_VERIFIED=False`。CHGNet 校准：MAE `0.001449 Å^-3`、Pearson `0.9499`。

## 固定协议与隔离

- 独立 worktree：`/mnt/datasets-livsyn/dxl/mattergen_v1_self_correcting_search`，branch `experiment/self-correcting-search-p0`。
- 基线：官方 `dft_mag_density` Full FT，checkpoint SHA256
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。
- 目标 `0.2 Å^-3`，constant CFG 2.0，1000 diffusion steps，每 timestep 1 次
  official corrector；Adaptive CFG 关闭。
- Probe cohort：192 条新 trajectory；train/validation/test=`128/32/32`，按 trajectory
  隔离，seeds `98000–98191`。独立搜索 seeds `98200–98207`。
- 数据完整性：192/192 均为 `[5,4,512]`，每条 2000 official score calls；单轨迹
  generation time mean/min/max=`85.870309` /
  `81.509885` / `92.294052` s。
- 未使用 `94000–96031` 或历史实验 seed；未恢复或改写任何 stash。
- 未使用 probe gradient、hidden steering、best-of-3、hard floor/ceiling、LoRA、RL、
  Adapter 或 MatterGen retraining。

## 参考实现审阅

Goodfire 公开了“内部 activation 预测最终属性，再概率接受/拒绝 proposal”的核心思想；
公开文章没有披露精确层、pooling、probe 网络或 transition correction。
`probe-gradient-guidance` 的 `frontier_v2.py` 是 single-proposal 编排器，v3 额外启用
best-of-3 与 hard bounds，`sampler_patch.py` 只提供 K-proposal 分支；MatterGen scorer 本体
并未随仓库发布。本项目因此使用公开 Metropolis 形式
`min(1, exp(-ΔE/0.01))`，但不称严格 MALA。完整证据边界见 `reference_review.md`。

## Probe P0

- BEST_PROBE_BLOCK：`Block 1`
  （GemNet-T `int_blocks` zero-based index `0`）
- architecture：`tiny_mlp`
- 选择依据：validation only；test 不参与模型或窗口选择。报告 schema 修正后的确定性重算
  没有改变 checkpoint、模型选择或 gate。
- 可靠 timesteps：`[]`；`max_t=None`。
- Test scope rows：`160`。

选中 probe 的 validation 总体指标：MAE `0.020586`，
Pearson `0.287864`，Spearman
`0.218211`，target-distance Spearman
`0.342816`，MAE improvement
`-0.858598`。

| 指标 | Test |
|---|---:|
| MAE | 0.021496 |
| RMSE | 0.030851 |
| Pearson | 0.189073 |
| Spearman | 0.173130 |
| R² | -5.275888 |
| target-distance Spearman | 0.324379 |
| train-mean baseline MAE | 0.009187 |
| MAE improvement | -1.339924 |

### Timestep 明细

| split | t | MAE | Pearson | Spearman | distance Spearman | MAE improvement | all pass |
|---|---:|---:|---:|---:|---:|---:|---|
| validation | 0.50 | 0.024204 | 0.209101 | 0.232038 | 0.116569 | -1.185256 | False |
| validation | 0.25 | 0.020080 | 0.151409 | 0.126833 | 0.210411 | -0.812953 | False |
| validation | 0.10 | 0.020626 | 0.243387 | 0.139663 | 0.547287 | -0.862246 | False |
| validation | 0.05 | 0.019752 | 0.388317 | 0.304619 | 0.284457 | -0.783366 | False |
| validation | 0.02 | 0.018266 | 0.420915 | 0.299120 | 0.572581 | -0.649169 | False |
| test | 0.50 | 0.024217 | 0.315174 | 0.207845 | 0.129399 | -1.636028 | False |
| test | 0.25 | 0.030530 | -0.074877 | -0.147727 | 0.125733 | -2.323258 | False |
| test | 0.10 | 0.018449 | 0.158156 | 0.202346 | 0.384897 | -1.008167 | False |
| test | 0.05 | 0.016589 | 0.290049 | 0.275293 | 0.560484 | -0.805726 | False |
| test | 0.02 | 0.017698 | 0.334177 | 0.308651 | 0.518328 | -0.926441 | False |

GO 同时要求 Pearson/Spearman/target-distance Spearman ≥0.50，且 MAE 相对 train-mean
constant 至少改善 25%。若 Probe FAIL，按预注册门控没有运行搜索，也没有扩大 probe。

## 独立 held-out self-correcting search

| 指标 | B0 | S0 |
|---|---:|---:|
| n | N/A | N/A |
| CHGNet Mag MAE | N/A | N/A |
| CHGNet median error | N/A | N/A |
| hits (`tau=0.01`) | N/A | N/A |
| valid fraction | N/A | N/A |

- MAE reduction：N/A
- paired better / same / worse：N/A /
  N/A / N/A
- extra hits：N/A
- PROBE_EXPLOITATION/EVALUATOR_MISMATCH：
  `False`
- MatterSim authorized/run：`False` /
  `False`。

### Search trace 与运行代价

| 指标 | B0 | S0 |
|---|---:|---:|
| mean runtime (s) | N/A | N/A |
| MatterGen score calls / sample | N/A | N/A |
| extra conditional forwards / sample | N/A | N/A |
| probe verification time / sample (s) | N/A | N/A |
| max peak VRAM (bytes) | N/A | N/A |

- S0 slowdown vs B0：N/A×。
- scored / accepted / rejected proposals：N/A /
  N/A / N/A。
- acceptance / rejection rate：N/A /
  N/A。

若 property gate 通过，冻结 MatterSim-5M 结果为：B0/S0 E-hull
`N/A` /
`N/A`，stable fraction
`N/A` /
`N/A`。若 gate 未通过，这些字段保持
`N/A`，MatterSim 未运行。

预实现的搜索器采用 ordinary corrector 的一份 proposal。若 Probe gate 通过，更接近目标者
必收，变差者仍按非零概率接受；没有 gradient、greedy selection、K 候选或硬阈值。本轮
Probe FAIL，搜索器未运行，因此 `search_trace.csv` 与 `efficiency_metrics.csv` 仅保留表头。

## 公开来源

- Goodfire：<https://www.goodfire.com/research/self-correcting-search>
- reproduction：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance>
- MatterGen：<https://github.com/microsoft/mattergen>
