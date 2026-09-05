# MatterGen 第二创新 V2 最终报告

## 结论

`DAgger/on-policy residual distillation + atomic-aware fallback + last-30% exact`
保住了加速，并把 V1 曾出现的约 2× RMSD 退化压回到 C0 的 1.19×；严格单 H20
配对 speedup 为 **1.362×**，95% CI 为 **[1.321, 1.406]**，forward reduction
为 **30.71%**。不过独立 64-seed 上 V2 的 E-hull、Stable、NUS、RMSD 与 force
点估计均没有全面优于 C0。各配对差值的 95% CI 都跨 0，因此没有“明确统计恶化”，
也不能声称等价。

最终判断是：**formal256 最低准入条件边界通过**。可以且只应使用当前冻结配置做
一次不调参的确认性 256-seed 实验；若 E-hull 或 force tail 在更大样本上确认恶化，
应按停止条件终止继续堆叠 Adapter 复杂度，转向其他第二创新方向。

质量指标全部来自 **MatterSim-5M surrogate，不是 DFT**。

## 1. V2 branch / HEAD

- 分支：`experiment/corrector-residual-distillation-v2`
- V1 基点：`d2736e01bf7ef9f1072e2760622c603403516403`
- 冻结候选提交：`b577f8c7324625ab295eb07aca9be8082435e5d4`
- 冻结产物清单提交：`7292d3a`
- 64-seed 独立评估提交：`34ff467`
- 最终文档提交之后的分支 HEAD 以 `git rev-parse HEAD` 为准；报告中的数值全部来自
  `34ff467` 已提交的小型结果文件。

## 2. 是否成功恢复并保护 V1

成功。V2 从指定 V1 HEAD `d2736e0` 创建；对
`experiments/corrector_residual_distillation_v1/` 执行基点差异检查为空。V1 的
63000–63031 未进入 V2 train/validation/final 调参流程，V1 正式产物没有被删除、
覆盖或修改。

从效果看，V1 曾有的约 2× RMSD 退化在 V2 独立测试中没有重现，但 V2 的几何与
force 点估计仍未完全追平当前 C0。

## 3. DAgger 新增多少 rollout records

- train seeds 64000–64015：16,000 条真正 on-policy rollout records。
- held-out validation seeds 65000–65015：16,000 条，只用于验证/校准，不进训练。
- DAgger manifest 合计：32 runs、32,000 records、442,026,624 bytes。
- V2 Adapter 的训练集合：V1 frozen teacher train 16,000 条（61000–61015）加
  DAgger train 16,000 条，共 32,000 条；ground truth 始终是冻结原始 MatterGen
  在真实 rollout `x_after` 上的 exact `score_after`，不是 Adapter 自身输出。
- train/validation/final 与 V1 frozen test 的交集均为空。

## 4. V1 vs V2 residual prediction

以下是在独立 exact validation trajectory（65000–65015）上的总体结果。MSE 越低、
cosine 越高越好。

| Field | V1 MSE | V2 DAgger MSE | V1 cosine | V2 cosine |
|---|---:|---:|---:|---:|
| pos | 2166.3049 | 2125.9214 | 0.38735 | 0.40759 |
| cell | 5.92468 | 5.70720 | 0.29279 | 0.34368 |
| atomic | 3.84357 | 3.73117 | 0.28438 | 0.32898 |

V2 三个 field 的总体 MSE 均小幅下降、cosine 均提高，说明 DAgger 改善了 exact
one-step residual prediction；改善幅度有限，并不能单独保证 closed-loop 质量。

## 5. early/middle/late residual 对比

阶段定义为 early `progress < 1/3`、middle `1/3 ≤ progress < 2/3`、late
`progress ≥ 2/3`。

| Stage | Field | V1 MSE / cosine | V2 MSE / cosine | 判断 |
|---|---|---:|---:|---|
| early | pos | 0.03386 / -0.00633 | 0.90043 / 0.00943 | MSE 明显变差 |
| early | cell | 0.01573 / 0.95879 | 0.01505 / 0.95881 | 基本持平 |
| early | atomic | 5.91775 / 0.28442 | 5.80880 / 0.31351 | 改善 |
| middle | pos | 101.2202 / 0.43667 | 94.7404 / 0.48734 | 改善 |
| middle | cell | 0.04799 / 0.88192 | 0.04722 / 0.88051 | MSE 微降、cosine 微降 |
| middle | atomic | 4.81557 / 0.29953 | 4.60222 / 0.36107 | 改善 |
| late | pos | 6384.9920 / 0.38672 | 6269.6798 / 0.40635 | 改善 |
| late | cell | 17.6750 / 0.26553 | 17.0253 / 0.32169 | 改善 |
| late | atomic | 0.80650 / 0.16009 | 0.79132 / 0.21293 | 改善 |

训练损失使用固定权重：early 0.5、middle 1.0、late 2.0。late 的三个 field 均改善；
early position MSE 反而大幅上升，是最终方法仍存在的关键风险之一。

## 6. pos/cell/atomic risk correlation

风险目标为各 field normalized residual RMSE 的 `log1p`，相关性在 held-out
validation 上计算。

| Field | Overall | Early | Middle | Late |
|---|---:|---:|---:|---:|
| pos | 0.95791 | 0.60849 | 0.82687 | 0.90672 |
| cell | 0.79867 | 0.25263 | 0.24377 | 0.78794 |
| atomic | 0.81163 | 0.53923 | 0.46786 | 0.70198 |

全局风险与三字段平均误差的相关性为 0.78061。cell 在 early/middle 的相关性偏弱；
最终冻结规则只使用 atomic risk，是 Stage-B 质量优先选择的结果，并不表示 pos/cell
误差不存在。

## 7. 最终 timestep schedule

Sampler 从高噪声到低噪声运行：`progress = sampling_step / (num_steps - 1)`，
`progress=0` 是 `t≈T` 的高噪声起点，`progress=1` 是 `t≈eps` 的最终阶段。

最终固定 1000 steps：

- `0 ≤ progress < 0.7`：700 个 eligible steps，运行 Adapter 后由 atomic risk
  决定接受或 exact fallback。
- `0.7 ≤ progress ≤ 1`：最后 300 steps 无条件调用 exact MatterGen second score。
- 无 early reuse。

Stage-B 同时实测了 last20、last30、last40 和 early-reuse ablation。

## 8. 最终 fallback rule

固定规则为：

```text
if progress >= 0.7:
    exact
elif adapter output is non-finite:
    exact
elif risk calibration/threshold missing or invalid:
    exact
elif atomic risk is non-finite:
    exact
elif risk_atomic_numbers > 0.8289545722625317:
    exact
else:
    accept Adapter score_after
```

一次 exact forward 同时返回 pos/cell/atomic 三字段，没有把字段级 decision 冒充成
字段级主干节省。所有异常路径 fail closed 到完整 MatterGen second forward。

## 9. 最终 Adapter 参数量

2,661 parameters，selected epoch 18，weighted validation objective
1.3172586676。冻结 checkpoint SHA256：
`a2fd0f17c27b760958b60c1d1386577883635a34fb2101eb08e38fd8472702c7`。
模型仍在 2k–20k 的限制内，没有扩成大型网络。

## 10. Stage-B 所有候选结果

16 个 validation seeds 65000–65015；force mean 是逐结构初始 max-force 的均值，
与 V1 口径一致。

| Candidate | Speedup | Coverage | Fwd reduction | E-hull | Stable | NUS | RMSD Å | Force mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 C0 | 1.000 | 0% | 0% | 0.12366 | 50.00% | 18.75% | 0.08891 | 0.4940 / 3.3665 |
| B1 V1 Global50 | 1.237 | 48.92% | 24.46% | 0.10288 | 50.00% | 18.75% | 0.03975 | 0.3180 / 0.9039 |
| B2 V2 Global50 | 1.266 | 51.51% | 25.76% | 0.16015 | 43.75% | 31.25% | 0.06554 | 0.3009 / 1.4175 |
| B3 V2 Global75 Late20 | 1.444 | 71.81% | 35.90% | 0.13211 | 50.00% | 31.25% | 0.06437 | 0.3021 / 1.0827 |
| B3 V2 Global75 Late30 | 1.384 | 64.34% | 32.17% | 0.13922 | 43.75% | 25.00% | 0.07236 | 0.3266 / 1.0827 |
| B3 V2 Global75 Late40 | 1.321 | 55.69% | 27.85% | 0.13973 | 62.50% | 31.25% | 0.06415 | 0.3049 / 1.3455 |
| B4 V2 all-field75 Late30 | 1.339 | 58.42% | 29.21% | 0.12610 | 56.25% | 37.50% | 0.05656 | 0.5001 / 2.8010 |
| B4a V2 atomic75 Late30 | 1.342 | 58.54% | 29.27% | 0.12669 | 56.25% | 37.50% | 0.05443 | 0.4170 / 1.6167 |
| B5 early-reuse20 all-field75 Late30 | 1.371 | 61.41% | 30.70% | 0.11729 | 43.75% | 25.00% | 0.05872 | 0.3717 / 0.9797 |

九个候选共 144/144 generation 和 relaxation 成功，没有几十组超参搜索。

## 11. 为什么选择 frozen candidate

按预先规定的优先级，B4a 在没有 all-field B4 force tail 的前提下得到最低 RMSD，
force mean/max 优于 validation C0，Stable/NUS 高于 C0，E-hull 差值 CI 跨 0，且
speedup 超过 1.20×。B5 的 max force 更低，但 Stable 低一个结构且 RMSD 稍差，
只被冻结为未运行 final seeds 的验证备选。最终只运行 B4a，没有根据 66000–66063
切换候选或调整阈值。

## 12. 64-seed C0 vs V2 完整质量表

三种方法都使用 66000–66063，均为 64/64 成功。为完整性同时列 V1。

| Metric | C0 | V1 Fallback50 | V2 frozen |
|---|---:|---:|---:|
| E-hull, eV/atom ↓ | 0.11251 | 0.10922 | 0.13251 |
| Stable ↑ | 56.25% | 51.56% | 50.00% |
| NUS ↑ | 31.25% | 28.12% | 26.56% |
| Novel ↑ | 73.44% | 71.88% | 75.00% |
| Unique ↑ | 98.44% | 100.00% | 98.44% |
| RMSD, Å ↓ | 0.06803 | 0.06474 | 0.08093 |
| Force mean, eV/Å ↓ | 0.31050 | 0.26976 | 0.34518 |
| Force median, eV/Å ↓ | 0.18760 | 0.16180 | 0.22084 |
| Force P95, eV/Å ↓ | 1.00371 | 0.83295 | 1.22171 |
| Max force, eV/Å ↓ | 2.19484 | 1.26124 | 2.61633 |

这些是 MatterSim-5M surrogate relaxation 结果，不是 DFT。

## 13. paired CI、median 和 std

以下是 V2 vs C0 的 64 个相同 seed 配对统计；CI 是 10,000 次 bootstrap、seed
20260905 的配对均值差区间。

| Metric | C0 mean / median / std | V2 mean / median / std | Paired Δ mean | 95% CI |
|---|---:|---:|---:|---:|
| E-hull | 0.11251 / 0.09168 / 0.09037 | 0.13251 / 0.10095 / 0.11903 | +0.02000 | [-0.01257, +0.05171] |
| Stable | 0.5625 / 1 / 0.5000 | 0.5000 / 0.5 / 0.5040 | -0.0625 | [-0.21875, +0.09375] |
| NUS | 0.3125 / 0 / 0.4672 | 0.2656 / 0 / 0.4452 | -0.0469 | [-0.20312, +0.10938] |
| Novel | 0.7344 / 1 / 0.4452 | 0.7500 / 1 / 0.4364 | +0.0156 | [-0.12500, +0.15625] |
| Unique | 0.9844 / 1 / 0.1250 | 0.9844 / 1 / 0.1250 | 0.0000 | [-0.04688, +0.04688] |
| RMSD, Å | 0.06803 / 0.01949 / 0.19950 | 0.08093 / 0.02694 / 0.20930 | +0.01290 | [-0.05745, +0.08623] |
| Force, eV/Å | 0.31050 / 0.18760 / 0.38651 | 0.34518 / 0.22084 / 0.43756 | +0.03468 | [-0.09818, +0.17162] |

所有区间跨 0，表示 n=64 没有检测到明确方向的恶化；这不是统计等价证明，因为
本轮没有预先指定 equivalence margin。

## 14. RMSD 是否恢复

**基本恢复，但点估计未完全追平。** V2/C0 RMSD 比为 1.190×，远低于 V1 曾出现
的约 2× 恶化；配对 Δ 为 +0.01290 Å，CI 跨 0。当前证据支持“严重退化已消除”，
不支持“V2 与 C0 已证明等价”。

## 15. Mean/Max force 是否恢复

**没有爆炸性异常，均值接近但尾部仍偏高。** V2/C0 force mean 为 1.112×；P95
为 1.217×，max 为 1.192×。平均力差 CI 跨 0，且 V2 最大值 2.616 eV/Å 未达到
Stage-B C0 已观察到的 3.367 eV/Å，因此不判为极端 outlier；但 P95/max 同向升高，
是 formal256 必须重点确认的风险。

## 16. coverage

- 独立 64-seed：overall Adapter acceptance / second-forward avoidance
  **60.586%**；eligible 仅前 700 steps，折算 eligible acceptance 约 **86.55%**。
- 严格单 H20 八种子：overall acceptance **61.413%**。
- `coverage_target=75%` 是 validation risk threshold 的标称覆盖率，不等于加上
  last-30% exact 后的整条轨迹实际 coverage。

## 17. forward reduction

- 独立 64-seed：**30.293%**；平均 logical score calls 1394.141，其中固定 first
  score 1000 次，second forward 394.141 次。
- 严格单 H20：**30.706%**；平均 logical score calls 1385.875，second forward
  385.875 次，Adapter 700 次，fallback 385.875 次。

因此优先目标 `forward reduction >=20%` 明确满足。

## 18. single-H20 speedup

同 GPU0、batch size 1、相同 seeds 66000–66007、相同精度和环境：

| Method | s/sample mean / median / std | samples/h | Speedup mean / median / std | 95% CI | Peak memory |
|---|---:|---:|---:|---:|---:|
| C0 | 118.749 / 119.022 / 3.026 | 30.316 | 1 / 1 / 0 | [1, 1] | 283.8 MiB |
| V1 | 92.835 / 92.308 / 7.637 | 38.778 | 1.287 / 1.286 / 0.114 | [1.214, 1.361] | 283.2 MiB |
| V2 | 87.384 / 85.394 / 5.474 | 41.198 | **1.362 / 1.373 / 0.067** | **[1.321, 1.406]** | 282.1 MiB |

这里使用 sampler sampling segment 计时，排除每个独立 worker 的模型启动成本；原始
全进程 wall throughput 也已保存在 `single_h20_throughput_raw.json`。

## 19. 8×H20 throughput

采用 seed-level parallelism，每卡一个 worker，不拆单个 sample：

| Method | Successful | Wall seconds | Throughput samples/h |
|---|---:|---:|---:|
| C0 | 64/64 | 1514.139 | 152.166 |
| V1 Fallback50 | 64/64 | 1348.051 | 170.913 |
| V2 frozen | 64/64 | 1292.240 | **178.295** |

此表只代表 8 卡吞吐，不用于计算算法 speedup。

## 20. 是否满足 formal256 准入条件

**最低门槛边界通过，建议进入冻结的确认性 formal256。**

- Efficiency：单 H20 speedup 1.362× 且 CI 下界 1.321×，通过 1.20×；forward
  reduction 30.71%，通过 20% 优先目标。
- Main quality：E-hull、Stable、NUS 点估计偏差，但配对 CI 均跨 0，因此没有
  64-seed 层面的明确统计恶化。
- Geometry：RMSD 为 C0 的 1.19×，没有重现约 2× 恶化，CI 跨 0。
- Force：mean 为 C0 的 1.11×，没有极端 max-force outlier，但 P95/max 偏高。

这只是准入，不是最终 formalize。formal256 必须保持 `34ff467` 对应的 checkpoint、
threshold、schedule、risk model 完全不变，且预先声明 E-hull 与 force-tail 停止标准。

## 21. 当前最大风险

最大风险是 **Stage-B 到独立 Stage-C 的质量排序不稳定**：validation 上 B4a 的
RMSD/force 优于 C0，但 final64 上 E-hull、Stable、NUS、RMSD、force 的点估计整体
偏向不利；同时 early-pos residual MSE 从 0.0339 上升到 0.9004，而冻结 decision
只看 atomic risk。这说明 on-policy one-step 改善仍未完全消除 closed-loop
speed-quality trade-off。其次，所有物理质量结论仅基于 MatterSim-5M surrogate。

## 22. 下一步建议

1. 不调任何 V2 参数，用全新且无交集的 256 seeds（建议 67000–67255）仅跑配对
   C0 + V2 frozen；V1 可作为附加参考，但不能据此切换候选。
2. 在开跑前固定通过标准，至少包含 E-hull/Stable/NUS 配对区间、RMSD ratio、force
   mean、P95、max 以及失败率；重点确认当前 E-hull 和 force-tail 不利点估计。
3. formal256 若通过，再做 A0 + V2 兼容性实验，并对少量代表结构做 DFT 验证。
4. formal256 若失败，执行停止条件：不再扩大 Adapter 或继续扫 threshold，明确报告
   Corrector 后 second-score approximation 的 closed-loop speed-quality trade-off，
   转向不近似 second score 的第二创新方向。

## 完整性、测试与复现审计

- Stage-A DAgger smoke：4 seeds、4000 records，teacher target/rollout alignment、
  progress 方向、shard SHA 全部通过。
- 相关 pytest：**29 passed, 0 failed**；覆盖 exact teacher target、rollout state
  对齐、late exact、forced exact=C0、NaN/missing threshold fail closed、不同 atom
  count、seed 集合无交集以及 final seeds freeze 前未使用。
- Stage-C generation：3 methods × 64 seeds = **192/192** 成功。
- Stage-C quality：3 methods × 64 = **192/192** 弛豫成功。
- MatterGen SHA256：`01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。
- MatterSim-5M SHA256：`e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`。
- Alex-MP reference：845,997 条全量顺序审计，SHA256
  `c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5`。
- H20 batched-det workaround 仅替换 CUDA 3×3 行列式；64 个随机 float64 矩阵的
  最大绝对/相对误差为 `8.88e-16`/`3.81e-15`，autograd 梯度全部有限。
- 未运行 A0 组合；未上传权重、teacher/DAgger tensor、cache、结构或弛豫轨迹。

主要机器可读结果：`stage_b_ablation.csv`、`stage_c_64seed.csv`、
`stage_c_speed_metrics.csv`、`single_h20_metrics.csv`、`quality_metrics.csv`、
`paired_metrics.csv`、`paired_per_seed.csv`、`formal256_admission.json`、
`stage_c_generation_audit.json`、`stage_c_quality_audit.json`。
