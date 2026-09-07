TCL P2 FINAL:
CLEAR GO

DFT_VERIFIED=False

# TCL P2 — Frozen 64-Paired-Seed Independent Validation

## 1. 实验身份与冻结条件

- Branch: `experiment/tcl-p2`
- P1 final parent HEAD: `56efbaaf7f005b058887b69088913ccbf4d3cde5`
- Final HEAD: 见提交后的最终交付信息
- P2 seeds: `82000–82063`，每种方法 64 个，三方法严格 paired
- Historical overlap: `none`（项目范围精确 seed 扫描无 P0/P1 历史重叠）
- P2 training steps: `0`
- 方法：冻结的 `C0 / FT0 / TCL`，未调参、未重新训练、未删除失败或 outlier
- 生成配置：1000 sampling steps，1 corrector step/time，CFG 2.0，目标 `dft_mag_density=0.1`
- 评价：MatterSim-5M surrogate，`fmax=0.05 eV/Å`；不是 DFT
- Base checkpoint SHA256: `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`
- FT0 checkpoint SHA256: `1cd254568463ae4d7774193fa0a0e97925cdad3b6c505c1e5f5dd4e7968f6960`
- TCL checkpoint SHA256: `8dd9879f11edea7f25f587ae72f928a6f3ab2e5c81fc9400c13642364ad90443`

## 2. 执行完整性

| 方法 | Generation | MatterSim relaxation | 配置偏差 | NaN/crash |
|---|---:|---:|---:|---:|
| C0 | 64/64 (100%) | 64/64 (100%) | 0 | 0 |
| FT0 | 64/64 (100%) | 64/64 (100%) | 0 | 0 |
| TCL | 64/64 (100%) | 64/64 (100%) | 0 | 0 |

三组均覆盖完整 `82000–82063`，无缺 seed、无重复 seed。每个生成样本均保留 `extxyz/json/zip` 产物；每组初始与松弛后结构均为 64 frames。

## 3. 完整主指标

### 3.1 Thermodynamic / discovery

| 方法 | E-hull mean | median | P95 | max | Stable | NUS | Novel | Unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.104179 | 0.081748 | 0.341871 | 1.185880 | 62.500% | 35.938% | 70.312% | 100.000% |
| FT0 | 0.146668 | 0.129870 | 0.347640 | 0.543693 | 42.188% | 9.375% | 64.062% | 93.750% |
| TCL | 0.108451 | 0.091374 | 0.256719 | 0.372996 | 56.250% | 32.812% | 75.000% | 100.000% |

E-hull 单位为 eV/atom。

### 3.2 Geometry / force

| 方法 | RMSD mean | median | P95 | max | Atomic force mean | Max-force mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.044624 | 0.009067 | 0.229444 | 0.481677 | 0.142915 | 0.305569 | 0.128762 | 1.091236 | 1.756507 | 2.427682 |
| FT0 | 0.091525 | 0.035988 | 0.382485 | 1.120354 | 0.315159 | 0.699844 | 0.386176 | 1.459548 | 7.198640 | 8.320858 |
| TCL | 0.043545 | 0.026213 | 0.201749 | 0.361550 | 0.162029 | 0.303784 | 0.185986 | 0.728089 | 2.038128 | 2.328436 |

RMSD 单位为 Å，force 单位为 eV/Å。

| 方法 | Max-force >1 | rate | Max-force >2 | rate |
|---|---:|---:|---:|---:|
| C0 | 7/64 | 10.938% | 1/64 | 1.562% |
| FT0 | 9/64 | 14.062% | 3/64 | 4.688% |
| TCL | 3/64 | 4.688% | 1/64 | 1.562% |

### 3.3 Relaxation

| 方法 | mean steps | median | P95 | max | >100 | >200 | >300 | >400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 44.188 | 19.5 | 146.50 | 437 | 5 (7.812%) | 3 (4.688%) | 2 (3.125%) | 1 (1.562%) |
| FT0 | 69.625 | 41.0 | 259.80 | 509 | 11 (17.188%) | 5 (7.812%) | 3 (4.688%) | 1 (1.562%) |
| TCL | 46.250 | 27.5 | 87.55 | 582 | 3 (4.688%) | 2 (3.125%) | 1 (1.562%) | 1 (1.562%) |

## 4. 20,000 次 paired bootstrap

所有 delta 均为 candidate − baseline。对 E-hull、RMSD、force、relaxation，负值更好；对 Stable、NUS、Novel、Unique，正值更好。W/T/L 按“candidate 更优/相同/更差”计数。CI 跨 0 只表示证据不足，不解释为等价或无退化。

### 4.1 TCL − FT0（主要机制对照）

| 指标 | paired delta | 95% CI | W/T/L |
|---|---:|---:|---:|
| E-hull (eV/atom) | -0.038217 | [-0.074808, -0.003471] | 35/0/29 |
| Stable (pp) | +14.0625 | [-3.1250, +31.2500] | 21/31/12 |
| NUS (pp) | +23.4375 | [+9.3750, +37.5000] | 20/39/5 |
| Novel (pp) | +10.9375 | [-4.6875, +26.5625] | 17/37/10 |
| Unique (pp) | +6.2500 | [+1.5625, +12.5000] | 4/60/0 |
| RMSD (Å) | -0.047980 | [-0.096349, -0.008924] | 37/0/27 |
| Atomic-force mean (eV/Å) | -0.153130 | [-0.282442, -0.041052] | 47/0/17 |
| Structure max-force (eV/Å) | -0.396061 | [-0.770943, -0.104541] | 45/0/19 |
| Force >1 (pp) | -9.3750 | [-18.7500, 0.0000] | 8/54/2 |
| Force >2 (pp) | -3.1250 | [-9.3750, +3.1250] | 3/60/1 |
| Relaxation steps | -23.3750 | [-50.3914, +3.8910] | 39/1/24 |

结论：TCL 相对 FT0 的核心信号在独立 64 seeds 上复现。E-hull、NUS、Unique、RMSD、atomic force 和 max-force 的 CI 均支持 TCL 优势；Stable 仍为 +14.06 pp 同方向但 CI 跨 0；relaxation 点估计改善且 39/64 paired wins，但 CI 略跨 0。Novel/Unique 没有 collapse。

### 4.2 TCL − C0（论文可接受性关键对照）

| 指标 | paired delta | 95% CI | W/T/L |
|---|---:|---:|---:|
| E-hull (eV/atom) | +0.004272 | [-0.042327, +0.042501] | 29/0/35 |
| Stable (pp) | -6.2500 | [-21.8750, +9.3750] | 10/40/14 |
| NUS (pp) | -3.1250 | [-18.7500, +12.5000] | 12/38/14 |
| Novel (pp) | +4.6875 | [-10.9375, +20.3125] | 14/39/11 |
| Unique (pp) | 0.0000 | [0.0000, 0.0000] | 0/64/0 |
| RMSD (Å) | -0.001078 | [-0.029879, +0.025915] | 29/0/35 |
| Atomic-force mean (eV/Å) | +0.019114 | [-0.039231, +0.087487] | 33/0/31 |
| Structure max-force (eV/Å) | -0.001785 | [-0.123517, +0.119876] | 31/0/33 |
| Force >1 (pp) | -6.2500 | [-15.6250, +1.5625] | 6/56/2 |
| Force >2 (pp) | 0.0000 | [-4.6875, +4.6875] | 1/62/1 |
| Relaxation steps | +2.0625 | [-23.1727, +28.4383] | 30/3/31 |

结论：TCL 对 C0 的 E-hull、Stable、NUS 和 atomic-force 点估计仍有轻微不利方向，但效应量较小且所有 CI 均跨 0；这些数据不支持“等价”，但也没有出现可信的系统性恶化证据。RMSD 和 max-force 均接近 C0，force >1 的发生率点估计更低；Novel 提高、Unique 持平。

## 5. P0/P1/P2 replication（TCL − FT0）

| Metric | P0 delta/方向 | P1 delta [95% CI]/方向 | P2 delta [95% CI]/方向 | replicated? |
|---|---|---|---|---|
| E-hull | -0.009535 ↓ | -0.017797 [-0.057059, +0.021977] ↓ | -0.038217 [-0.074808, -0.003471] ↓ | yes, all three |
| Stable | 0.000 flat | +28.125 [+9.375, +46.875] ↑ | +14.0625 [-3.125, +31.250] ↑ | yes, P1→P2 |
| NUS | +25.000 pp ↑ | +21.875 [+6.250, +37.500] ↑ | +23.4375 [+9.375, +37.500] ↑ | yes, all three |
| Novel | +25.000 pp ↑ | -9.375 [-25.000, +6.250] ↓ | +10.9375 [-4.688, +26.562] ↑ | no, P1→P2 direction changed |
| RMSD | -0.019737 ↓ | -0.026404 [-0.128775, +0.069754] ↓ | -0.047980 [-0.096349, -0.008924] ↓ | yes, all three |
| Atomic force | -0.083695 ↓ | -0.092882 [-0.173351, -0.006803] ↓ | -0.153130 [-0.282442, -0.041052] ↓ | yes, all three |
| Maximum force | -0.242409 ↓ | -0.151221 [-0.363951, +0.082483] ↓ | -0.396061 [-0.770943, -0.104541] ↓ | yes, all three |
| Relaxation | -9.125 steps ↓ | -22.156 [-59.032, +5.469] ↓ | -23.375 [-50.391, +3.891] ↓ | yes, all three |

八个重点指标中，七个在 P1→P2 保持方向；除 Novel 外，其余信号都复现。NUS、atomic force 最稳定；P2 进一步让 E-hull、RMSD、max-force 的 CI 排除 0。

## 6. 收敛与稳定性回答

1. **E-hull 是否收敛：是，向 C0 收敛。** TCL−C0 从 P1 的 +0.01570 变为 P2 的 +0.00427 eV/atom，P2 CI 为 [-0.04233, +0.04250]；不能宣称等价，但没有可信系统性恶化。
2. **Stable 是否收敛：部分收敛。** TCL−C0 从 -9.375 pp 缩小到 -6.250 pp，CI 也收窄但仍跨 0；TCL−FT0 优势从 +28.125 pp 变为 +14.0625 pp，同方向但 P2 CI 跨 0。
3. **NUS 是否复现：明确复现。** TCL−FT0 为 +23.4375 pp，95% CI [+9.375, +37.500]；对 C0 为 -3.125 pp 且 CI 跨 0。
4. **RMSD 是否稳定：是。** TCL mean/P95/max 均优于 FT0，TCL−FT0 CI 排除 0；TCL 与 C0 mean 几乎相同，且 TCL 没有严重 RMSD tail。
5. **Force 是否改善：明确改善 FT0，接近 C0。** TCL−FT0 atomic/max-force 均显著下降；对 C0 的 max-force mean 几乎相同，force >1 从 C0 7 个降至 TCL 3 个，但 paired proportion CI 仍跨 0。
6. **Relaxation tail 是否改善：总体改善。** TCL 的 P95 87.55，优于 C0 146.50 和 FT0 259.80；>200 个数 TCL/FT0/C0 为 2/5/3。TCL 有一个 582-step 极端 seed，必须保留并报告，因此不能称所有尾部都更好。
7. **Novel / Unique 是否保护：是。** TCL 为 75% / 100%，相对 FT0 分别 +10.94 pp / +6.25 pp，相对 C0 为 +4.69 pp / 0 pp。

## 7. 严重 outlier seeds

沿用 P1 阈值：`RMSD >0.5 Å`、`structure max-force >1 eV/Å`、`relaxation >200 steps`。所有 outlier 均保留。

### C0（8 个 unique seeds）

- `82017 Eu5V2N5`: steps 232
- `82023 Fe7P2O`: max-force 1.0492
- `82028 Al(FeO2)2`: max-force 1.0987
- `82034 Rb3Mn2O5`: max-force 1.0490, steps 437
- `82052 Tb(Mn2Al)2`: max-force 1.3623
- `82053 EuSeO3`: max-force 1.1137, steps 378
- `82055 GdMnFeCo9`: max-force 1.0430
- `82059 GdCo13(B2H)2`: max-force 2.4277

### FT0（10 个 unique seeds）

- `82003 Mn5C3`: max-force 1.1719
- `82010 Mn2C`: RMSD 1.1204, steps 270
- `82015 Mn8(SnC)3`: max-force 1.4213, steps 202
- `82026 MnC`: max-force 8.3209, steps 325
- `82027 Mn3C`: max-force 6.5396
- `82032 Mn13PC6`: RMSD 0.5218, max-force 3.1573, steps 509
- `82047 Mn11FeC6`: max-force 1.1719, steps 387
- `82049 Mn9C7`: max-force 1.4663
- `82050 Mn5C3`: max-force 1.4045
- `82059 Mn4C`: max-force 1.3623

### TCL（4 个 unique seeds）

- `82025 Gd2BeZn(Si3Pt)2`: max-force 1.8676
- `82030 Mn`: steps 582
- `82047 Ti2Mn7P8C`: max-force 1.1152, steps 220
- `82055 GdMn3C2`: max-force 2.3284

严重 tail 计数汇总：RMSD tail C0/FT0/TCL = 0/2/0；force >1 = 7/9/3；force >2 = 1/3/1；relaxation >200 = 3/5/2。TCL 明显减少 FT0 的系统性 bad tail，也没有相对 C0 引入系统性 tail；但 TCL seed 82030 的 582-step 单点为三组最大值，不能忽略。

## 8. 预注册决策

1. TCL 相对 FT0 的主要 P1 signal 大体复现：**满足**。
2. Stable / NUS / force 至少两个保持优势：**满足**。NUS 与 force 的 P2 CI 明确支持 TCL；Stable 同方向但 CI 跨 0。
3. TCL 相对 C0 不出现可信 E-hull / Stable / NUS 系统性恶化：**满足**。不利点估计均较小且 CI 跨 0；这不等同于证明等价。
4. Geometry / force tail 无 robustness collapse：**满足**。TCL 严重 RMSD/force tail 少于 FT0，未多于 C0。
5. Novel / Unique 无 collapse：**满足**。TCL 分别为 75% / 100%。

因此最终判定：**CLEAR GO**。

## 9. 科学结论与下一步

P2 回答了唯一科学问题：冻结 TCL 在全新 64 paired seeds 上独立复现了相对 FT0 的主要收益，尤其是 NUS、E-hull、RMSD 和 force；相对原始 C0 的核心质量处于可接受范围，未观察到可信的系统性退化。证据仍不允许宣称 TCL 全面或统计等价于 C0；Stable 的 FT0 优势强度也比 P1 下降，需要在更大样本中继续估计。

- 是否值得继续正式研究：**是**。
- 推荐下一步：**冻结当前 TCL，进行 C0/FT0/TCL formal256 独立评价**。
- 本轮未启动 formal256，未运行 DML/TCL+DML，也不建议在 formal256 前根据 P2 结果调参。
- 机制消融可在 formal256 之后用于论文解释；当前优先级高于 DML 新路线，但低于先获得冻结方法的大样本确认。
