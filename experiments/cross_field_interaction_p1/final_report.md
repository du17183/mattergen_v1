P1 FINAL: FAIL

`DFT_VERIFIED=False`。本报告全部能量、力、松弛和稳定性结论来自 MatterSim-5M surrogate evaluation，不是 DFT 验证。

# Cross-field Feature Interaction P1 独立验证报告

## 总体结论

冻结 CFI 在 32 个全新 paired seeds 上没有稳定复现 P0 中相对 Matched MLP 的热力学质量优势。CFI 相对 MLP 的 E-hull mean 略差 `+0.002760 eV/atom`，Stable 低 `12.5 pp`，NUS 低 `25.0 pp`，Novel 低 `15.625 pp`；其中 NUS 的 paired bootstrap 95% CI 为 `[-43.75, -6.25] pp`，方向明确不利。CFI 相对 C0 虽然 E-hull mean、Stable、RMSD mean 和力指标方向改善，但 Novel 降低 `18.75 pp`，NUS 也降低 `6.25 pp`，且 E-hull/RMSD 的 CI 均跨 0。

CFI 确实展现出相对 MLP 更好的几何和力稳健性：RMSD mean、atomic force mean、structure max-force mean 均降低，后两者 CI 不跨 0；CFI 也没有 `RMSD >0.5 Å`、`force >1 eV/Å` 或 `>400-step` 样本。然而这不足以支持 CFI 在核心生成质量上稳定优于 generic PEFT。按预注册规则，P0 的 E-hull、Stable、NUS 核心 signal 基本未复现，且 Novel 明显下降，因此判定 FAIL，不建议进入 64-seed P2。

## 完整 50 项汇报

1. **Branch**：`experiment/cross-field-interaction-p1`。

2. **Final HEAD**：最终提交完成后在交付消息中给出。Git commit 无法在自身内容中嵌入自身 hash。

3. **P1 seeds**：实际使用 `79000–79031`，共 32 个连续 paired seeds。原请求区间 `78000–78031` 中的 `78000–78007` 已被 Quality Adapter P0 真实生成使用，按要求整体替换，未局部换 seed。

4. **历史独立性**：启动前对历史实验文本记录和 seed 目录精确检索，`79000–79031` 无命中；与 Adaptive CFG、Q3、V1/V2/V3、Quality Adapter、Transformer P0 和 CFI P0 独立。

5. **C0 generation success**：32/32，100%。

6. **MLP generation success**：32/32，100%。

7. **CFI generation success**：32/32，100%。总生成 96/96，无 silent drop。

8. **C0 MatterSim success**：32/32，100%。

9. **MLP MatterSim success**：32/32，100%。

10. **CFI MatterSim success**：32/32，100%。总松弛 96/96；MatterSim-5M，`fmax=0.05`。

11. **E-hull 汇总**，单位 eV/atom：

| Method | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| C0 | 0.111410 | 0.094934 | 0.273816 | 0.369608 |
| MLP | **0.096489** | **0.059646** | **0.248079** | 0.577455 |
| CFI | 0.099249 | 0.069467 | 0.297380 | 0.370209 |

12. **Stable**：C0 `0.50000`，MLP `0.71875`，CFI `0.59375`。

13. **NUS**：C0 `0.21875`，MLP `0.40625`，CFI `0.15625`。

14. **Novel**：C0 `0.65625`，MLP `0.62500`，CFI `0.46875`。

15. **Unique**：C0、MLP、CFI 均为 `1.00000`，没有 uniqueness collapse；CFI 存在 Novel collapse。

16. **RMSD 汇总**，单位 Å：

| Method | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| C0 | 0.045233 | 0.006332 | **0.110240** | 0.791344 |
| MLP | 0.115636 | 0.014079 | 0.614126 | 1.593062 |
| CFI | **0.035009** | **0.002470** | 0.161844 | **0.424149** |

17. **Atomic force mean**，单位 eV/Å：C0 `0.085449`，MLP `0.157786`，CFI `0.068764`。

18. **Structure max-force 汇总**，单位 eV/Å：

| Method | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| C0 | 0.176395 | 0.107635 | 0.590427 | 0.846101 |
| MLP | 0.356066 | 0.110385 | 0.991424 | 4.320723 |
| CFI | **0.150192** | **0.083605** | **0.588855** | 0.869435 |

19. **Relaxation steps 汇总**：

| Method | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| C0 | **32.4375** | 24.0 | **87.55** | 203 |
| MLP | 58.0625 | 21.5 | 166.35 | 717 |
| CFI | 39.8438 | **16.0** | 165.60 | 388 |

20. **大于 100 steps**：C0 `1/32=3.125%`；MLP `3/32=9.375%`；CFI `2/32=6.25%`。

21. **大于 200 steps**：C0 `1/32=3.125%`；MLP `2/32=6.25%`；CFI `2/32=6.25%`。

22. **大于 400 steps**：C0 `0/32=0%`；MLP `1/32=3.125%`；CFI `0/32=0%`。

23. **严重 outliers**：正式主结果全部保留。

| Method | Seed | Formula | E-hull | RMSD | Atomic force mean | Max force | Steps | Flags |
|---|---:|---|---:|---:|---:|---:|---:|---|
| C0 | 79028 | NdMn2AsO4 | 0.266529 | 0.791344 | 0.374491 | 0.588642 | 203 | RMSD>0.5; steps>200 |
| MLP | 79002 | Eu2Al3Au | 0.099990 | 1.593062 | 0.191739 | 0.592006 | 717 | RMSD>0.5; steps>200; steps>400 |
| MLP | 79004 | K2Mn3VO8 | 0.095875 | 0.154116 | 0.207148 | 0.408470 | 234 | steps>200 |
| MLP | 79028 | Mn5NO2 | 0.577455 | 1.176361 | 1.337157 | 4.320723 | 79 | RMSD>0.5; max-force>1 |
| MLP | 79031 | EuGd2H3O8 | 0.103952 | 0.120958 | 0.711547 | 1.373570 | 111 | max-force>1 |
| CFI | 79004 | MgMn3V2O8 | 0.211492 | 0.220248 | 0.297322 | 0.585299 | 280 | steps>200 |
| CFI | 79013 | Eu10Si5H3 | 0.147661 | 0.424149 | 0.111403 | 0.322218 | 388 | steps>200 |

   Leave-one-largest 辅助敏感性不替代 full mean：去掉各方法最大值后，C0/MLP/CFI 的 RMSD mean 为 `0.02116/0.06798/0.02246`，relaxation mean 为 `26.94/36.81/28.61`，max-force mean 为 `0.15479/0.22817/0.12699`。MLP 的 relaxation mean 劣势明显受 717-step 样本驱动；CFI 的正式结果仍保留全部样本。

24. **CFI − C0 E-hull**：mean delta `-0.012161 eV/atom`，95% CI `[-0.056466, +0.034435]`。改善方向，但当前样本不足以确认。

25. **CFI − C0 Stable**：`+9.375 pp`，95% CI `[-9.375, +28.125] pp`。改善方向，CI 跨 0。

26. **CFI − C0 NUS**：`-6.250 pp`，95% CI `[-21.875, +9.375] pp`。P0 正向方向未复现。

27. **CFI − C0 Novel**：`-18.750 pp`，95% CI `[-37.500, 0.000] pp`。是有意义的负向 tradeoff。

28. **CFI − C0 RMSD mean**：`-0.010224 Å`，95% CI `[-0.073037, +0.035819]`。mean 改善但 CI 跨 0；CFI P95 高于 C0。

29. **CFI − C0 力**：structure max-force mean delta `-0.026203 eV/Å`，95% CI `[-0.103496, +0.053230]`；atomic force mean delta `-0.016686`，CI `[-0.050597, +0.014630]`。均为改善方向但不足以确认。

30. **CFI − C0 relaxation mean**：`+7.406 steps`，95% CI `[-16.000, +35.188]`。mean/P95 较差，但 median 更好且无 >400 tail；不是系统性崩坏。

31. **CFI − MLP E-hull**：`+0.002760 eV/atom`，95% CI `[-0.049109, +0.050674]`。没有复现 P0 的额外 E-hull 收益。

32. **CFI − MLP Stable**：`-12.500 pp`，95% CI `[-34.375, +9.375] pp`。P0 正向方向反转。

33. **CFI − MLP NUS**：`-25.000 pp`，95% CI `[-43.750, -6.250] pp`。可信负向，CI 不跨 0。

34. **CFI − MLP Novel**：`-15.625 pp`，95% CI `[-31.250, 0.000] pp`。明显负向。

35. **CFI − MLP RMSD mean**：`-0.080627 Å`，95% CI `[-0.215063, +0.016364]`。几何改善方向强，但均值受 MLP 两个大 RMSD outlier 影响，CI 仍跨 0。

36. **CFI − MLP 力**：structure max-force mean delta `-0.205875 eV/Å`，95% CI `[-0.516170, -0.005700]`；atomic force mean delta `-0.089022`，CI `[-0.189383, -0.018965]`。这是 CFI 相对 MLP 最可信的额外优势。

37. **CFI − MLP relaxation mean**：`-18.219 steps`，95% CI `[-71.407, +21.188]`。改善方向，但去掉 MLP 的 717-step 最大值后 MLP mean 为 36.81，低于 CFI 39.84，故不能宣称稳定优势。

38. **P0 E-hull signal 是否复现**：相对 C0 仅弱方向复现；相对 MLP 未复现，delta 从 P0 的 `-0.06595` 变为 `+0.00276`。

39. **P0 Stable signal 是否复现**：相对 C0 方向复现 `+9.375 pp` 但不确定；相对 MLP 从 P0 `+25 pp` 反转为 `-12.5 pp`，未复现。

40. **P0 NUS signal 是否复现**：没有。相对 C0 为 `-6.25 pp`；相对 MLP 为 `-25 pp`，且 CI 完全低于 0。

41. **P0 RMSD improvement 是否复现**：mean 和 max 相对两组对照方向改善；相对 MLP 的 geometry tail 明显更好。但相对 C0 的 P95 从 `0.11024` 增至 `0.16184`，且 paired mean CI 跨 0，因此是部分复现，不是全面确认。

42. **495-step 类似 long-tail 是否再次出现**：没有。CFI max 为 388，0 个 >400；P0 的 495-step tail 未再次出现。

43. **CFI long-tail occurrence 是否高于 C0/MLP**：CFI >200 为 2 个，高于 C0 的 1 个、等于 MLP 的 2 个；CFI >400 为 0，等于 C0、低于 MLP。不能认为 CFI 系统性高于两个 baseline，但其 P95 steps 明显高于 C0。

44. **MLP 本轮相对 C0**：MLP 改善 E-hull mean `-0.014921`、Stable `+21.875 pp`、NUS `+18.75 pp`，Novel 仅 `-3.125 pp`；但 RMSD mean 增加 `0.070403 Å`、max-force mean 增加 `0.179671 eV/Å`、relaxation mean 增加 `25.625 steps`，并出现 717-step、1.593 Å 和 4.321 eV/Å tails。generic PEFT 表现为质量提升但几何鲁棒性变差的明显 tradeoff。

45. **CFI 是否稳定优于 Matched MLP**：否。CFI 在 RMSD/force/极端 tail 上更好，但 E-hull 没有额外收益，Stable/NUS/Novel 均更差，NUS 差异具有不跨 0 的 paired CI。

46. **是否支持 Cross-field interaction 提供超越 generic PEFT 的额外收益**：不支持这一总体表述。数据只支持 CFI 可能比 plain MLP 更好地约束几何与力，不支持它在核心生成质量、稳定性和 NUS 上有额外收益。

47. **P1 FINAL**：`FAIL`。

48. **是否建议进入 64-seed P2**：不建议。

49. **最主要科学原因**：P0 中决定 cross-field 机制价值的相对 MLP 信号未复现；CFI 的 Stable、NUS、Novel 全部低于 MLP，尤其 NUS `-25 pp` 的 95% CI 不跨 0，同时相对 C0 存在 Novel `-18.75 pp` 的明显损失。几何收益不足以抵消核心质量与新颖性信号失败。

50. **唯一下一步建议**：停止当前 CFI 路线，不修改 gate、hidden、RBF 或插入点，也不启动 P2；回到已完成的候选创新路线结果中，选择一个在独立 seeds 上同时保持质量与鲁棒性的方向作为论文第二创新主线。

## 协议与计算说明

- 三组均使用 `dft_mag_density=0.1`、constant CFG `2.0`、1000 diffusion steps、原 Predictor/Corrector、每 timestep 1 次 corrector、batch=1。
- C0/MLP/CFI 平均纯采样耗时分别为 `116.42/116.88/124.94 s`；CFI 约比 C0 慢 7.3%。
- MLP checkpoint SHA256：`f21bcd702a81fff04e48a6f79507dbdce97bebaedb19e7f72868c3743861ee78`。
- CFI checkpoint SHA256：`5573b71a6bd6a8b57beb36ff27ce309f81bc3f82fb7b4759c3a7aa1b248b5d1b`。
- 20,000 次 bootstrap 全部保持 seed pairing；bootstrap seed 为 `20260906`。
- CI 跨 0 表述为当前样本不足以确认差异，不解释为方法等价。
- 本轮未训练、未生成 checkpoint、未修改模型、未做新消融、未启动 P2。
