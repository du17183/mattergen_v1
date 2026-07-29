# 创新点一正式验证：Adaptive CFG

> 当前分支：`feature/convergence-aware-corrector-gating`

这个分支名称保留了早期探索历史，但本分支最终保留并正式验证的是 **Multi-field Residual-driven Online Adaptive CFG**，不是已被否决的 Corrector Gating。

## 冻结结论

```text
FORMAL_INNOVATION1_CONFIRMED=True
FROZEN_COMMIT=5de00419eea2d8a9be303638f2db8ece15a22366
FORMAL_SEEDS=20000–20255
N=256
```

| 指标 | 相对原始条件 MatterGen |
|---|---:|
| 平均 E-hull | -0.003435 eV/atom |
| Stable | +5.859 pp |
| NUS | +3.516 pp |

这些差值方向为正，但配对统计未全部达到显著水平，因此论文中应同时报告点估计、置信区间和显著性限制。

## 方法

每个采样步分别计算 cell、position、atomic 三字段的 conditional–unconditional score residual，使用指数移动平均在线调整 CFG 强度：

```text
base_guidance=2.0
adaptive_alpha=0.50
adaptive_ema=0.95
adaptive_epsilon=1e-6
guidance_min_scale=0.0
guidance_max_scale=5.0
```

完整 Predictor/Corrector 流程保持不变。

## 工作过程

1. 在 CFG 融合位置加入三字段残差统计和在线 guidance schedule。
2. 完成关闭功能时的一致性、相同 seed 确定性和批次状态重置测试。
3. 通过开发 seeds 冻结唯一参数组。
4. 使用 256 个独立 seeds 完成配对生成和 MatterSim relaxation。
5. 冻结正式 commit 与报告，后续不再调参。

## 本分支与其他分支的关系

- 它是创新点一的正式科学证据分支。
- Adaptive CFG 是最终组合方法的上游采样模块。
- 它不是所有历史分支都共享的代码基础。
- 创新点二的正式证据位于 `feature/q3-e3-pcr-formal256`。

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

完整项目脉络见[论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)。

## 算法实现

核心控制器是 [`GuidanceController`](mattergen/diffusion/sampling/guidance_schedule.py)。对第 t 步和字段 k：

```text
r_k = score_cond,k - score_uncond,k
delta_k = RMS(r_k)
delta = mean(valid delta_k)
ema_t = 0.95 * ema_(t-1) + 0.05 * delta
multiplier = 1 + 0.50 * (delta / (ema_t + 1e-6) - 1)
guidance_t = clamp(base_guidance * multiplier, 0, 5)
```

cell、position、atomic 的张量形状不同，因此先分别约化为标量 RMS，再求有效字段均值。EMA 按 Predictor/Corrector 采样阶段分别维护，批次结束后重置。出现空 residual、NaN/Inf 或非法倍率时，控制器回退到基础 guidance。

## 实现文件

| 文件 | 内容 |
|---|---|
| [`guidance_schedule.py`](mattergen/diffusion/sampling/guidance_schedule.py) | `GuidanceDecision`、`GuidanceController`、EMA 和 clamp |
| [`classifier_free_guidance.py`](mattergen/diffusion/sampling/classifier_free_guidance.py) | `score_residual_rms` 和 `GuidedPredictorCorrector` |
| [`pc_sampler.py`](mattergen/diffusion/sampling/pc_sampler.py) | 保持完整 Predictor/Corrector 采样 |
| [`test_guidance_schedule.py`](mattergen/diffusion/tests/test_guidance_schedule.py) | 边界、确定性、fallback、phase reset 测试 |

## 数据与评价设计

- 基线：官方 `dft_mag_density` 条件 MatterGen 的固定 CFG。
- 方法：相同 checkpoint、相同初始状态和相同 seeds，仅替换 guidance schedule。
- 正式 seeds：`20000–20255`，共 256 对。
- 评价：MatterSim-5M relaxation 后计算 E-hull、Stable、NUS 等指标。
- 该分支保留核心实现；完整正式统计副本集中在论文归档分支，避免把服务器大型缓存提交到 GitHub。

## 验证命令

```bash
python -m pytest mattergen/diffusion/tests/test_guidance_schedule.py -q
```

审阅时重点检查 `test_differently_shaped_field_residuals`、`test_invalid_residual_falls_back`、`test_phase_specific_ema_and_reset`。完整生成需要服务器上的条件 checkpoint 和 MatterSim 权重，GitHub 不包含这些二进制文件。
