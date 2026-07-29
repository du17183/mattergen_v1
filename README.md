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
