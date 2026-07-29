# MatterGen 两项创新研究：共享代码基线

> 当前分支：`main`

本仓库基于 MatterGen 开展材料生成研究。这个分支不再保留上游 MatterGen 的通用安装与产品说明；它只记录本项目的共享代码基线、已冻结结论和各实验分支入口。

## 本分支定位

`main` 是两项最终创新共同使用的稳定集成基线：

- 创新点一：Multi-field Residual-driven Online Adaptive CFG。
- 创新点二：Q3 E3-PCR Learned-Gated Post-generation Refinement。
- 保留完整 Predictor/Corrector 采样流程。
- 不包含 REPA、Corrector Gating 等已判定 No-Go 的实验实现。

`main` 适合用于理解共享代码和继续集成，不应替代各正式验证分支中的冻结报告。

## 当前总体结论

| 项目 | 当前结论 | 主要证据 |
|---|---|---|
| Adaptive CFG | 正式验证通过 | 256-seed 分支中 E-hull、Stable、NUS 均呈正向变化 |
| Q3 E3-PCR | 正式验证通过 | 独立 256-seed 中预松弛最大力下降 23.28% |
| A0 + E3-G 兼容性 | 通过 | 两次独立 64-seed 均保持质量并降低最大力 |
| 泄漏诊断 | 仅诊断、不可作为独立验证 | 重叠 seeds 显著夸大安全性 |

所有稳定性与 E-hull 结论均来自 MatterSim-5M 代理评价：

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

## 主要工作过程

1. 在官方条件 MatterGen 上实现三字段残差驱动的在线 Adaptive CFG。
2. 冻结创新点一参数并完成 256-seed 独立验证。
3. 依次排查采样删减、REPA、物理引导和 GPU 执行优化路线。
4. 从六个后生成质量模块中筛选出 Q3 E3-PCR。
5. 完成 Q3 的 64-seed 冻结验证、256-seed 正式验证。
6. 完成 A0 与 E3-G 的兼容性及独立复现。
7. 专门审计训练—测试 seed 重叠，保留负面诊断证据。

## 分支导航

- [创新点一正式验证](https://github.com/du17183/mattergen_v1/tree/feature/convergence-aware-corrector-gating)
- [创新点二正式验证](https://github.com/du17183/mattergen_v1/tree/feature/q3-e3-pcr-formal256)
- [A0 + E3-G 兼容性验证](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-compatibility64)
- [A0 + E3-G 第二次独立验证](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-independent64)
- [泄漏诊断](https://github.com/du17183/mattergen_v1/tree/experiment/a0-e3g-leakage-diagnostic256)
- [论文归档与全局索引](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 使用建议

- 查看整个项目：进入 `archive/thesis-analysis-package-v1`。
- 查看某项实验：点击对应分支名称，再阅读该分支根目录的 `README.md`。
- 固定复现版本：使用报告中记录的 commit，而不是把所有分支合并到 `main`。
- 权重、环境、数据集和大型缓存不在 GitHub 中，需在服务器按报告路径准备。

## 核心实现位置

| 文件 | 作用 |
|---|---|
| [`mattergen/diffusion/sampling/guidance_schedule.py`](mattergen/diffusion/sampling/guidance_schedule.py) | `GuidanceController`、EMA 状态、分阶段/自适应倍率、上下界和异常回退 |
| [`mattergen/diffusion/sampling/classifier_free_guidance.py`](mattergen/diffusion/sampling/classifier_free_guidance.py) | 三字段 residual RMS、conditional/unconditional score 融合、trace |
| [`mattergen/diffusion/sampling/pc_sampler.py`](mattergen/diffusion/sampling/pc_sampler.py) | Predictor/Corrector 采样主循环 |
| [`mattergen/diffusion/tests/test_guidance_schedule.py`](mattergen/diffusion/tests/test_guidance_schedule.py) | EMA、phase reset、clamp、无效 residual fallback 和 CLI 参数测试 |

Adaptive CFG 每一步执行：

```text
conditional score + unconditional score
→ 分别计算 cell / position / atomic residual RMS
→ 更新当前采样阶段的 residual EMA
→ 计算 adaptive multiplier
→ clamp 到冻结 guidance 范围
→ 进入原 Predictor/Corrector 更新
```

## 最小代码验证

```bash
python -m pytest mattergen/diffusion/tests/test_guidance_schedule.py -q
```

正式科学数字不要从 `main` 重新推断，应进入对应冻结分支读取 `final_report`、逐 seed CSV 和 frozen manifest。服务器运行还需要未上传的 MatterGen/MatterSim 权重及项目环境。

## 数据与版本管理规则

- `main` 保存共享实现，不保存所有实验的大型生成与松弛缓存。
- 每个实验分支的根 README 是该实验入口；报告和小型 CSV/JSON 与分支一起版本化。
- 正式结果以 frozen commit、seed audit 和 paired statistics 为准。
- 旧 No-Go 分支不合并回 `main`，用于保留工作过程和负面证据。
