# A0 + E3-G 兼容性验证

> 当前分支：`feature/a0-e3g-compatibility64`

本分支验证两个创新点能否串行组合：

```text
A0 Adaptive CFG
→ E3-G Learned-Gated Post-generation Refinement
```

## 冻结结论

```text
A0_E3G_COMPATIBILITY_GO=True
FROZEN_COMMIT=ba2303c284210fdae0a35bb0153a8ef3af45a54c
SEEDS=41000–41063
N=64
```

| 指标 | A0 + E3-G 相对 A0 |
|---|---:|
| 预松弛最大力 | **-27.10%** |
| Relaxation RMSD | -1.93% |
| E-hull | 基本不变 |
| Stable / NUS | 不变 |

## 工作过程

1. 固定 A0 的正式 Adaptive CFG 参数。
2. 固定 E3-G 的后处理配置，不重新训练 MatterGen 主干。
3. 对 64 个新 seeds 生成严格配对的 A0 与 A0 + E3-G 结构。
4. 检查 initial state、原子序列和评估配置。
5. 完成 MatterSim relaxation、配对统计和质量安全门槛判定。

## 结论用途

该实验说明两个创新点在算法接口上兼容，E3-G 能在不明显损害 A0 质量指标的情况下进一步降低预松弛最大力。它是组合兼容性证据，不替代两个创新点各自的 256-seed 正式验证。

第二次独立 64-seed 复现位于 `feature/a0-e3g-independent64`。

## 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

完整项目脉络见[论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)。

## 组合实现

本分支没有训练第三个联合模型。A0 先按冻结 Adaptive CFG 完成扩散生成，然后 E3-G 读取 A0 的最终结构：

```text
same seed / same initial state
→ A0 Adaptive CFG generation
→ 固定原子种类
→ Q3 特征提取与 learned gate
→ 安全 E3 position refinement 或 fallback
→ A0 与 A0+E3-G 分别做 MatterSim relaxation
```

这使两个创新点保持模块化：A0 负责条件采样，E3-G 只负责生成后的局部几何质量。

## 实现位置

| 文件 | 内容 |
|---|---|
| [`research/a0_e3g_compat64.py`](research/a0_e3g_compat64.py) | 64-seed 组合 pipeline、resume、relax 和分析 |
| [`guidance_schedule.py`](mattergen/diffusion/sampling/guidance_schedule.py) | A0 Adaptive CFG |
| [`refiner_eval.py`](research/postgen_fastgate/refiner_eval.py) | E3-G proposal、gate 与 fallback |
| [`configs/q3_e3_pcr_frozen64.json`](configs/q3_e3_pcr_frozen64.json) | 冻结 E3-G 配置 |
| [`tests/test_a0_e3g_compat64.py`](tests/test_a0_e3g_compat64.py) | seed、输入一致性、任务恢复和判定测试 |

## 数据文件

- [冻结 manifest](reports/a0_e3g_compat64/frozen_manifest.md)
- [Seed 审计](reports/a0_e3g_compat64/seed_audit.json)
- [A0 逐结构指标](reports/a0_e3g_compat64/A0/official_metrics_per_structure.csv)
- [A0+E3-G 逐结构指标](reports/a0_e3g_compat64/A0_E3G/official_metrics_per_structure.csv)
- [配对统计](reports/a0_e3g_compat64/paired_statistics.csv)
- [机制分析](reports/a0_e3g_compat64/mechanism_report.md)
- [最终报告](reports/a0_e3g_compat64/final_report.md)

## 复现入口

```bash
python -m pytest tests/test_a0_e3g_compat64.py -q
python -m research.a0_e3g_compat64 status
```

服务器上可使用 `bash tools/a0_e3g_compat64/run.sh` 启动完整 pipeline；它依赖 `/data/dxl` 下未上传的权重和环境。
