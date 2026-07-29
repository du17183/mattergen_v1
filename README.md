# RP-QTFG Phase 0

> 当前分支：`feature/rp-qtfg`

本分支验证训练自由 CHGNet 物理引导能否在保护条件残差、trust region、backtracking 和 fallback 的约束下改善 A0 结构。

## 最终结论

```text
CHGNET_MAG_ORACLE_GO=True
PHYSICS_DIRECTION_GO=True
RP_QTFG_EIGHT_SEED_GO=False
RP_QTFG_MVP_NO_GO=True
THIRTY_TWO_SEED_STARTED=False
```

## Gate 0：离线方向

磁 Oracle 在 8,716 个样本上通过；64 个结构的离线小步修正中：

| 指标 | 改善率 |
|---|---:|
| MatterSim 起始能量 | 78.12% |
| 最大力 | 73.44% |
| Relaxation RMSD | 60.94% |

## Gate 1：在线采样

最接近门槛的 `G1P75S`：

- E-hull：`-0.003353 eV/atom`
- Stable 等离散指标：不变
- RMSD：`+68.28%`
- 最大力：仅 `-0.26%`
- 延迟：`+30.19%`

所有在线候选的 RMSD 恶化约 `+16.3%` 至 `+293.3%`，因此按预设 gate 停止。

## 工作过程

1. 核验训练自由 guidance、clean-x0 恢复和 CHGNet 接口。
2. 完成磁密度候选 Oracle 和离线 CHGNet→MatterSim 方向验证。
3. 实现字段级梯度、条件残差保护、trust region、backtracking 和 A0 fallback。
4. 完成 25/25 专项测试及 38/38 全套相关测试。
5. 运行 8-seed A0/G1/G2 smoke；因 RMSD 系统性恶化未进入 32 seeds。

## 证据

- [Phase 0 最终报告](research/rp_qtfg/artifacts/phase0/reports/final_report.md)
- [论文归档分支](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1)

## 方法实现

RP-QTFG 首先用 A0 score 恢复预测 clean structure，再在中低噪声区间调用冻结 CHGNet：

```text
A0 safe update
→ clean-x0 position/cell estimate
→ CHGNet energy/force/stress objective
→ field-wise gradient normalization
→ 与 A0 conditional residual 做冲突检测
→ trust-region proposal
→ 最多 3 次 backtracking
→ 通过安全检查则使用，否则 exact fallback 到 A0
```

Atomic field 从不接受物理梯度，原子种类不被修改。MatterSim 只用于独立评价，不参与 guidance。

## 实现位置

| 文件 | 内容 |
|---|---|
| [`physics_guidance.py`](research/rp_qtfg/physics_guidance.py) | CHGNet objective、梯度、conflict、trust region 和 backtracking |
| [`sampler.py`](research/rp_qtfg/sampler.py) | A0 safe update 与 physics update 组合 |
| [`mag_oracle.py`](research/rp_qtfg/mag_oracle.py) | CHGNet site magmom Oracle 评估 |
| [`offline_probe.py`](research/rp_qtfg/offline_probe.py) | 64 结构离线方向 Gate 0B |
| [`experiment_config.py`](research/rp_qtfg/experiment_config.py) | G1/G2、起点和 trust radius |
| [`test_rp_qtfg.py`](mattergen/diffusion/tests/test_rp_qtfg.py) | bitwise-off、原子不变、fallback 和 RNG 测试 |

## 数据索引

- [MatterGen 代码映射](research/rp_qtfg/artifacts/phase0/reports/mattergen_code_map.md)
- [磁 Oracle 报告](research/rp_qtfg/artifacts/phase0/reports/mag_oracle_report.md)
- [离线方向报告](research/rp_qtfg/artifacts/phase0/reports/offline_direction/offline_direction_report.md)
- [8-seed 候选对比](research/rp_qtfg/artifacts/phase0/reports/eight_seed/comparisons.csv)
- [8-seed 配对统计](research/rp_qtfg/artifacts/phase0/reports/eight_seed/paired_statistics.json)

## 复现入口

```bash
bash research/rp_qtfg/scripts/status_phase0.sh
python -m pytest mattergen/diffusion/tests/test_rp_qtfg.py -q
```

完整 runner 为 `research/rp_qtfg/scripts/run_phase0.sh`。离线方向通过但在线采样失败，说明静态结构上的 CHGNet 下降方向不能直接等价为扩散轨迹中的安全 guidance。
