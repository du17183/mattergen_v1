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
