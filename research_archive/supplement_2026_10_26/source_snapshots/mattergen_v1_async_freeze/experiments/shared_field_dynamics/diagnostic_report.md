# Shared Field Dynamics Diagnostic FINAL

## 实验口径

- 64 条 fresh trajectories；calibration 32、validation 32，拆分单位为完整 trajectory。
- Official dft_mag_density Full FT；target=0.2 Å^-3；CFG=2；1000 steps；corrector=1；Adaptive CFG=OFF。
- 每条 55 个 observation points，位于 corrector 之后、predictor 更新之前；诊断不增加 GemNet forward，均为 2000 score calls。
- 网格覆盖 model_t=1.0→0.001；请求的 t=0.00 映射到官方最后一次模型调用 t=0.001，不虚构额外 t=0 forward。
- 属性 evaluator 为冻结 CHGNet magnetic-density surrogate（held-out MAE=0.001449，Pearson=0.9499，tau=0.01）；SURROGATE_PROPERTY_EVAL=True，DFT_VERIFIED=False。
- 诊断统计共 3520 行，clean estimates 有限 3520/3520；64 轨迹平均采样 83.480 秒，合计 128000 score calls。
- 当前 symmetric-lattice gauge 下 L^T L 与 row-vector physical Gram L L^T 的 residual 差异很小：mean abs diff=1.275e-07，max=1.609e-05；仍按预注册使用 L^T L 作 gate。

## 收敛定义

- Atomic：CFG 合并后的 p(x0|xt) logits 的平均 entropy、argmax change、mean max probability，并强制 mask_fraction=0。
- Position：相邻 observation 的 clean-x0 分数坐标最小像位移，使用前一 clean cell 映射到笛卡尔空间。
- Cell：相邻 clean-x0 的 L^T L 相对 Frobenius 变化为主，同时约束 volume、length、angle 变化；L L^T 仅作 row-lattice audit。
- 阈值只由 calibration 32 在 t<=0.05 的尾部冻结：残差 q90、confidence q10；首次连续 K=3 个 observation 满足即得到 T_field；t>0.5 禁止判为可冻结。

## 冻结阈值

```json
{
  "A": {
    "atomic_argmax_change_rate": 0.0,
    "atomic_entropy": 0.014992803707718806,
    "atomic_mask_fraction": 0.0,
    "atomic_mean_max_probability": 0.9973955512046814
  },
  "L": {
    "cell_angle_max_change_deg": 0.5943072962542787,
    "cell_angle_rms_change_deg": 0.42147894850893847,
    "cell_length_max_relative_change": 0.00940039983422504,
    "cell_length_rms_relative_change": 0.007568211551283561,
    "cell_metric_ltl_relative_change": 0.016859863133989343,
    "cell_volume_relative_change": 0.0152636937952408
  },
  "X": {
    "position_max_cart_displacement": 0.09972141036038294,
    "position_rms_cart_displacement": 0.07613865823182703
  }
}
```

## Validation convergence statistics

| Field | converged | mean T | median T | P25 | P75 | P90 | r(T,num_atoms) | r(T,MaxF) | r(T,E-hull) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 27/32 | 0.1389 | 0.1200 | 0.0700 | 0.1700 | 0.2320 | -0.2346 | -0.0166 | 0.0885 |
| X | 27/32 | 0.0656 | 0.0600 | 0.0500 | 0.0700 | 0.0960 | 0.6007 | -0.1098 | -0.3448 |
| L | 30/32 | 0.0690 | 0.0600 | 0.0500 | 0.1000 | 0.1020 | -0.1882 | -0.2351 | -0.1704 |

最终 CHGNet property/validity：Mag MAE=0.011513 Å^-3，Hit=39/64，valid=64/64。

最终 MatterSim 质量：已合并；E-hull=0.078961 eV/atom，Stable=79.69%，Novel=45.31%，Unique=67.19%，NUS=18.75%，RMSD=0.032229，MaxF mean/median/P95=0.151970/0.088529/0.428771 eV/Å。

## Convergence order（validation only）

First-field（并列与 NC 独立成类，分母 32）计数：`{'A': 15, 'NC': 11, 'X': 3, 'L': 2, 'A=X=L': 1}`。

First-field 比例：`{'A': 0.46875, 'NC': 0.34375, 'X': 0.09375, 'L': 0.0625, 'A=X=L': 0.03125}`。即 Atomic=46.88%、Position=9.38%、Cell=6.25%；其余为 tie 或 NC。

完整 ordering 分布：`{'NC': 11, 'A>L>X': 10, 'A>X>L': 6, 'X>A>L': 3, 'L>X>A': 2}`。

## Heterogeneity gate（validation only）

- validation median T：`{'A': 0.1200000047683715, 'X': 0.0600000023841857, 'L': 0.0600000023841857}`
- 最大 median pair difference：`0.060000`（阈值 0.10）
- 单轨存在 >=0.10 pair gap：13/32 = 40.62%（阈值 60%）
- 更严格的三字段 T 均有限口径：7/32 = 21.88%。主口径将未完成 K-stable 的字段按终点 T=0 截尾，因此更有利于检出异质性。
- **FIELD_CONVERGENCE_HETEROGENEITY=FALSE**

若 gate=False，A/B 均按预注册直接记为 FAIL_DIAGNOSTIC，禁止强行实现；若 gate=True，才进入两个相互独立的 P0。
