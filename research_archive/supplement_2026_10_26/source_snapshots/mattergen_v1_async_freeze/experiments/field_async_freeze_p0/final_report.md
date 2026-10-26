# Field-Asynchronous Freeze P0 FINAL

1. branch / HEAD：`experiment/field-async-freeze-p0` / `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`。
2. Base checkpoint SHA：`01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。
3. diagnostic sample count：64（calibration=32，validation=32）。
4. Atomic convergence：clean categorical entropy + argmax change + confidence，且 mask_fraction=0。
5. Position convergence：相邻 clean-x0 的 periodic Cartesian RMS/max displacement。
6. Cell convergence：clean-x0 的 L^T L、volume、length、angle change。
7. thresholds：只由 calibration 32 的 t<=0.05 尾部冻结，residual q90 / confidence q10。
8. convergence-time statistics：见共享统计。
9. convergence-order statistics：见共享 `diagnostic_report.md`。
10. **FIELD_CONVERGENCE_HETEROGENEITY=FALSE**。

阈值：`{"A": {"atomic_argmax_change_rate": 0.0, "atomic_entropy": 0.014992803707718806, "atomic_mask_fraction": 0.0, "atomic_mean_max_probability": 0.9973955512046814}, "L": {"cell_angle_max_change_deg": 0.5943072962542787, "cell_angle_rms_change_deg": 0.42147894850893847, "cell_length_max_relative_change": 0.00940039983422504, "cell_length_rms_relative_change": 0.007568211551283561, "cell_metric_ltl_relative_change": 0.016859863133989343, "cell_volume_relative_change": 0.0152636937952408}, "X": {"position_max_cart_displacement": 0.09972141036038294, "position_rms_cart_displacement": 0.07613865823182703}}`。

Validation 收敛统计：A: n=27/32, mean=0.138889, median=0.120000, P25/P75/P90=0.070000/0.170000/0.232000; X: n=27/32, mean=0.065556, median=0.060000, P25/P75/P90=0.050000/0.070000/0.096000; L: n=30/32, mean=0.069000, median=0.060000, P25/P75/P90=0.050000/0.100000/0.102000。

Validation median pair 最大差=0.060000；满足单轨 gap>=0.10 的比例=13/32=40.62%（未收敛字段按终点 T=0；严格三字段均有限口径=7/32=21.88%）。

11. freeze state machine：预注册为 ACTIVE→CANDIDATE_STABLE→FROZEN，但因 gate 失败未实现/未运行。
12. minimum active horizon：预注册 t>0.5 禁止 freeze；未运行。
13. K=3；未运行。
14. unfreeze rule：预注册每 20 steps 做 2×threshold shadow check；未运行。
15. B0 results：NOT_RUN_GATE_FAIL。
16. F-static results：NOT_RUN_GATE_FAIL。
17. F-adaptive results：NOT_RUN_GATE_FAIL。
18–27. Mag MAE/Hit、RMSD/Force/MaxF、E-hull/Stable/Novel/Unique/NUS：均无 A-P0 结果，禁止把共享诊断 B0 当成三臂实验。
28–30. freeze time/frozen fraction/unfreeze count：NOT_RUN_GATE_FAIL。
31–33. runtime/score calls/VRAM：P0 三臂均未运行；不能声称效率收益。
34. paired statistics：NOT_RUN_GATE_FAIL（n=0）。
35. **FIELD_ASYNC_FREEZE_P0=FAIL_DIAGNOSTIC（candidate verdict=FAIL）**。
36. Is adaptive better than static? **未检验，不能声称是**。
37. Adaptive CFG used? **NO**。
38. DFT_VERIFIED=False；共享质量指标均为 surrogate。
39. 结论：validation 未证明三个字段具有足够大的收敛时间异质性，因此按预注册停止，不构造动态冻结 sampler，不进行 32-seed P0，也不生成无数据图表。
