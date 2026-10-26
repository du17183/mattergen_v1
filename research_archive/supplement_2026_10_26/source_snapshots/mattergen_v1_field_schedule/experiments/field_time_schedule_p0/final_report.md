# Field-wise Time Schedule P0 FINAL

1. branch / HEAD：`experiment/field-time-schedule-p0` / `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`。
2. Base checkpoint SHA：`01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。
3. diagnostic basis：64 trajectories，calibration/validation=32/32，冻结阈值与 gate 如下。

阈值：`{"A": {"atomic_argmax_change_rate": 0.0, "atomic_entropy": 0.014992803707718806, "atomic_mask_fraction": 0.0, "atomic_mean_max_probability": 0.9973955512046814}, "L": {"cell_angle_max_change_deg": 0.5943072962542787, "cell_angle_rms_change_deg": 0.42147894850893847, "cell_length_max_relative_change": 0.00940039983422504, "cell_length_rms_relative_change": 0.007568211551283561, "cell_metric_ltl_relative_change": 0.016859863133989343, "cell_volume_relative_change": 0.0152636937952408}, "X": {"position_max_cart_displacement": 0.09972141036038294, "position_rms_cart_displacement": 0.07613865823182703}}`。

Validation 收敛统计：A: n=27/32, mean=0.138889, median=0.120000, P25/P75/P90=0.070000/0.170000/0.232000; X: n=27/32, mean=0.065556, median=0.060000, P25/P75/P90=0.050000/0.070000/0.096000; L: n=30/32, mean=0.069000, median=0.060000, P25/P75/P90=0.050000/0.100000/0.102000。

Validation median pair 最大差=0.060000；满足单轨 gap>=0.10 的比例=13/32=40.62%（未收敛字段按终点 T=0；严格三字段均有限口径=7/32=21.88%）。

4–6. Atomic/Position/Cell schedules：NOT_CONSTRUCTED_GATE_FAIL。
7. update counts：NOT_RUN_GATE_FAIL。
8. equal-sparse control schedule：NOT_CONSTRUCTED_GATE_FAIL。
9–11. B0/B-equal-sparse/B-field results：NOT_RUN_GATE_FAIL。
12–21. Mag MAE/Hit、RMSD/Force/MaxF、E-hull/Stable/Novel/Unique/NUS：均无 B-P0 三臂结果。
22. field update count：NOT_RUN_GATE_FAIL。
23. total field update reduction：NOT_RUN_GATE_FAIL；不能声称计算下降。
24–26. runtime/score calls/VRAM：P0 三臂均未运行。
27–28. paired B-field vs B0 / equal-sparse：NOT_RUN_GATE_FAIL（n=0）。
29. **FIELD_TIME_SCHEDULE_P0=FAIL_DIAGNOSTIC（candidate verdict=FAIL）**。
30. Does field-specific scheduling beat equal sparsification? **未检验，不能声称是**。
31. Adaptive CFG used? **NO**。
32. DFT_VERIFIED=False；共享质量指标均为 surrogate。
33. 结论：异质性 gate 失败，按预注册不从噪声中强行构造 schedule，不进入 32-seed P0，也不生成无数据图表。
