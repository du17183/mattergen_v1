# Site-Level Position Convergence Diagnostic FINAL

1. trajectories reused/generated：`REUSED_EXISTING_TRAJECTORIES=False`。原 64 条仅有 field aggregate，故用相同 seeds 作只读重放；最终结构和 MatterSim per-atom force 复用。
2. sample count：64 trajectories（calibration 32 / validation 32），共 576 atomic sites（calibration 276 / validation 300）。
3. observation count：正式分析使用与既有 field diagnostic 完全相同的 55 个 predictor points / trajectory；复跑被动保存全部 1000 点作审计。每条仍为 2000 score calls，无额外 GemNet forward。
4. clean-estimate：`wrap(x_t + sigma_t^2 * CFG-guided score_pos)`，predictor 前、官方 corrector 后。
5. site displacement：相邻 clean estimate，用前一 clean cell 和 ASE `find_mic` float64 求严格 triclinic minimum-image Cartesian distance。
6. threshold：`0.07655689 Å`；calibration t<=0.05 trajectory-balanced q90，validation 未参与。
7. K=3 consecutive global predictor observations；t>0.5 不判 convergence。未收敛 site 保留 `T_site=NaN`，结构 spread 使用 endpoint-censored `T=0` 并明确标记。
8. median SiteSpread：`0.000000`。
9. mean SiteSpread：`0.011969`。
10. P25/P75/P90 SiteSpread：`0.000000 / 0.009250 / 0.048800`。
11. fraction SiteSpread>=0.10：`0.00%`。
12. fraction max-min>=0.15：`0.00%`；同晶体同时有 early/late converged site：`9.38%`。
13. within-species heterogeneity：validation pooled within-species std=`0.036684`，median species std=`0.028214`。
14. species-level differences：validation eta²=`0.2077`；`SPECIES_DRIVEN_HETEROGENEITY=FALSE`。
Measurement-noise audit：converged validation sites 中 transient 比例=`9.75%`；0.9× threshold 下 SiteSpread gate fraction=`3.12%`；`MEASUREMENT_NOISE_DRIVEN=FALSE`。
15–17. 与 coordination / local geometry / final-generated-site MatterSim force 的 validation Spearman：

- atomic_number: rho=-0.2857, structure-cluster bootstrap 95% CI [-0.5117, -0.0471]
- coordination_number: rho=-0.0686, structure-cluster bootstrap 95% CI [-0.2962, 0.1452]
- nearest_neighbor_distance_Angstrom: rho=-0.1989, structure-cluster bootstrap 95% CI [-0.4796, 0.1162]
- local_neighbor_count_3A: rho=0.3670, structure-cluster bootstrap 95% CI [0.1550, 0.5825]
- final_unrelaxed_force_Angstrom_eV: rho=0.1568, structure-cluster bootstrap 95% CI [-0.0929, 0.4166]

18. **SITE_LEVEL_HETEROGENEITY=FALSE**。
19. **SITE_LEVEL_DIAGNOSTIC=FAIL**。
20. **NEXT=STOP_ASYNC_FAMILY**。

结论只针对当前 official Full FT、target=0.2、CFG=2 和冻结定义。MatterSim force 是最终生成但松弛前结构的 surrogate site force；`SURROGATE_PROPERTY_EVAL=True`，`DFT_VERIFIED=False`。
