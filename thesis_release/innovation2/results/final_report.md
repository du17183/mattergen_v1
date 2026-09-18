# Innovation 2 unified final report

All requested cohorts evaluated and audited: True

| Track | Status | Report |
|---|---|---|
| A1 | STRONG_CONFIRMED | [mattersim_late_force_guidance_formal256](/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_formal256/final_report.md) |
| A3 | SUPPORTED | [mattersim_force_direction_ablation](/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_force_direction_ablation/final_report.md) |
| A4 | NOT_SUPPORTED | [mattersim_trust_region_ablation](/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_trust_region_ablation/final_report.md) |
| B | FAIL | [closed_loop_force_guidance_p0](/mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance/experiments/closed_loop_force_guidance_p0/final_report.md) |
| A5 | FAIL | [adaptive_cfg_force_guidance_compatibility](/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/adaptive_cfg_force_guidance_compatibility/final_report.md) |
| A6 | COMPLETE | [mattersim_equal_budget_post_generation](/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_equal_budget_post_generation/final_report.md) |

Independent Formal256: PASS
Complete F0 evidence chain supported: False
Final model recommendation: F0 standalone; combination not confirmed
Retain F0; no rescue tuning

F0 Formal32 remains frozen. Formal256 governs generalization claims; failed ablations must be reported. CHGNet force agreement is cross-potential evidence, not DFT verification or proof of nonoverlapping training data. The same CHGNet model supplies magnetic guardrails. G4 reverses the paired F0 reference trajectory force directions; it is not closed-loop antigradient descent on its own trajectory. Trust-region necessity is conditional on the selected uncapped comparator. F1 is only 16-pair P0; a GO does not replace F0 without a fresh next-round confirmation. No new neural network, no tuning, no budget-allocation method, and no thesis-body changes. DFT_VERIFIED=False.
