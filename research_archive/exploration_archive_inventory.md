# Exploration Archive Inventory

Inventory date: 2026-09-18

Experiments found: **81**.

## Category counts

- `FULL_ARCHIVE`: 21
- `SUMMARY_ONLY`: 56
- `EXTERNAL_DATA_ONLY`: 3
- `IGNORE`: 1

## Status counts

- `ABANDONED_BEFORE_EVALUATION`: 5
- `ENGINEERING_ONLY`: 7
- `FAIL`: 18
- `INCONCLUSIVE`: 11
- `MIXED`: 11
- `NOT_SUPPORTED`: 9
- `SUPPORTED`: 20

## Manual review required

- `X-01` ALM-GEN P0 Residual Structures: No defensible scientific verdict can be reconstructed.
- `X-02` Composable Property Adapter Residual Artifacts: The surviving evidence is insufficient for a scientific verdict.
- `X-03` MatterGen LoRA Residual Artifacts: No recoverable confirmatory conclusion.
- `X-04` MatterGen RL P0: The training run completed, but the remaining evidence is not a confirmatory method result.

## Full table

The machine-readable source of truth is [experiment_inventory.csv](experiment_inventory.csv).

| ID | Name | Category | Status | Source | Covered by final release |
| -- | ---- | -------- | ------ | ------ | ------------------------ |
| I1-00 | [Fixed CFG baseline](innovation1/fixed_cfg/README.md) | SUMMARY_ONLY | SUPPORTED | release/thesis-final-2026 @ 5b572c61c70c | yes/core |
| I1-01 | [Adaptive CFG V1](innovation1/adaptive_cfg_v1/README.md) | FULL_ARCHIVE | MIXED | archive/thesis-analysis-package-v1 @ a367f85efe80 | yes/summary |
| I1-02 | [Robust Adaptive CFG V2](innovation1/robust_v2/README.md) | FULL_ARCHIVE | FAIL | experiment/robust-adaptive-cfg-v2 @ 1af301c5e534 | yes/key-negative |
| I1-03 | [Counterfactual Oracle V3](innovation1/oracle_v3/README.md) | FULL_ARCHIVE | MIXED | experiment/counterfactual-adaptive-cfg-v3 @ 1af301c5e534 | yes/key-negative |
| I1-04 | [Risk-Calibrated CFG V4](innovation1/risk_v4/README.md) | FULL_ARCHIVE | FAIL | experiment/risk-calibrated-adaptive-cfg @ a4fda8a573ee | yes/key-negative |
| I1-05 | [Safe-Selection CFG V5](innovation1/safe_v5/README.md) | FULL_ARCHIVE | FAIL | experiment/calibrated-safe-selection-cfg @ 7c9d9261d506 | yes/key-negative |
| I1-06 | [Stage-Calibrated CFG](innovation1/stage_cfg/README.md) | FULL_ARCHIVE | MIXED | experiment/stage-calibrated-bounded-cfg @ f281462d8f69 | yes/key-negative |
| I1-07 | [Field-Decoupled CFG](innovation1/field_cfg/README.md) | FULL_ARCHIVE | FAIL | experiment/field-decoupled-adaptive-cfg @ fca6b8d90e86 | yes/key-negative |
| I1-08 | [Adaptive Neighborhood P0](innovation1/adaptive_neighborhood/README.md) | SUMMARY_ONLY | FAIL | experiment/adaptive-neighborhood-p0 @ 1af301c5e534 | no |
| I1-09 | [Field Async Freeze](innovation1/field_async_freeze/README.md) | SUMMARY_ONLY | ABANDONED_BEFORE_EVALUATION | experiment/field-async-freeze-p0 @ 1af301c5e534 | no |
| I1-10 | [Shared Field Dynamics Diagnostic](innovation1/shared_field_dynamics/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/field-async-freeze-p0 @ 1af301c5e534 | no |
| I1-11 | [Site-Level Convergence Diagnostic](innovation1/site_convergence/README.md) | SUMMARY_ONLY | FAIL | experiment/field-async-freeze-p0 @ 1af301c5e534 | no |
| I1-12 | [Cell-Atom Fusion P0](innovation1/cell_atom_fusion/README.md) | SUMMARY_ONLY | FAIL | experiment/cell-atom-fusion-p0 @ 1af301c5e534 | no |
| I1-13 | [Field-Time Schedule P0](innovation1/field_time_schedule/README.md) | SUMMARY_ONLY | ABANDONED_BEFORE_EVALUATION | experiment/field-time-schedule-p0 @ 1af301c5e534 | no |
| I1-14 | [FP-PC P0](innovation1/fp_pc/README.md) | SUMMARY_ONLY | ABANDONED_BEFORE_EVALUATION | experiment/fp-pc-p0 @ 1af301c5e534 | no |
| I1-15 | [Multifield Self-Conditioning P0](innovation1/multifield_self_conditioning/README.md) | SUMMARY_ONLY | FAIL | experiment/multifield-self-conditioning-p0 @ 1af301c5e534 | no |
| I1-16 | [Self-Correcting Search P0](innovation1/self_correcting_search/README.md) | SUMMARY_ONLY | ABANDONED_BEFORE_EVALUATION | experiment/self-correcting-search-p0 @ 1af301c5e534 | no |
| I1-17 | [Branch-Compatible Oracle](innovation1/branch_oracle/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/raab-sc-p0 @ 1af301c5e534 | yes/core |
| I1-18 | [Reference-Preserved Budgeted Branching](innovation1/reference_branching/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/raab-sc-p0 @ 1af301c5e534 | yes/core |
| I1-19 | [Phase B Learned Allocator](innovation1/phase_b_allocator/README.md) | FULL_ARCHIVE | NOT_SUPPORTED | experiment/raab-sc-p0 @ 1af301c5e534 | yes/core-negative |
| I1-20 | [Frozen C1 Confirmation](innovation1/frozen_confirmation/README.md) | FULL_ARCHIVE | MIXED | experiment/frozen-linear-k2-confirmation @ 1af301c5e534 | yes/core |
| I1-21 | [Final Fixed-K2 Method](innovation1/fixed_k2_final/README.md) | SUMMARY_ONLY | SUPPORTED | release/thesis-final-2026 @ 5b572c61c70c | yes/core |
| I2-00 | [Physics-Guidance Route Selection](innovation2/route_selection/README.md) | SUMMARY_ONLY | ENGINEERING_ONLY | experiment/raab-sc-p0 @ 1af301c5e534 | no |
| I2-01 | [Stage Reliability Study](innovation2/stage_reliability/README.md) | SUMMARY_ONLY | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/summary |
| I2-02 | [Clean-x0 Force Evaluation](innovation2/clean_x0/README.md) | SUMMARY_ONLY | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/summary |
| I2-03 | [Force-Coordinate Mapping](innovation2/force_coordinate_mapping/README.md) | SUMMARY_ONLY | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/summary |
| I2-04 | [Early Closed-Loop Force Guidance](innovation2/early_force_guidance/README.md) | FULL_ARCHIVE | FAIL | experiment/closed-loop-force-guidance-p0 @ 1af301c5e534 | no |
| I2-05 | [RC-NFGD P0](innovation2/p0/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/core |
| I2-06 | [RC-NFGD Formal32](innovation2/formal32/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/core |
| I2-07 | [RC-NFGD Formal256](innovation2/formal256/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/core |
| I2-08 | [Independent CHGNet Validation](innovation2/chgnet_validation/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/raab-sc-p0 @ 1af301c5e534 | yes/core |
| I2-09 | [Force-Direction Ablation](innovation2/direction_ablation/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/core |
| I2-10 | [Trust-Region Ablation](innovation2/trust_region_ablation/README.md) | FULL_ARCHIVE | NOT_SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/key-negative |
| I2-11 | [Equal-Budget Post-Generation Comparison](innovation2/equal_budget_post/README.md) | FULL_ARCHIVE | SUPPORTED | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/core |
| I2-12 | [Adaptive-CFG Compatibility](innovation2/compatibility/README.md) | FULL_ARCHIVE | FAIL | experiment/mattersim-late-force-guidance-p0 @ 1af301c5e534 | yes/key-negative |
| I2-13 | [Bounded-Correction Ablation](innovation2/bounded_correction/README.md) | FULL_ARCHIVE | NOT_SUPPORTED | experiment/raab-sc-p0 @ 1af301c5e534 | yes/key-negative |
| I2-14 | [E3G Baseline Audit](innovation2/e3g_baseline_audit/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/raab-sc-p0 @ 1af301c5e534 | yes/summary |
| I2-15 | [DFT Environment Verification](innovation2/dft_verification/README.md) | SUMMARY_ONLY | ABANDONED_BEFORE_EVALUATION | experiment/raab-sc-p0 @ 1af301c5e534 | yes/core-limit |
| I2-16 | [Final RC-NFGD Interpretation](innovation2/final_rc_nfgd/README.md) | SUMMARY_ONLY | SUPPORTED | release/thesis-final-2026 @ 5b572c61c70c | yes/core |
| O-01 | [Corrector Residual Distillation V1](other_explorations/corrector_distillation/v1/README.md) | SUMMARY_ONLY | MIXED | experiment/corrector-residual-distillation-v1 @ d2736e01bf7e | no |
| O-02 | [Corrector Residual Distillation V2](other_explorations/corrector_distillation/v2/README.md) | SUMMARY_ONLY | NOT_SUPPORTED | experiment/corrector-residual-distillation-v2 @ 1b8b62284f69 | no |
| O-03 | [Corrector Residual Distillation V3 Anchor](other_explorations/corrector_distillation/v3_anchor/README.md) | SUMMARY_ONLY | FAIL | experiment/corrector-residual-distillation-v3-anchor @ f983e1285de0 | no |
| O-04 | [Corrector Distillation Formal256](other_explorations/corrector_distillation/formal256/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/corrector-residual-distillation-formal256 @ 0a1a8e2631ec | no |
| O-05 | [Cross-Field Interaction P0](other_explorations/cross_field_interaction/p0/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/cross-field-interaction-p0 @ 7bc228247c25 | no |
| O-06 | [Cross-Field Interaction P1](other_explorations/cross_field_interaction/p1/README.md) | SUMMARY_ONLY | FAIL | experiment/cross-field-interaction-p1 @ 70a85b70b2b0 | no |
| O-07 | [Distribution-Constrained Quality Adapter P0](other_explorations/distribution_adapter/p0/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/distribution-constrained-quality-adapter-p0 @ 399488be3352 | no |
| O-08 | [Distribution-Constrained Quality Adapter P1](other_explorations/distribution_adapter/p1/README.md) | SUMMARY_ONLY | MIXED | experiment/distribution-constrained-quality-adapter-p1 @ 180acb6005fa | no |
| O-09 | [Distribution Adapter P1b](other_explorations/distribution_adapter/p1b/README.md) | SUMMARY_ONLY | FAIL | experiment/distribution-constrained-quality-adapter-p1b @ 71f4619f9c55 | no |
| O-10 | [Distribution-Balanced Quality Adapter](other_explorations/distribution_adapter/balanced/README.md) | SUMMARY_ONLY | FAIL | experiment/distribution-balanced-quality-adapter @ d8f8014404df | no |
| O-11 | [Global Transformer Adapter P0](other_explorations/global_transformer/README.md) | SUMMARY_ONLY | FAIL | experiment/global-transformer-adapter-p0 @ 499dba7df79d | no |
| O-12 | [TCL and DML P0](other_explorations/tcl/dml_p0/README.md) | SUMMARY_ONLY | MIXED | experiment/tcl-dml-p0 @ 2f95aa0c361c | no |
| O-13 | [TCL P1](other_explorations/tcl/p1/README.md) | SUMMARY_ONLY | SUPPORTED | experiment/tcl-p1 @ 56efbaaf7f00 | no |
| O-14 | [TCL P2](other_explorations/tcl/p2/README.md) | SUMMARY_ONLY | SUPPORTED | experiment/tcl-p2 @ b9fd99019657 | no |
| O-15 | [TCL Formal256](other_explorations/tcl/formal256/README.md) | SUMMARY_ONLY | FAIL | experiment/tcl-formal256 @ 6ed463cfcba4 | no |
| O-16 | [TCL Formal Root-Cause Analysis](other_explorations/tcl/root_cause/README.md) | SUMMARY_ONLY | SUPPORTED | analysis/tcl-formal256-root-cause @ eaf269258ae7 | no |
| O-17 | [GBSA-TCL P0](other_explorations/gbsa/p0/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/gbsa-tcl-p0 @ 5a72cfde92be | no |
| O-18 | [GBSA-TCL P1](other_explorations/gbsa/p1/README.md) | SUMMARY_ONLY | FAIL | experiment/gbsa-tcl-p1 @ b12985130a7e | no |
| O-19 | [Frozen GBSA Prospective32](other_explorations/gbsa/prospective32/README.md) | SUMMARY_ONLY | FAIL | experiment/gbsa-tcl-prospective32 @ 1af301c5e534 | no |
| O-20 | [Metric Framework Reassessment](other_explorations/metric_reassessment/README.md) | SUMMARY_ONLY | ENGINEERING_ONLY | release/thesis-final-2026 @ 5b572c61c70c | yes/summary |
| R-01 | [A0-E3G Independent64](other_explorations/remote_branch_studies/a0_e3g_independent64/README.md) | SUMMARY_ONLY | SUPPORTED | origin/feature/a0-e3g-independent64 @ 3bafb64a7a2c | no |
| R-02 | [A0-E3G Compatibility64](other_explorations/remote_branch_studies/a0_e3g_compatibility64/README.md) | SUMMARY_ONLY | SUPPORTED | origin/feature/a0-e3g-compatibility64 @ 447fa7ea61bd | no |
| R-03 | [A0-E3G Formal256 Eligibility](other_explorations/remote_branch_studies/a0_e3g_formal256/README.md) | SUMMARY_ONLY | INCONCLUSIVE | origin/feature/a0-e3g-formal256 @ 33f8a69136fb | no |
| R-04 | [A0-E3G Leakage Diagnostic256](other_explorations/remote_branch_studies/a0_e3g_leakage256/README.md) | SUMMARY_ONLY | MIXED | origin/experiment/a0-e3g-leakage-diagnostic256 @ 4efe6d5a690f | no |
| R-05 | [Budget-Aware Corrector Gating](other_explorations/remote_branch_studies/budget_aware_gating/README.md) | SUMMARY_ONLY | NOT_SUPPORTED | origin/feature/budget-aware-corrector-gating @ 7561f4e2a9b6 | no |
| R-06 | [Convergence-Aware Corrector Gating](other_explorations/remote_branch_studies/convergence_aware_gating/README.md) | SUMMARY_ONLY | ENGINEERING_ONLY | origin/feature/convergence-aware-corrector-gating @ 9feb77f5adef | no |
| R-07 | [CG-TDR](other_explorations/remote_branch_studies/cg_tdr/README.md) | SUMMARY_ONLY | NOT_SUPPORTED | origin/feature/cg-tdr @ cb67f1e08786 | no |
| R-08 | [CrystalREPA Reproduction](other_explorations/remote_branch_studies/crystalrepa/README.md) | SUMMARY_ONLY | NOT_SUPPORTED | origin/feature/crystalrepa-repro @ 5411a7d3bc6e | no |
| R-09 | [FN-PRA](other_explorations/remote_branch_studies/fn_pra/README.md) | SUMMARY_ONLY | NOT_SUPPORTED | origin/feature/fn-pra @ a620523ba8c0 | no |
| R-10 | [GemNet Fused Inference Fastgate](other_explorations/remote_branch_studies/gemnet_fused/README.md) | SUMMARY_ONLY | ENGINEERING_ONLY | origin/feature/gemnet-fused-inference-fastgate @ 83b468fd143f | no |
| R-11 | [MPS Runtime Fastgate](other_explorations/remote_branch_studies/mps_runtime/README.md) | SUMMARY_ONLY | ENGINEERING_ONLY | origin/feature/mps-runtime-fastgate @ e2aef29bdfe7 | no |
| R-12 | [Post-Generation Quality Modules Fastgate](other_explorations/remote_branch_studies/postgen_modules/README.md) | SUMMARY_ONLY | MIXED | origin/feature/postgen-quality-modules-fastgate @ 9aa23f94e1b5 | no |
| R-13 | [Q3 E3-PCR Frozen64](other_explorations/remote_branch_studies/q3_e3_pcr_frozen64/README.md) | SUMMARY_ONLY | MIXED | origin/feature/q3-e3-pcr-frozen64 @ 87853b030ff7 | no |
| R-14 | [Q3 E3-PCR Formal256](other_explorations/remote_branch_studies/q3_e3_pcr_formal256/README.md) | SUMMARY_ONLY | SUPPORTED | origin/feature/q3-e3-pcr-formal256 @ d91598b4b5bd | no |
| R-15 | [RP-QTFG](other_explorations/remote_branch_studies/rp_qtfg/README.md) | SUMMARY_ONLY | NOT_SUPPORTED | origin/feature/rp-qtfg @ f413e4c7b8e1 | no |
| R-16 | [SPG MatterGen Fastgate](other_explorations/remote_branch_studies/spg_fastgate/README.md) | SUMMARY_ONLY | MIXED | origin/feature/spg-mattergen-fastgate @ a6718d2c0a0b | no |
| R-17 | [SPG Static Periodic Graph MVP](other_explorations/remote_branch_studies/spg_static_mvp/README.md) | SUMMARY_ONLY | ENGINEERING_ONLY | origin/feature/spg-static-periodic-graph-mvp @ 425bfdc8cab0 | no |
| X-01 | [ALM-GEN P0 Residual Structures](other_explorations/alm_gen_p0/README.md) | EXTERNAL_DATA_ONLY | INCONCLUSIVE | experiment/alm-gen-p0 @ 1af301c5e534 | no |
| X-02 | [Composable Property Adapter Residual Artifacts](other_explorations/composable_property_adapter/README.md) | EXTERNAL_DATA_ONLY | INCONCLUSIVE | experiment/composable-property-adapter-p0 @ 1af301c5e534 | no |
| X-03 | [MatterGen LoRA Residual Artifacts](other_explorations/mattergen_lora/README.md) | EXTERNAL_DATA_ONLY | INCONCLUSIVE | experiment/mattergen-lora-p0 @ 1af301c5e534 | no |
| X-04 | [MatterGen RL P0](other_explorations/mattergen_rl/README.md) | SUMMARY_ONLY | INCONCLUSIVE | experiment/mattergen-rl-p0 @ 1af301c5e534 | no |
| X-05 | [Final Thesis Results Build Package](ignored/final_thesis_results_duplicate/README.md) | IGNORE | ENGINEERING_ONLY | experiment/raab-sc-p0 @ 1af301c5e534 | yes/full |
