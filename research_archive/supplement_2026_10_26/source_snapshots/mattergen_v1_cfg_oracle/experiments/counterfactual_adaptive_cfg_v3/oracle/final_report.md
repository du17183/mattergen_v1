# Counterfactual Adaptive CFG V3 — Oracle Diagnostic

## Frozen design

32 fresh calibration-only seeds; CFG candidates 1.75/2.00/2.25; eight pre-registered reverse timesteps; H=15; exactly 25% of states receive full-horizon validation. The scalar objective is normalized CHGNet magnetic-density target error plus a frozen invalid penalty. A 5% deadband selects KEEP.

## Headroom result

```text
ORACLE_HEADROOM = FAIL
NEXT = STOP_ADAPTIVE_CFG_RESEARCH
```

- Short-state Oracle objective improvement: 51.809500%
- Full-outcome Oracle objective improvement: 27.222166%
- Full improvement obtained by short labels: -6.903366%
- Short/full exact action agreement: 62.500000%
- Short action distribution: {'DOWN': 0.17578125, 'KEEP': 0.6328125, 'UP': 0.19140625}
- Full action distribution: {'DOWN': 0.234375, 'KEEP': 0.625, 'UP': 0.140625}
- Full Oracle harm: {'property_harm_rate': 0.0, 'ehull_harm_rate': 0.1875, 'stable_harm_rate': 0.09375, 'validity_harm_rate': 0.0}
- Gate checks: {'short_property_improvement_ge_10pct': True, 'full_property_improvement_ge_10pct': True, 'short_non_keep_ge_20pct': True, 'full_non_keep_ge_20pct': True, 'full_max_action_share_le_80pct': True, 'full_down_share_ge_5pct': True, 'full_up_share_ge_5pct': True, 'short_full_label_agreement_ge_50pct': True, 'short_selected_full_improvement_ge_5pct': False, 'mean_ehull_worsening_le_0.02': True, 'stable_drop_le_5pp': True, 'nus_drop_le_5pp': True, 'validity_drop_le_5pp': True}

## Scientific boundary

This is a diagnostic oracle, not a deployable controller. Candidate selection is retrospective. All property and energy/stability values are frozen surrogate evaluations (`SURROGATE_PROPERTY_EVAL=True`, `DFT_VERIFIED=False`).
