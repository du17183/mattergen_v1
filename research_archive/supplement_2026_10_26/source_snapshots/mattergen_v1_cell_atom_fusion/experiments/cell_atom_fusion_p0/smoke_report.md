# Global Cell--Atom Bidirectional Fusion: 16-seed Smoke P0

## Result

`CELL_ATOM_FUSION_P0=FAIL`; `NEXT=STOP`.

This is a paired, fresh-seed smoke over seeds 110000--110015. B0 is the frozen official Full FT model; C0 changes only the trained fusion module. Both use target 0.2, constant CFG 2, 1000 diffusion steps, one corrector step, and Adaptive CFG off.

## Aggregate metrics

| method   |   n |   mag_absolute_error_mean |   rmsd_a_mean |   atomic_force_mean_ev_a_mean |   max_force_ev_a_mean |   e_hull_ev_atom_mean |   runtime_seconds_mean |   peak_vram_mb_mean |   score_calls_mean |   mag_hit_rate |   valid_rate |   stable_rate |   novel_rate |   unique_rate |   nus_rate |   mag_density_mean |   relaxation_steps_mean |
|:---------|----:|--------------------------:|--------------:|------------------------------:|----------------------:|----------------------:|-----------------------:|--------------------:|-------------------:|---------------:|-------------:|--------------:|-------------:|--------------:|-----------:|-------------------:|------------------------:|
| B0       |  16 |                  0.009398 |      0.029899 |                      0.057635 |              0.106192 |              0.073566 |              83.500544 |          277.915771 |        2000.000000 |       0.750000 |     1.000000 |      0.750000 |     0.375000 |      0.812500 |   0.125000 |           0.196789 |               24.375000 |
| C0       |  16 |                  0.007051 |      0.044181 |                      0.049602 |              0.093132 |              0.041325 |              87.820703 |          278.955017 |        2000.000000 |       0.687500 |     1.000000 |      0.875000 |     0.187500 |      0.750000 |   0.062500 |           0.201337 |               25.062500 |

Lower-is-better relative improvements (positive favors C0):

```json
{
  "atomic_force_mean_ev_a": 0.13938708309828604,
  "e_hull_ev_atom": 0.4382522665891109,
  "mag_absolute_error": 0.24975739338132102,
  "max_force_ev_a": 0.12298562541581996,
  "peak_vram_mb": -0.00373942651731508,
  "rmsd_a": -0.4776792753697638,
  "runtime_seconds": -0.051738097664606825,
  "score_calls": 0.0
}
```

Rate changes are absolute fractions (0.05 = 5 percentage points):

```json
{
  "mag_hit": -0.0625,
  "novel": -0.1875,
  "nus": -0.0625,
  "stable": 0.125,
  "unique": -0.0625,
  "valid": 0.0
}
```

## Gates

Fail-fast checks: `{"e_hull_increase_gt_0p02": false, "mag_mae_worsening_gt_15pct": false, "max_force_worsening_gt_20pct": false, "nus_drop_gt_10pp": false, "validity_drop_gt_10pp": false}`.

Expansion signals: `{"geometry_max_force_improvement_ge_8pct": true, "geometry_rmsd_improvement_ge_8pct": false, "property_mag_mae_improvement_ge_8pct": true, "quality_nus_gain_ge_5pp": false}`.

Formal32 eligibility additionally requires a validation GO signal and a used fusion mechanism; result: `False`.

## Scope

Magnetic density is evaluated by the frozen CHGNet 0.3.0 surrogate (held-out MAE 0.001449, Pearson 0.9499, hit tolerance 0.01). `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`. Energy/force/relaxation metrics use MatterSim-5M. Paired bootstrap uses 20,000 resamples and is descriptive at n=16.
