# Adaptive CFG V1 failure-pattern report

Status: `COMPLETE_NO_NEW_GENERATION`

The audit joins 144,000 saved V1 decisions from 72 historical paired seeds to their final outcomes. Formal256 contributes all 256 paired endpoints but no controller traces because those traces were not archived. No sample was regenerated or excluded in Stage 0.

## Frozen harm definitions

- Harm_Ehull: Adaptive minus Base E-hull > 0.02 eV/atom.
- Harm_Stable: Base Stable=1 and Adaptive Stable=0.
- Harm_Property: Adaptive magnetic-property absolute error > Base error +10%.

## Cohort results

```json
{
  "A5_64": {
    "n": 64,
    "harm_rate_ehull": 0.28125,
    "harm_rate_stable": 0.03125,
    "harm_rate_property": 0.421875,
    "harm_rate_any": 0.515625,
    "ehull_degradation": {
      "mean": -0.029488934512275,
      "median": -0.004304442215332251,
      "p75": 0.02224952935253955,
      "p90": 0.06756979724932068,
      "p95": 0.07566072762299757,
      "max": 0.6733646626171891
    },
    "property_degradation": {
      "mean": -0.0008556410434899437,
      "median": -0.0001260145647397501,
      "p75": 0.002991198902687624,
      "p90": 0.010153301736678333,
      "p95": 0.019443897560260538,
      "max": 0.027139765356880098
    },
    "stable_degradation": {
      "mean": 0.15625,
      "median": 0.0,
      "p75": 0.0,
      "p90": 1.0,
      "p95": 1.0,
      "max": 1.0
    },
    "cfg": {
      "minimum": 1.1570102990006852,
      "maximum": 5.0,
      "mean_max_excursion": 1.927580813929107,
      "mean_extreme_rate": 0.006859375
    }
  },
  "minimal8": {
    "n": 8,
    "harm_rate_ehull": 0.625,
    "harm_rate_stable": 0.375,
    "harm_rate_property": 0.125,
    "harm_rate_any": 0.625,
    "ehull_degradation": {
      "mean": 0.042506228986099,
      "median": 0.051167070786946496,
      "p75": 0.07044324394517472,
      "p90": 0.1004450208699337,
      "p95": 0.1143617940245746,
      "max": 0.1282785671792155
    },
    "property_degradation": {
      "mean": -0.011651117646861225,
      "median": -0.00772247994282515,
      "p75": -0.0016148962895428257,
      "p90": 0.008402788236584646,
      "p95": 0.01793047685711356,
      "max": 0.0274581654776425
    },
    "stable_degradation": {
      "mean": -0.375,
      "median": 0.0,
      "p75": 0.0,
      "p90": 0.0,
      "p95": 0.0,
      "max": 0.0
    },
    "cfg": {
      "minimum": 1.284239527675747,
      "maximum": 5.0,
      "mean_max_excursion": 2.4030217470304054,
      "mean_extreme_rate": 0.009437500000000001
    }
  }
}
```

## Diagnostic answers

1. Extreme CFG: harmful seeds have mean extreme-event rate 0.006736842105263159; non-harm seeds 0.007602941176470588. The corresponding mean maximum excursions are 1.8722945728947757 and 2.1012397734619013.
2. Field disagreement: harmful-event-weighted seed mean 1.5923329827320383; non-harm 1.3880596096926379.
3. Raw residual is timestep-dependent; the field/phase max:min ratios are recorded in `diagnostic_summary.json` and visualized in `field_residual_by_timestep.png`.
4. Cohort drift is present to the degree quantified by absolute shifts in time-binned log residual means in `diagnostic_summary.json`. This motivates frozen-CFG calibration rather than raw cross-field averaging.
5. Predictor and corrector distributions are reported separately; calibration will also be phase-specific.
6. Harm-associated excess CFG excursion by stage is `{'early': 0.0015500375168189529, 'middle': 0.0005327888832185879, 'late': -0.006310305583961839}`. Frozen stage-gating decision: `OFF`.

## V2 design consequence

The once-only V2 uses log-residual timestep calibration, a median across the three standardized fields, confidence fallback to CFG=2, CFG bounds [1.5, 2.5], and a per-score-call slew cap of 0.05. A stage gate is used only if the explicit 60% concentration diagnostic selected one; otherwise it remains off. No learned controller, field-specific CFG, property gradient, or parameter sweep is introduced.

## Limitations

This is retrospective failure diagnosis, not independent validation. The fresh P0 seeds must remain unseen until the V2 configuration is frozen. All stability and property outcomes are surrogate-based (`DFT_VERIFIED=False`).
