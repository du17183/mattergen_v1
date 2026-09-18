# Innovation 2: Reliability-Calibrated Late-stage Neural Force Guidance

## Final method

**Reliability-Calibrated Late-stage Neural Force-Guided Diffusion (RC-NFGD)**
is the frozen online F0 method. At `t <= 0.02`, it extracts predicted clean x0,
evaluates MatterSim force, removes translation, bounds a Cartesian displacement,
maps it to periodic fractional coordinates, changes only the position predictor,
and continues reverse diffusion. Atomic and cell scores are unchanged.

Post-generation `POST` relaxation is a separate comparator and is not part of
RC-NFGD.

## Formal evidence

Paired Formal256 results:

| Metric | C0 | RC-NFGD | Relative change | Paired W/T/L |
|---|---:|---:|---:|---:|
| MatterSim MaxF | 0.226408 | 0.158911 | 29.81% lower | 254/0/2 |
| MatterSim mean force | 0.098702 | 0.069373 | 29.72% lower | 255/0/1 |
| RMSD | 0.051137 | 0.043893 | 14.17% lower | 216/21/19 |
| Property MAE | 0.009757 | 0.009868 | 1.14% worse | 95/0/161 |
| Stable | 80.86% | 80.86% | unchanged | 0/256/0 |
| Validity | 100% | 100% | unchanged | 0/256/0 |

Independent CHGNet surrogate evaluation on the same paired n=256 cohort found
14.36% lower MaxF and 9.59% lower mean force. This is directional surrogate
agreement, not DFT validation.

## Ablation boundaries

- Force-direction evidence was supported on its frozen n=32 cohort, with stated
  causal limitations in the archived report.
- Trust-region necessity was not supported; the cap remains a conservative
  design choice.
- Equal-budget post-generation correction was stronger than online F0 on the
  tested n=32 cohort.
- The Adaptive-CFG + force-guidance compatibility track failed; no synergy is
  claimed.
- `DFT_VERIFIED = false`.

## Directory map

- `method/`: frozen sampler, logging, evaluation and ablation switches.
- `configs/`, `seeds/`: P0/Formal32/Formal256 locks and registered seeds.
- `results/`: paired formal, CHGNet and ablation evidence.
- `tables/`, `figures/`: publication-ready derived outputs.
