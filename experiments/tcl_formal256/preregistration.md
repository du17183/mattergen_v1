# TCL Formal256 preregistration

Frozen before any Formal256 outcome was generated or inspected.

- Branch start: `b9fd99019657df7c605976255ef07d774b61f4f9`
- Formal seeds: `83000–83255`, 256 paired seeds, historical overlap `none`
- Methods: frozen C0, FT0 and TCL from P2
- Formal training steps: `0`
- Generation and MatterSim/quality pipelines: unchanged from P1/P2
- MatterSim-5M surrogate evaluation, `fmax=0.05 eV/Å`
- `DFT_VERIFIED=False`
- Bootstrap: 20,000 paired resamples, frozen RNG seed `20260912`

## Question A: TCL superiority over FT0

The five primary endpoints are frozen as:

1. E-hull mean, lower is better.
2. NUS proportion, higher is better.
3. RMSD mean, lower is better.
4. Atomic-force mean, lower is better.
5. Structure maximum-force mean, lower is better.

Each endpoint receives a percentile paired-bootstrap 95% CI and a one-sided,
null-centered paired-bootstrap p-value. The five one-sided p-values form one
Holm family at alpha 0.05. Endpoints and correction method will not be changed
after outcomes are observed.

Secondary endpoints are Stable, Novel, Unique, maximum-force P95/P99 and
relaxation. Formal PASS does not require all five primary endpoints to reject
their null; it requires a majority in the favorable direction and multiple
raw/Holm-supported improvements, together with the C0 and robustness criteria.

## Question B: TCL non-inferiority to C0

These margins were frozen before inspecting the Formal256 outcomes. They are
study-specific engineering/scientific tolerances, not universal materials
standards.

| Endpoint | Frozen NI rule |
|---|---|
| E-hull mean delta | upper 95% CI < +0.025 eV/atom |
| Stable delta | lower 95% CI > -10 percentage points |
| NUS delta | lower 95% CI > -10 percentage points |
| RMSD mean delta | upper 95% CI < +0.020 Å |
| Atomic-force mean ratio | upper 95% CI < 1.20 |
| Maximum-force mean ratio | upper 95% CI < 1.20 |
| Maximum-force P95 ratio | upper 95% CI < 1.30 |
| Maximum-force P99 ratio | upper 95% CI < 1.50 |
| Force >1 eV/Å delta | upper 95% CI < +5 percentage points |
| Force >2 eV/Å delta | upper 95% CI < +2 percentage points |

`NOT ESTABLISHED` is not interpreted as proven inferiority. A CI crossing zero
is not interpreted as equivalence.

## Frozen tail rules

All seeds are retained. Severe outliers use the existing P1/P2 thresholds:
RMSD >0.5 Å, structure maximum force >1 or >2 eV/Å, and relaxation >200,
>300 or >400 steps. Tail frequency and magnitude are interpreted jointly;
one extreme sample alone is not an automatic failure.

## Frozen final decision

- `PASS`: the P1/P2 TCL-over-FT0 pattern is again broadly reproduced, with a
  majority of primary endpoints favorable and multiple corrected/raw results
  supporting superiority; core C0 NI is broadly established; Novel/Unique and
  RMSD/force tails do not collapse.
- `BORDERLINE`: TCL remains clearly better than FT0, but a small number of C0
  NI endpoints are not established because of precision, without credible
  systematic degradation.
- `FAIL`: the TCL-over-FT0 pattern broadly disappears, or TCL shows credible
  systematic degradation against C0, or mode/geometry/force-tail collapse.

No DML, TCL+DML, tuning, threshold changes, outlier deletion or replacement
sampling is permitted in this experiment.
