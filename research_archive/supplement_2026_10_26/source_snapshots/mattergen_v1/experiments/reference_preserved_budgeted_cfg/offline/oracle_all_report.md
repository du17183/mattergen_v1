# SAFE-A / SAFE-B Oracle-All report

This is a retrospective offline upper-bound analysis over the retained 48-seed Field-Decoupled cohort. It is not a deployable controller, fresh validation, or DFT verification.

The split was fixed by SHA-256 before subset fitting: 24 train, 12 validation, 12 test. Mixed-batch uniqueness was recomputed from the selected relaxed structures with MatterGen's `DefaultDisorderedStructureMatcher`; NUS was then recomputed as stable AND novel AND mixed-batch unique. Source-group uniqueness labels were not reused.

## Offline-test result

| Selector | C0 Property MAE | Oracle-All Property MAE | Relative improvement | Stable | Novel | Unique | NUS | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFE-A | 0.02879873 | 0.01642374 | 42.97% | 75.00% | 58.33% | 100.00% | 50.00% | 100.00% |
| SAFE-B | 0.02879873 | 0.02010972 | 30.17% | 75.00% | 83.33% | 100.00% | 66.67% | 100.00% |

SAFE-A requires validity and stability not below paired C0, E-hull no more than 0.01 eV/atom above paired C0, and property improvement greater than 1e-06. SAFE-B additionally preserves paired novelty. Uniqueness is deliberately evaluated only after batch selection.
