# Scout-Matter code map and selective migration boundary

The public fork modifies MatterGen sampling rather than training the backbone.

| Concern | Scout-Matter behavior | RP-QTFG phase-0 decision |
|---|---|---|
| Constraint registry | Loss registry in diffusion-loss/sampling path | New isolated RP-QTFG module; no replacement of MatterGen losses |
| Clean estimate | Continuous `x0_hat` from score, alpha, sigma | Reimplement against local `MultiCorruption` APIs and test per field |
| Forward guidance | Gradient through noisy state | Not enabled in first MVP |
| Backward guidance | Gradient on `x0_hat`, mapped into score update | Used only for position and weak cell fields |
| Gradient scale | Optional normalized gradient | Normalize position and cell independently |
| Time profile | Constant or scheduled | Only progress 0.60/0.75 starts |
| Self-recurrence | Supported and commonly configured | Disabled; it changes cost and RNG trajectory |
| Atomic field | Generic framework can expose full state | Frozen to A0; never receives physics gradient |
| Energy guide | MatterSim in public README | CHGNet guide; MatterSim reserved for evaluation |
| Failure handling | General sampling exceptions | Explicit field rejection, backtracking, exact A0 fallback |

No Scout-Matter source file is copied wholesale. Only equations and narrow
control-flow ideas that are compatible with the local A0 sampler are migrated.
