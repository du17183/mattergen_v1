# Stage 0 historical diagnostic

Historical outcomes are used only to delimit the candidate region. They are not calibration, P0, or Formal evidence.

## Findings

- V1 residual-driven Adaptive CFG was unstable across cohorts; this pre-registered historical conclusion rules out reusing its sample-wise controller.
- V3 full-outcome Oracle reduced mean property MAE by 27.22% on n=64; the short-horizon-selected full outcome instead changed it by -6.90%.
- V4 Oracle relative headroom declined from 48.34% at step 100 to 33.18% at step 550.
- Across V4 permanent +/-0.25 interventions, property harm rates ranged from 34.38% to 57.29%; this supports smaller bounded pulses rather than long-held changes.
- V2 normalized residual calibration has nonzero stage ranges for every field/phase (minimum log-residual range 1.037), confirming timestep dependence without claiming sample-level predictability.
- V5 frozen safe-selection validation status was FAIL; it is used as evidence to stop sample-wise gating, not as evidence for this schedule.

## Frozen candidate-region rationale

- Starts: steps 100, 250, and 400 cover early-middle, middle, and middle-late regions while excluding step 550, where historical Oracle headroom was lowest.
- Amplitude: delta_g in {-0.10, +0.10}. V4 used permanent +/-0.25 changes and showed large tails; halving the magnitude is the conservative stability-first choice.
- Duration: K in {50, 100}. These windows are long enough to affect multiple reverse steps, but force return to CFG=2 far earlier than the permanent comparator.
- This choice is frozen before any new schedule-calibration outcome is generated.

STAGE0_DIAGNOSTIC = COMPLETE
