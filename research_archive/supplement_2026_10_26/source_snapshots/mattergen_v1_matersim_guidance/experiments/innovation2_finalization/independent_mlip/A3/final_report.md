# Independent CHGNet force evaluation

{
  "baseline": "G0",
  "method": "G1",
  "n": 32,
  "baseline_mean": 0.12862659032737078,
  "method_mean": 0.11833101336207465,
  "mean_relative_reduction": 0.08004236868203225,
  "relative_bootstrap_ci95": [
    0.01643113346090434,
    0.13601545588130082
  ],
  "absolute_improvement": 0.010295576965296135,
  "absolute_bootstrap_ci95": [
    0.002044636337982118,
    0.018415200284925607
  ],
  "wins": 25,
  "ties": 0,
  "losses": 7,
  "status": "PASS",
  "ci_excludes_zero_positive": true,
  "bootstrap_resamples": 20000,
  "formal256_evaluation_trigger": true,
  "cohort": "A3",
  "elements_Z": [
    1,
    7,
    8,
    25,
    26,
    27,
    28,
    63,
    64
  ],
  "coverage_fraction": 1.0,
  "elapsed_seconds": 2.026722051028628,
  "DFT_VERIFIED": false
}

This evaluator is independent of guidance, but previously supplied magnetic guardrails. Positive central effect is not statistical confirmation unless the interval excludes zero. DFT_VERIFIED=False.
