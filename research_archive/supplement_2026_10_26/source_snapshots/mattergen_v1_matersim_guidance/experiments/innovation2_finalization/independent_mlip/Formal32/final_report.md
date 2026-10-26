# Independent CHGNet force evaluation

{
  "baseline": "C0",
  "method": "F0",
  "n": 32,
  "baseline_mean": 0.16486474328048553,
  "method_mean": 0.15765154594668918,
  "mean_relative_reduction": 0.043752212815595715,
  "relative_bootstrap_ci95": [
    0.001325321829258572,
    0.09803638578161744
  ],
  "absolute_improvement": 0.007213197333796309,
  "absolute_bootstrap_ci95": [
    9.89039539818481e-05,
    0.01627860729563593
  ],
  "wins": 19,
  "ties": 0,
  "losses": 13,
  "status": "PASS",
  "ci_excludes_zero_positive": true,
  "bootstrap_resamples": 20000,
  "formal256_evaluation_trigger": true,
  "cohort": "Formal32",
  "elements_Z": [
    8,
    26,
    27,
    46,
    63,
    64
  ],
  "coverage_fraction": 1.0,
  "elapsed_seconds": 6.265058413031511,
  "DFT_VERIFIED": false
}

This evaluator is independent of guidance, but previously supplied magnetic guardrails. Positive central effect is not statistical confirmation unless the interval excludes zero. DFT_VERIFIED=False.
