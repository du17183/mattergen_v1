# FN-PRA Phase-1 test report

- Focused FN-PRA suite: `summary unavailable`.
- Full suite, repository defaults: `summary unavailable`.
- Full suite with PyTorch 2.7 checkpoint compatibility enabled: `summary unavailable`.
- Residual failures after compatibility mode: 6.
- Failures exercising FN-PRA code: 0.

Residual baseline failures comprise four stale corrector tests that omit the
current required `dt` argument, one default MP-20 cache path mismatch, and one
pre-existing RDF numeric expectation mismatch. Training save/resume, 8-GPU DDP,
cache mapping, inference isolation, gradients, and frozen-backbone checks all
passed in the completed Phase-1 runs.
