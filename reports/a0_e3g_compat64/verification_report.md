# A0 + E3-G compatibility verification

## Frozen experiment checks

- A0 generation: 64/64 successful, every seed attempted exactly once.
- Learned-gated E3-PCR refinement: 64/64 successful.
- MatterSim relaxation: A0 64/64 and A0+E3-G 64/64.
- Focused compatibility tests: 21/21 passed.
- Python compilation, shell syntax, `git diff --check`, archive equality, and
  final-decision assertions passed.

## Repository-wide test audit

The unmodified repository-wide suite produced 183 passes and 11 failures.
Re-running with the trusted-local-fixture setting
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` resolved six PyTorch 2.6 fixture-loading
failures. Five inherited compatibility failures remain:

- Four sampling tests omit the now-required `dt` keyword argument.
- One training smoke test passes the removed `verbose` argument to
  `torch.optim.lr_scheduler.ReduceLROnPlateau`.

The implementation and test/config files involved in these five failures are
byte-for-byte unchanged from compatibility base commit
`0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`. They were not modified because
the compatibility study freezes A0, Predictor/Corrector, and training
configuration.

## Result

`A0_E3G_COMPATIBILITY_GO`

The primary endpoint and every frozen quality/safety gate passed.
