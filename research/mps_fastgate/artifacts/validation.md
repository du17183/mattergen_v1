# Validation

- MPS fast-gate unit tests: 9 passed.
- Repository suite in the installed environment: 157 passed, 11 failed.
- Re-running the failed tests with the project's trusted-load compatibility variable: 6 passed, 5 failed.
- Four remaining failures are existing sampling tests that omit the `dt` argument required by the current main corrector API.
- The fifth is the existing CPU training smoke using `ReduceLROnPlateau(verbose=...)`, which the installed PyTorch no longer accepts.
- This branch changes zero files under `mattergen/`; these compatibility failures are outside the MPS fast-gate scope.
