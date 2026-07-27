# CG-TDR full-test baseline comparison

Both revisions were tested with the same Python environment, cache, environment variables, and exact command:

```text
/data/dxl/envs/mattergen_py310/bin/python -m pytest -q
```

| Revision | Commit | Passed | Failed | Warnings | Time |
|---|---|---:|---:|---:|---:|
| `main` | `9bc6747a3ddfd26db6d931bcdb6df5d299844544` | 148 | 11 | 136 | 31.18s |
| `feature/cg-tdr` | `81796fd53a40f2916f256b97f054f8554284b4bb` | 155 | 11 | 136 | 31.66s |

## Failure attribution

| Node ID | Exception type on both revisions |
|---|---|
| `mattergen/common/tests/data_utils_test.py::test_polar_decomposition` | `_pickle.UnpicklingError` |
| `mattergen/common/tests/gemnet_test.py::test_lattice_score_scale_invariance` | `_pickle.UnpicklingError` |
| `mattergen/common/tests/gemnet_test.py::test_nonconservative_lattice_score_translation_invariance` | `_pickle.UnpicklingError` |
| `mattergen/common/tests/gemnet_test.py::test_lattice_parameterization_invariance` | `_pickle.UnpicklingError` |
| `mattergen/common/tests/gemnet_test.py::test_symmetric_lattice_score` | `_pickle.UnpicklingError` |
| `mattergen/common/tests/gemnet_test.py::test_rotation_invariance` | `_pickle.UnpicklingError` |
| `mattergen/diffusion/tests/test_sampling.py::test_corrector[VPSDE-LangevinCorrector]` | `TypeError` |
| `mattergen/diffusion/tests/test_sampling.py::test_corrector[VESDE-LangevinCorrector]` | `TypeError` |
| `mattergen/diffusion/tests/test_sampling.py::test_corrector[WrappedVPSDE-WrappedLangevinCorrector]` | `TypeError` |
| `mattergen/diffusion/tests/test_sampling.py::test_corrector[WrappedVESDE-WrappedLangevinCorrector]` | `TypeError` |
| `mattergen/tests/test_diffusion_instantiation.py::test_train_on_one_batch[default]` | `TypeError` |

```text
TEST_FAILURE_SET_IDENTICAL=True
PRE_EXISTING_TEST_FAILURES=11
CG_TDR_INTRODUCED_TEST_FAILURES=0
FULL_TEST_BLOCKER_CLEARED=True
```

The feature branch adds 7 passing CG-TDR tests and introduces no failing node or exception-type change. The 11 failures are therefore pre-existing in `main`; this task does not modify unrelated legacy tests.
