# Repository upload scope

This repository is a source snapshot based on the Microsoft MatterGen repository:

- Upstream: `https://github.com/microsoft/mattergen.git`
- Upstream commit: `ac9ddd406171138c3f037d06b9b53fedbbb1c536`
- Source branch at export: `feature/stage-adaptive-guidance`

The snapshot includes the current source code, configurations, tests, documentation,
and local guidance changes present in the working tree at export time.

Model weights and checkpoint payloads are intentionally excluded. In particular,
`checkpoints/*/checkpoints/last.ckpt` files are not part of this repository. The
official model assets remain available from the upstream MatterGen distribution.

The small `mattergen/common/tests/mp_20_debug_batch.pt` file is retained because it
is a test fixture, not a trained model weight.
