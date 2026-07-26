# CrystalREPA reproduction test report

- Targeted CrystalREPA tests: **4 passed, 0 failed**, including the SMACT
  missing-oxidation-state boundary.
- Eight-rank DDP EA-NCE all-gather/mapping test: **passed**.
- Strict U0 load, block-2 hook, atom mapping, masking, finite loss/gradients,
  save/resume, and inference Teacher exclusion: **passed**. Same-seed inference
  determinism is scheduled in the 8-seed gate.
- Full repository suite has five unrelated baseline failures: four legacy
  Corrector tests omit the now-required `dt` argument, and one scheduler test
  uses the `ReduceLROnPlateau(verbose=...)` argument removed by PyTorch 2.7.
  No CrystalREPA-targeted regression failed.

`git diff --check` is rerun on the final publishable tree.
