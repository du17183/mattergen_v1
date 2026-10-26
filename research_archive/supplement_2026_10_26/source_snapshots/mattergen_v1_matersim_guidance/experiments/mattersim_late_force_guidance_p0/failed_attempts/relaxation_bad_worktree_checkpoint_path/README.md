# Retained technical failure: worktree-relative checkpoint path

The first P0 relaxation launcher invocation exited before loading a model or
relaxing any structure. It incorrectly resolved the untracked local MatterSim
checkpoint relative to this independent worktree, where that file does not
exist. Both raw logs and the two empty output directories are retained here.

The only repair was to pass the already frozen absolute checkpoint path from
the main project and verify its recorded SHA256 before relaunching. No
scientific setting, structure, seed, force threshold, or gate was changed.
