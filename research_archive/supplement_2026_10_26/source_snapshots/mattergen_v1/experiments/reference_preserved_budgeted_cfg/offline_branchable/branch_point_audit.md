# Branch-point and compatibility audit

The frozen policy configuration sets `pulse_start=400`, `pulse_duration=100`, and `N=1000`. Sampling indices run from 0 at the high-noise `max_t` end to 999 at `eps_t`; the sampler uses `torch.linspace(max_t, eps_t, N)`. Thus branch point 400 is after 400 reverse predictor/corrector updates (40% of the update sequence), leaving a 60% suffix.

For pulse policies, the implementation first advances the shared base with field scales `(2.0, 2.0, 2.0)`. At index 400 it snapshots the RNG and hashes the base tensor, restores that RNG for every branch, clones the complete base state, applies the policy for steps 400–499, and resumes CFG2 for steps 500–999. The C0 base is advanced from the same pre-branch state and RNG snapshot.

- Static implementation audit: `PASS`
- Pulse prefix digest equality: `PASS` across 48 seeds and 4 policies.
- State tensors at every prefix step were not serialized. The strongest available audit is the frozen schedule/config, code path, and common step-400 tensor digest.
- Constant adaptive policies start changing at step 0 and are not branch-compatible with a step-400 C0 prefix.

Config SHA-256: `e502087d1e87549b049c505804dea913c3262ad41f917d7b2a8d026291a2d9d3`  
Implementation SHA-256: `a9be95d598aa05c2563fe1900cffe28cdbebce5db9018352c7b958f3e92595a2`
