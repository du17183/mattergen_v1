# RP-QTFG literature verification

Verified 2026-07-26 against primary papers and official repositories.

## Scout-Matter / adaptive constraint guidance

- Paper: https://arxiv.org/html/2604.13354v3
- Code: https://github.com/link-lab3629/scout-matter
- Base: MatterGen predictor-corrector sampling.
- Clean estimate for continuous variables:
  `z0_hat = (z_t + sigma_t^2 * score(z_t,t)) / alpha_t`.
- Forward guidance differentiates the clean-estimate loss through `z_t`.
- Backward guidance differentiates the clean-estimate loss with respect to
  `z0_hat` and rescales it into score space.
- The implementation supports separate forward/backward weights, gradient
  normalization, time schedules, periodic guidance, and self-recurrence.
- Published case studies use cell volume and smooth coordination constraints.
- The repository also exposes an energy objective backed by MatterSim.
- RP-QTFG will not copy the fork wholesale, will not use self-recurrence in
  phase 0, and will not use MatterSim as the guide because MatterSim is the
  independent evaluator in this project.

## TFG / Universal Guidance

- TFG paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2818054fc6de6dacdda0f142a3475933-Abstract-Conference.html
- Universal Guidance paper: https://arxiv.org/abs/2302.07121
- TFG identifies recurrence, backward guidance iterations, and time-dependent
  guidance strength as separate design axes.
- A clean-data predictor can be evaluated on `x0_hat`; the predictor does not
  need to consume highly noisy states directly.
- RP-QTFG freezes recurrence at zero and limits the phase-0 search to two start
  progresses and two trust radii.

## CFG++

- Paper: https://openreview.net/forum?id=E77uvbOTtp
- CFG++ attributes part of high-scale CFG degradation to off-manifold updates
  and motivates smaller, manifold-constrained guidance.
- RP-QTFG uses this only as motivation for residual-conflict projection and
  trust-region clipping. It does not claim to be a direct CFG++ implementation.

## CHGNet

- Paper: https://www.nature.com/articles/s42256-023-00716-3
- Code: https://github.com/CederGroupHub/chgnet
- Frozen checkpoint: CHGNet 0.3.0.
- Official outputs: energy per atom, forces, stress, and site magnetic moments.
- Package version in the isolated teacher environment is 0.4.2; it loads the
  requested 0.3.0 checkpoint explicitly.
- Site magnetic moments are reported in Bohr magnetons and cell volume is in
  cubic angstroms. The candidate density is therefore
  `sum(abs(site_magmoms))/volume`, in `mu_B/A^3`.
- This candidate signal must pass the held-out MP-20 gate before it may be used
  as guidance. It is never treated as independent DFT proof.
