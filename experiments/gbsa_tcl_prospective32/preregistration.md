# Frozen GBSA-TCL Prospective32 preregistration

This file is frozen before generation of seeds 87000–87031. The experiment asks
whether the existing GBSA-TCL checkpoint improves NUS versus C0 while meeting
energy non-inferiority and physical guardrails. Historical GBSA P1 remains FAIL.

- Methods: C0 and frozen GBSA-TCL only; 32 paired seeds, 64 generations.
- No training, checkpoint/config/sampler/CFG/condition/loss modification.
- Same target 0.1, constant CFG 2, 1000 diffusion steps, one Corrector step,
  batch size 1 and frozen MatterSim-5M evaluation pipeline.
- NUS CLEAR GO: delta at least +5 pp and paired-bootstrap 95% lower bound > 0.
- E-hull PASS: paired delta CI upper <= +0.010 eV/atom; FAIL only when CI
  lower > +0.010; otherwise BORDERLINE.
- RMSD PASS/FAIL analogously with +0.020 Angstrom.
- Initial atomic-force mean ratio PASS when CI upper <= 1.20, FAIL when CI
  lower > 1.20, otherwise BORDERLINE.
- MaxF>1 and MaxF>2 paired-rate harm margins are +5 pp and +2 pp. Each is
  PASS when CI upper is within the margin, FAIL when CI lower exceeds it,
  otherwise BORDERLINE. A single maximum never automatically decides the arm.
- Stable, Novel and Unique explain NUS. Because the user supplied no numeric
  definition of “clear collapse”, before seeing results it is operationalized as
  a point drop of at least 10 pp AND a paired CI upper bound below 0. A negative
  direction not meeting both conditions is a warning, not automatic collapse.
- Main analyses retain all attempts and seeds. Technical reruns are allowed only
  for identified infrastructure failures and must be recorded. Leave-one-out is
  diagnostic and cannot change the main verdict.
- CLEAR GO requires NUS PASS, E-hull PASS, RMSD PASS, atomic-force PASS, no
  MaxF-tail FAIL, and no Novel/Unique clear collapse. Only CLEAR GO may justify
  a separately authorized independent64. This run never starts it automatically.
- A generated/evaluated structure with non-finite coordinates, cell, energy or
  force, or a non-positive cell determinant, is a technical-invalid FAIL; it is
  retained in the attempt denominator and is never replaced for scientific reasons.
- NUS failure, a confirmed margin failure, or diversity collapse gives FAIL.
  Otherwise an unresolved guardrail gives BORDERLINE and stops for reporting.

Seed scan before branch creation found no strict integer seed-field occurrence for
87000–87031 in the readable current experiment/research tree. A full `git log
--all -G` scan was attempted but encountered a pre-existing unreadable Git object
`f44a68b1ff02ee44a8d7c5b61b57c9a6253892d3`; this limitation is disclosed rather
than silently claiming a perfect repository audit. All explicitly known historical
ranges listed in the task are disjoint. No individual seed substitutions are allowed.

All stability, energy, force and relaxation claims are MatterSim-5M surrogate
results. `DFT_VERIFIED=False`.
