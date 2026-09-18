# Innovation 2 method sources

`late_force_sampler.py` contains the frozen late-stage clean-x0 force guidance
implementation. The finalization sources preserve the force-direction,
bounded-update, independent-surrogate and equal-budget analyses.

RC-NFGD is online guidance during reverse diffusion. Post-generation relaxation
is retained only as a baseline and is not part of the RC-NFGD method.

These files are exact archival sources. Portable table and figure reproduction
uses only the compact result files in this release and does not require model
weights or GPU sampling.
