# Frozen Allocator Confirmatory Study

This is an independent prospective confirmation of the directional Phase-B
Linear-K2 signal.  It does not change or reinterpret the historical Phase-B
gate failure.

## Frozen inference

At sampling step 400, 179 leakage-free prebranch statistics are computed.  The
reconstructed and verified Phase-B pipeline scores `GPulse`, `APulse`,
`PPulse`, and `CPulse`; only its Top-2 suffixes are generated for the proposed
method.  Fixed-K2 always uses `GPulse+PPulse`.  Random-K2 uses a deterministic
pre-registered mapping.  Each method receives the exact C0 reference and the
same terminal SAFE-A selector.

The experiment runner may share the physical prefix and exact C0 result across
the three paired methods, but it executes two explicitly method-tagged suffix
branches per method, including duplicate policy branches when allocations
overlap.  It never generates an unallocated policy for a method.  Formal
per-method deployment accounting is therefore 2.2x C0 for Fixed, Random, and
Linear.  A separate standalone C0 replay is produced for exact reference
reproduction; total study acquisition cost is reported separately from method
deployment cost.

Repeated-C0 variance estimation is `NOT_APPLICABLE`: all randomized paired
methods use equal deployment compute and the exact same paired C0 reference.
Instead, determinism is audited more strictly by requiring a standalone full
C0 replay to be bitwise identical to the shared-prefix C0 for every seed.

## Cohorts

All 256 seeds are registered before any confirmatory generation.  C1 is the
first 128 registered seeds and C2 is the remaining 128.  Neither cohort may be
used to fit, normalize, select features, select models, or change thresholds.

## C1 continuation

C2 runs only when Linear has lower MAE than Fixed by at least 1%, is no worse
than Random, and has no catastrophic quality degradation relative to Fixed:
E-hull <= +0.01 eV/atom and Stable/NUS/Validity drops <=5 percentage points.

## Formal primary endpoint

For pooled N=256, the primary paired value is
`PropertyError_FixedK2 - PropertyError_LinearK2`.  Confirmation requires the
lower endpoint of a deterministic 20,000-resample paired bootstrap CI to be
strictly greater than zero, along with quality, equal-compute, and C1/C2
consistency gates.

No rescue tuning is permitted.
