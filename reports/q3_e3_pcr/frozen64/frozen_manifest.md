# Q3 E3-PCR frozen64 manifest

- Frozen source commit: `b65f42a8792004c7c820e59fa4413e1310e06143`
- Q3 checkpoint: `/data/dxl/results/postgen_fastgate/q3_refiner/model/q3_gate.joblib`
- Q3 checkpoint SHA256: `b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e`
- Config SHA256: `50d10efdea1050a84de6b2872f78742c2468ff4bef45cd7544fb30cef31eb87a`
- Training seeds: `20000–20063`
- Evaluation seeds: `32000–32063`
- Gate: 14 → 8 → 1 tanh MLP, threshold 0.5, 129 parameters
- Refiner: 5 steps, eta 0.01, 0.02 Å per-step radius, 3 backtracks
- Atomic numbers modified: `False`
- Cell modified: `False`
- Tuning after freeze: `False`
