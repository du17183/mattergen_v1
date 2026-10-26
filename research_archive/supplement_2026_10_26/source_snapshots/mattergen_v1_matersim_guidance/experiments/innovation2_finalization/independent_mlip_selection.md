# Independent MLIP selection — frozen before force evaluation

Selected evaluator: **CHGNet 0.3.0**, original pretrained checkpoint, independently trained from MatterSim.

| Candidate | Local audit | Decision |
|---|---|---|
| MACE | No importable package or weights found in audited project/ALM/root model caches | Not selected |
| SevenNet | No importable package or weights found in audited caches | Not selected |
| ORB | orb-models 0.5.5 installed; no pretrained checkpoint found in audited caches | Not selected |
| CHGNet | Existing 0.3.0 package and exact pretrained checkpoint; periodic force inference already supported by the local environment | Selected before examining force outcomes |

Checkpoint: `/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar`

SHA256: `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`

Source package: `/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages/chgnet/`.
Runtime: ALM Python 3.10, torch 2.9.0+cu128; legacy CHGNet imported after ABI-compatible graph extensions and ASE ExpCellFilter alias.

Formal32 elements are O, Fe, Co, Pd, Eu, Gd: all six are supported; 64/64 structures have valid supported atomic numbers. The model embedding supports Z=1..94; that numerical range is not a claim of equal accuracy or training coverage for every element. Formal256 coverage will be audited after generation, without selecting a replacement model based on performance.

CHGNet is not called inside C0/F0/F1 sampling, candidate acceptance, or physical correction. It previously supplied the magnetic-property guardrail, so it is independent of the **guidance force model**, not an evaluator completely unseen during method development. Its training data may overlap MatterSim's data; no claim of disjoint training datasets is made.

Evaluation uses final generated, unrelaxed periodic structures. Outputs: energy per atom (eV/atom), energy per structure (eV), per-atom forces (eV/Å), atomic force mean, MaxF. Energy-per-atom and total energy are explicitly separated. No structure is optimized using CHGNet.

Primary: paired cohort mean independent MaxF relative reduction `(mean(C0)-mean(F0))/mean(C0)`; 20,000 paired bootstrap resamples (seed 20260915). Report mean, median, P75/P90/P95/max, wins/ties/losses, absolute and relative changes. Positive central reduction triggers Formal256 evaluation; clear reversal (95% CI entirely below zero) is FAIL; a CI crossing zero is disclosed as uncertain and never called statistically confirmed. Directional PASS requires positive central reduction and more wins than losses; evidence-strength flag states whether CI excludes zero.

Sources: [CHGNet original paper](https://www.nature.com/articles/s42256-023-00716-3), [official implementation and units](https://github.com/CederGroupHub/chgnet).

SURROGATE_PROPERTY_EVAL=True. DFT_VERIFIED=False.
