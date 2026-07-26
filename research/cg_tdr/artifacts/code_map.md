# CG-TDR code map

- A0 Adaptive CFG combination:
  `mattergen/diffusion/sampling/classifier_free_guidance.py`,
  `GuidedPredictorCorrector._score_fn`.
- Complete predictor/corrector trajectory:
  `mattergen/diffusion/sampling/pc_sampler.py`,
  `PredictorCorrector._denoise`.
- GemNet terminal invariant node features:
  `mattergen/common/gemnet/gemnet.py`, `ModelOutput.node_embeddings`;
  captured from `GemNetTAdapter.gemnet` by a temporary forward hook.
- Fractional positions, lattice and atomic fields:
  `mattergen/common/data/chemgraph.py`, `ChemGraph`.
- Terminal integration:
  `research/cg_tdr/sampler.py`,
  `CGTDRGuidedPredictorCorrector._terminal_refine_or_dump`.
- Equivariant periodic position head and bounded strain head:
  `research/cg_tdr/model.py`, `CGTDRRefiner`.

CG-TDR does not alter any Predictor, Corrector, D3PM transition, or RNG call.
The extra conditional GemNet forward happens only after the reverse trajectory.
