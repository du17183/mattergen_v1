# Reference review: Multi-field Self-Conditioned MatterGen

This P0 is an independent test of explicit temporal memory in MatterGen. It does
not use Adaptive CFG, adaptive neighborhoods, or any stopped historical route.

## Primary sources

- MatterGen introduces the diffusion formulation and GemNet-based denoiser that
  this experiment keeps as its frozen starting point: Zeni et al., *A
  generative model for inorganic materials design*, arXiv:2312.03687,
  https://arxiv.org/abs/2312.03687.
- Chen et al. describe self-conditioning as feeding a model's own earlier
  clean-data estimate into a later denoising prediction, with the estimate
  detached during training: *Analog Bits: Generating Discrete Data using
  Diffusion Models with Self-Conditioning*, arXiv:2208.04202,
  https://arxiv.org/abs/2208.04202.

## Gap and scoped hypothesis

MatterGen jointly predicts atom classes, fractional coordinates, and lattice.
A raw concatenation is ill-defined because the three fields have different
geometry and units. This experiment therefore tests three field-specific clean
prediction encoders: soft atom probabilities projected by the frozen atom table,
periodic sine/cosine coordinate deltas, and lattice deltas normalized by the
same atom-count length scaling used by MatterGen's lattice corruption. Three
zero-initialized time gates combine these encodings once, immediately before the
GemNet block stack.

The hypothesis is deliberately falsifiable: S-on must outperform the matched
S-off checkpoint. Any gain shared by S-off is attributed to partial fine-tuning,
not to self-conditioning.
