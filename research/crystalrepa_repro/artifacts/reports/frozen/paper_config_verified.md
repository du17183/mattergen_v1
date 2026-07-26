# Verified CrystalREPA MatterGen configuration

Source: https://arxiv.org/html/2605.08960 (Appendix B.2–B.3).

The verified MatterGen MP-20 setting uses four GNN layers, block 2 alignment, symmetric EA-NCE, temperature 0.1, alignment weight 1, Adam at 1e-4 with ReduceLROnPlateau to 1e-6, batch 128/GPU, accumulation 4, one A800-80GB, and 1900 epochs. Inference keeps the original 1000-step sampler.

Important controlled deviation: the local experiment uses the user-frozen CHGNet 0.3.0 cache. CHGNet is not one of the paper's ten teachers, so this is a CrystalREPA-like isolated diagnostic, not a bit-for-bit paper reproduction. The paper does not state a frozen-backbone protocol; its 44.6M parameter row supports full-backbone training. Base-checkpoint initialization and exact teacher internal layer are marked NOT_VERIFIED.
