# Adaptive Neighborhood P0 reference review

## Scope

This candidate tests whether the importance of the already available `d <= 7 Å` GemNet-T edges varies across MatterGen diffusion time. It does not expand the cutoff, construct unseen neighbors, or claim the first adaptive-neighborhood graph network.

## Primary antecedents

- [MatterGen](https://arxiv.org/abs/2312.03687) jointly denoises atom types, fractional coordinates, and the periodic lattice for crystal generation. The local official magnetic-density checkpoint is the frozen starting point.
- [GemNet](https://arxiv.org/abs/2106.08903) develops directional geometric message passing through directed edge embeddings and two-hop interactions. This motivates measuring the effect of distance-banded edge messages at the actual message-aggregation path.
- Related atomistic architectures use local radial neighborhoods, but this scoped review did not find a primary source establishing the exact proposed combination of a fixed-topology MatterGen GemNet-T graph, diffusion-timestep-conditioned three-band radial gains, identity initialization, and field-wise causal ablation.

The P0 therefore makes no priority claim. A systematic novelty review would still be required if the mechanism and held-out gates pass.
