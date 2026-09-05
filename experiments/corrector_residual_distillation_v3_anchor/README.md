# Corrector Residual Distillation V3: Periodic Exact Anchor

This experiment adds only a periodic exact second-forward anchor to the frozen V2
policy. The registered choices are K=4, 8, and 16; V2 is K=infinity. Late exact
(progress >= 0.7) has highest priority, followed by frozen atomic-risk fallback,
then the periodic anchor. Every exact call resets the eligible Adapter streak.

Seed intervals are disjoint: smoke 67900-67903, Stage-B 68000-68031, and
Stage-C 69000-69063. Formal256 seeds 67000-67255 are permanently frozen and are
never used for V3 tuning.
