# GemNet joint-CFG numerical equivalence

- States: `64`
- Coverage: `{'states': 64, 'num_atoms': [8, 9, 10, 11, 12], 'phases': ['corrector', 'predictor'], 'noise_bins': ['high', 'low', 'mid'], 'sampling_step_min': 0, 'sampling_step_max': 999, 'edge_count_min': 396, 'edge_count_max': 626, 'triplet_count_min': 15866, 'triplet_count_max': 32156}`
- Static/fallback calls: `64/0`
- Bitwise tensor rate: `100.000000%`
- Maximum absolute error: `0.0`
- Maximum relative L2 error: `0.0`
- Minimum cosine similarity: `0.9999997615814209`
- First nonzero location: `None`
- First tolerance failure: `None`
- GEMNET_NUMERICAL_EQUIVALENT: `True`
- JOINT_CFG_NUMERICAL_EQUIVALENT: `True`

| Location | Max abs error | Relative L2 | Min cosine | Bitwise rate |
|---|---:|---:|---:|---:|
| block_1.0 | 0 | 0 | 0.999999762 | 100.000% |
| block_1.1 | 0 | 0 | 0.999999762 | 100.000% |
| block_2.0 | 0 | 0 | 0.999999821 | 100.000% |
| block_2.1 | 0 | 0 | 0.999999881 | 100.000% |
| block_3.0 | 0 | 0 | 0.999999762 | 100.000% |
| block_3.1 | 0 | 0 | 0.999999821 | 100.000% |
| block_4.0 | 0 | 0 | 0.999999821 | 100.000% |
| block_4.1 | 0 | 0 | 0.999999881 | 100.000% |
| conditional.atomic_numbers | 0 | 0 | 0.999999881 | 100.000% |
| conditional.cell | 0 | 0 | 0.999999881 | 100.000% |
| conditional.pos | 0 | 0 | 0.999999821 | 100.000% |
| final_cfg.atomic_numbers | 0 | 0 | 0.999999762 | 100.000% |
| final_cfg.cell | 0 | 0 | 0.999999821 | 100.000% |
| final_cfg.pos | 0 | 0 | 0.999999821 | 100.000% |
| input_edge.tensor | 0 | 0 | 0.999999881 | 100.000% |
| input_node.tensor | 0 | 0 | 0.999999821 | 100.000% |
| unconditional.atomic_numbers | 0 | 0 | 0.999999821 | 100.000% |
| unconditional.cell | 0 | 0 | 0.999999821 | 100.000% |
| unconditional.pos | 0 | 0 | 0.999999881 | 100.000% |
