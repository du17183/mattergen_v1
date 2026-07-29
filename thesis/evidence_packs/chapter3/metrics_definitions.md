# 第3章评价指标定义

| 指标 | 定义 | 单位 | 方向 | 数据字段 | 实现来源 |
| --- | --- | --- | --- | --- | --- |
| Pre-relaxation maximum force | max atom-wise force norm before MatterSim relaxation | eV/Å | lower | *_max_force / pre_relax_max_force_ev_ang | S14_E3_FROZEN_CORE |
| Relaxation RMSD | RMSDStructureMatcher displacement between initial and relaxed structures | Å | lower | *_rmsd / rmsd_from_relaxation | S29_EVAL_RMSD,S30_RMSD_UTIL |
| E-hull | energy above the TRI2024correction convex hull | eV/atom | lower | *_ehull / energy_above_hull_per_atom | S28_EVAL_ENERGY |
| Stable | E-hull <= 0.1 eV/atom | bool/rate | higher | *_stable / stable | S28_EVAL_ENERGY |
| Metastable | E-hull <= 0.2 eV/atom in project reports | bool/rate | higher | report aggregate | S14_E3_FROZEN_CORE |
| NUS | Novel AND Unique AND Stable | bool/rate | higher | *_nus / novel_unique_stable | S28_EVAL_ENERGY |
| MSUN | Metastable AND Novel AND Unique | bool/rate | higher | report aggregate msun | S14_E3_FROZEN_CORE |
| Novel | no structure match in reference dataset | bool/rate | higher | *_novel / novel | S27_EVAL_STRUCTURE |
| Unique | unique within the generated sample set | bool/rate | higher | *_unique / unique | S27_EVAL_STRUCTURE |
| Composition validity | SMACT composition validity | bool/rate | higher | *_composition_valid / comp_validity | S27_EVAL_STRUCTURE |
| Structure validity | minimum distance >=0.5 Å and volume >=0.1 Å^3 | bool/rate | higher | *_structure_valid / structure_validity | S27_EVAL_STRUCTURE |
| Harm rate | selected max force exceeds baseline by >1e-6 eV/Å | rate | lower | refinement_harm | S15_E3_FORMAL_RUNNER |
| Refinement rate | fraction with gate_applied=True | rate | context-dependent | gate_on / gate_applied | S14_E3_FROZEN_CORE |
| Exact fallback | output structure hash equals input hash for Gate-off/full rejection | bool/rate | higher | exact_fallback / exact_baseline_fallback | S14_E3_FROZEN_CORE |

## 统计方法

- Paired mean difference：selected−baseline；力/E-hull/RMSD为负表示改善。
- Relative change：selected mean / baseline mean − 1。
- Bootstrap 95% CI：按seed配对差重采样；正式E3/组合使用20,000次，seed=20260728。
- Wilcoxon signed-rank：连续配对指标；正式runner采用Pratt zero method。
- McNemar exact/paired discordant binomial：二值配对指标。
- Fisher exact：泄漏overlap与held-out harm列联表，单侧alternative=less。
- Win/Tie/Loss：必须区分raw 1e-12与algorithmic 1e-6口径。
- Holm correction：E3-A与E3-G两个主力端点，family size=2。
- Leave-one-out与remove-most-favorable：正式力端点的敏感性分析。
