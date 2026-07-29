#!/usr/bin/env python3
"""Build traceable chapter 3–6 evidence packs from frozen repository facts.

The content in this script is a documentation index, not a new experiment.
It writes only under ``thesis/evidence_packs`` and never edits archived CSV,
JSON, model code, checkpoints, seeds, or evaluation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "thesis/evidence_packs"
BASE_COMMIT = "a7d778265103cd5b547ddc07c1db4083c75513fc"
ADAPTIVE_COMMIT = "5de00419eea2d8a9be303638f2db8ece15a22366"
E3_COMMIT = "0275cbf08ed3c6321cea7d06f7a3a8edb83b7483"
E3_FROZEN_SOURCE_COMMIT = "b65f42a8792004c7c820e59fa4413e1310e06143"
E3_FORMAL_CODE_COMMIT = "5293b4b71be88b6663bbe349f3b57694a916835f"


def write_text(relative: str, text: str) -> None:
    path = PACK / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(relative: str, payload: Any) -> None:
    path = PACK / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


SOURCES = [
    {
        "source_id": "S01_MANIFEST",
        "type": "manifest",
        "relative_path": "thesis_archive/FINAL_EXPERIMENT_MANIFEST.json",
        "commit": BASE_COMMIT,
        "chapters": [3, 4, 5, 6],
        "claims": ["all experiment identity, seed, n, branch and evidence qualification"],
        "qualification": "authoritative archive manifest",
    },
    {
        "source_id": "S02_CLAIMS",
        "type": "report",
        "relative_path": "thesis/PAPER_CLAIMS_FINAL.md",
        "commit": BASE_COMMIT,
        "chapters": [3, 4, 5, 6],
        "claims": ["C1–C6 frozen wording and scientific boundaries"],
        "qualification": "authoritative writing claim register",
    },
    {
        "source_id": "S03_DATA_DICTIONARY",
        "type": "report",
        "relative_path": "thesis_archive/DATA_DICTIONARY.md",
        "commit": BASE_COMMIT,
        "chapters": [3, 5, 6],
        "claims": ["archived column definitions and units"],
        "qualification": "authoritative archive schema",
    },
    {
        "source_id": "S04_LINEAGE",
        "type": "report",
        "relative_path": "thesis_archive/EXPERIMENT_LINEAGE.md",
        "commit": BASE_COMMIT,
        "chapters": [3, 4, 5, 6],
        "claims": ["method identity, branch lineage, negative-result boundaries"],
        "qualification": "authoritative lineage narrative",
    },
    {
        "source_id": "S05_LIMITATIONS",
        "type": "report",
        "relative_path": "thesis/PAPER_LIMITATIONS.md",
        "commit": BASE_COMMIT,
        "chapters": [3, 4, 5, 6],
        "claims": ["MatterSim surrogate, no DFT/property verification, leakage and cohort limits"],
        "qualification": "mandatory disclosure",
    },
    {
        "source_id": "S06_ADAPTIVE_CONTROLLER",
        "type": "source code",
        "relative_path": "mattergen/diffusion/sampling/guidance_schedule.py",
        "commit": ADAPTIVE_COMMIT,
        "chapters": [4],
        "claims": ["phase-specific EMA, multiplier, clipping and fallback"],
        "qualification": "formal implementation snapshot",
    },
    {
        "source_id": "S07_ADAPTIVE_CFG",
        "type": "source code",
        "relative_path": "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "commit": ADAPTIVE_COMMIT,
        "chapters": [4],
        "claims": ["field residual RMS, joint cond/uncond forward and CFG lerp"],
        "qualification": "formal implementation snapshot",
    },
    {
        "source_id": "S08_PC_SAMPLER",
        "type": "source code",
        "relative_path": "mattergen/diffusion/sampling/pc_sampler.py",
        "commit": ADAPTIVE_COMMIT,
        "chapters": [3, 4],
        "claims": ["corrector then predictor calls; full path retained"],
        "qualification": "formal implementation snapshot",
    },
    {
        "source_id": "S09_ADAPTIVE_CONFIG",
        "type": "config",
        "relative_path": "thesis_archive/configs/adaptive_cfg_final.yaml",
        "commit": ADAPTIVE_COMMIT,
        "chapters": [3, 4],
        "claims": ["g0=2, alpha=.5, beta=.95, eps=1e-6, [0,5], FP32, B1, full PC"],
        "qualification": "frozen configuration",
    },
    {
        "source_id": "S10_I1_DATA",
        "type": "per-seed data",
        "relative_path": "thesis_archive/data/innovation1/per_seed_metrics.csv",
        "commit": ADAPTIVE_COMMIT,
        "chapters": [3, 4],
        "claims": ["Adaptive CFG paired 256 metrics"],
        "qualification": "formal independent data; seeds 20000–20255",
    },
    {
        "source_id": "S11_I1_REPORT",
        "type": "report",
        "relative_path": "thesis_archive/reports/innovation1/formal_final_report.json",
        "commit": "20255f1a857cd763a7ef2bf2f24c1889c98c4d1c",
        "chapters": [4, 6],
        "claims": ["Adaptive effect estimates, CI, tests and Corrector Gating No-Go"],
        "qualification": "frozen formal report",
    },
    {
        "source_id": "S12_I1_FIGURE",
        "type": "figure",
        "relative_path": "thesis/figures/generated/pdf/fig05_adaptive_cfg_results.pdf",
        "commit": BASE_COMMIT,
        "chapters": [4],
        "claims": ["Adaptive paired E-hull and Stable/NUS effects"],
        "qualification": "generated only from archived data",
    },
    {
        "source_id": "S13_E3_REFINER",
        "type": "source code",
        "relative_path": "research/postgen_fastgate/refiner_eval.py",
        "commit": E3_COMMIT,
        "chapters": [5],
        "claims": ["14 features, Gate training, position proposal, safety, backtracking"],
        "qualification": (
            "available at formal snapshot; frozen source identity "
            f"{E3_FROZEN_SOURCE_COMMIT}"
        ),
    },
    {
        "source_id": "S14_E3_FROZEN_CORE",
        "type": "source code",
        "relative_path": "research/q3_frozen64.py",
        "commit": E3_COMMIT,
        "chapters": [3, 5, 6],
        "claims": ["formal feature extraction, learned/always arms, exact fallback and metrics"],
        "qualification": "frozen core reused by formal runner",
    },
    {
        "source_id": "S15_E3_FORMAL_RUNNER",
        "type": "source code",
        "relative_path": "research/q3_formal256.py",
        "commit": E3_COMMIT,
        "chapters": [3, 5, 6],
        "claims": ["formal contract, seed audit, paired statistics and mechanism checks"],
        "qualification": f"formal run snapshot; recorded code commit {E3_FORMAL_CODE_COMMIT}",
    },
    {
        "source_id": "S16_E3_CONFIG",
        "type": "config",
        "relative_path": "thesis_archive/configs/e3_pcr_final.yaml",
        "commit": E3_COMMIT,
        "chapters": [5],
        "claims": ["129 parameters, 14→8→1, threshold, trust and backtracking constants"],
        "qualification": "frozen configuration",
    },
    {
        "source_id": "S17_I2_DATA",
        "type": "per-seed data",
        "relative_path": "thesis_archive/data/innovation2/per_seed_metrics.csv",
        "commit": E3_COMMIT,
        "chapters": [3, 5],
        "claims": ["C0/E3-A/E3-G paired 256 metrics and gate behavior"],
        "qualification": "formal independent data; seeds 40000–40255",
    },
    {
        "source_id": "S18_I2_REPORT",
        "type": "report",
        "relative_path": "thesis_archive/reports/innovation2/final_report.md",
        "commit": "41479015c5c3edc389601c4b7cc44a6db5e115cd",
        "chapters": [5],
        "claims": ["E3-PCR force, quality and sensitivity results"],
        "qualification": "frozen formal report",
    },
    {
        "source_id": "S19_GATE_MECHANISM",
        "type": "summary JSON",
        "relative_path": "reports/q3_e3_pcr/formal256/gate_mechanism_summary.json",
        "commit": E3_COMMIT,
        "chapters": [5, 6],
        "claims": ["coverage, harm, low-force harm, gain retention and displacement"],
        "qualification": "formal mechanism evidence",
    },
    {
        "source_id": "S20_RANDOM_GATE",
        "type": "table",
        "relative_path": "reports/q3_e3_pcr/frozen64/random_gate_ablation.csv",
        "commit": E3_FROZEN_SOURCE_COMMIT,
        "chapters": [5, 6],
        "claims": ["five random gates at equal 42/64 coverage; result range and mean"],
        "qualification": "frozen64 supplementary ablation, not formal256",
    },
    {
        "source_id": "S21_COHORT1_DATA",
        "type": "per-seed data",
        "relative_path": "thesis_archive/data/compatibility_1/per_seed_metrics.csv",
        "commit": "ba2303c284210fdae0a35bb0153a8ef3af45a54c",
        "chapters": [3, 6],
        "claims": ["first independent A0+E3-G cohort"],
        "qualification": "independent; seeds 41000–41063",
    },
    {
        "source_id": "S22_COHORT1_REPORT",
        "type": "report",
        "relative_path": "thesis_archive/reports/compatibility/final_report.md",
        "commit": "e358ee39a8cdd2a061a18bfaddbe88316b455048",
        "chapters": [6],
        "claims": ["cohort 1 effect, CI, p and quality"],
        "qualification": "independent compatibility report",
    },
    {
        "source_id": "S23_COHORT2_DATA",
        "type": "per-seed data",
        "relative_path": "thesis_archive/data/compatibility_2/per_seed_metrics.csv",
        "commit": "22e1db74a59476562f1f746cd4210b9420cbdf05",
        "chapters": [3, 6],
        "claims": ["second independent A0+E3-G cohort"],
        "qualification": "fully independent; seeds 50000–50063",
    },
    {
        "source_id": "S24_COHORT2_REPORT",
        "type": "report",
        "relative_path": "thesis_archive/reports/replication/final_report.md",
        "commit": "85485bc956fce1cf7d01c55baaa92c0b69fd745e",
        "chapters": [6],
        "claims": ["cohort 2 effect, CI, p, semantic W/T/L and quality"],
        "qualification": "independent replication report",
    },
    {
        "source_id": "S25_LEAK_DATA",
        "type": "per-seed data",
        "relative_path": "thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv",
        "commit": "01e9b2c30e5c58e05eaae908ba291c518b977d03",
        "chapters": [3, 6],
        "claims": ["training-overlap versus held-out Gate safety"],
        "qualification": "diagnostic only; mixed 256 invalid for independent claims",
    },
    {
        "source_id": "S26_LEAK_REPORT",
        "type": "report",
        "relative_path": "thesis_archive/reports/leakage_diagnostic/final_report.md",
        "commit": "d5bf7d00ab51a2a0b319203443391e3463e7a91b",
        "chapters": [6],
        "claims": ["Fisher test and leakage interpretation"],
        "qualification": "diagnostic report; held-out 192 supplementary only",
    },
    {
        "source_id": "S27_EVAL_STRUCTURE",
        "type": "source code",
        "relative_path": "mattergen/evaluation/metrics/structure.py",
        "commit": BASE_COMMIT,
        "chapters": [3],
        "claims": ["novel, unique, composition and structure validity"],
        "qualification": "official evaluator implementation in repository snapshot",
    },
    {
        "source_id": "S28_EVAL_ENERGY",
        "type": "source code",
        "relative_path": "mattergen/evaluation/metrics/energy.py",
        "commit": BASE_COMMIT,
        "chapters": [3],
        "claims": ["E-hull, Stable and NUS"],
        "qualification": "official evaluator implementation in repository snapshot",
    },
    {
        "source_id": "S29_EVAL_RMSD",
        "type": "source code",
        "relative_path": "mattergen/evaluation/utils/metrics_structure_summary.py",
        "commit": BASE_COMMIT,
        "chapters": [3],
        "claims": ["relaxation RMSD source structures"],
        "qualification": "official evaluator implementation in repository snapshot",
    },
    {
        "source_id": "S30_RMSD_UTIL",
        "type": "source code",
        "relative_path": "mattergen/evaluation/utils/utils.py",
        "commit": BASE_COMMIT,
        "chapters": [3],
        "claims": ["RMSDStructureMatcher conversion to angstrom"],
        "qualification": "official evaluator implementation in repository snapshot",
    },
    {
        "source_id": "S31_BASE_CONDITION_CONFIG",
        "type": "config",
        "relative_path": "configs/q3_e3_pcr_frozen64.json",
        "commit": E3_COMMIT,
        "chapters": [3, 5],
        "claims": ["dft_mag_density target 0.1 and immutable refinement fields"],
        "qualification": "frozen evaluation config",
    },
    {
        "source_id": "S32_NEGATIVE_RESULTS",
        "type": "report",
        "relative_path": "docs/experiments/negative_results_summary.md",
        "commit": BASE_COMMIT,
        "chapters": [6],
        "claims": ["representative and supplementary No-Go routes"],
        "qualification": "archive-level synthesis; some original server reports not in GitHub",
    },
]


FORMULAS = [
    {
        "formula_id": "F3_MAX_FORCE",
        "chapter": 3,
        "formula_latex": r"F_{\max}=\max_i\lVert\mathbf F_i\rVert_2",
        "variables": {"F_i": "MatterSim/CHGNet force vector for atom i"},
        "source_file": "research/q3_frozen64.py",
        "source_symbol": "relax worker pre_relax_max_force_ev_ang",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "For thesis results the force values are from MatterSim-5M; CHGNet is used only inside E3-PCR.",
    },
    {
        "formula_id": "F3_RMSD",
        "chapter": 3,
        "formula_latex": r"\operatorname{RMSD}=\operatorname{MatcherRMSD}(X_{\mathrm{relaxed}},X_{\mathrm{initial}})\;[\AA]",
        "variables": {"X_initial": "pre-relax structure", "X_relaxed": "MatterSim-relaxed structure"},
        "source_file": "mattergen/evaluation/utils/utils.py",
        "source_symbol": "compute_rmsd_angstrom",
        "source_commit": BASE_COMMIT,
        "exact_or_interpreted": "interpreted",
        "manual_confirmation_required": False,
        "notes": "Uses RMSDStructureMatcher and converts the normalized match distance to angstrom.",
    },
    {
        "formula_id": "F3_STABLE",
        "chapter": 3,
        "formula_latex": r"\mathrm{Stable}=\mathbb 1[E_{\mathrm{hull}}\le 0.1\;\mathrm{eV/atom}]",
        "variables": {"E_hull": "MatterSim-derived energy above the TRI2024correction hull"},
        "source_file": "mattergen/evaluation/metrics/energy.py",
        "source_symbol": "EnergyCapability.is_stable",
        "source_commit": BASE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Proxy stability only.",
    },
    {
        "formula_id": "F3_NUS",
        "chapter": 3,
        "formula_latex": r"\mathrm{NUS}=\mathrm{Novel}\land\mathrm{Unique}\land\mathrm{Stable}",
        "variables": {"Novel": "no reference match", "Unique": "unique within generated set"},
        "source_file": "mattergen/evaluation/metrics/energy.py",
        "source_symbol": "FracNovelUniqueStableStructures.compute_pre_aggregation_values",
        "source_commit": BASE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Stable component remains a MatterSim surrogate.",
    },
    {
        "formula_id": "F3_HARM",
        "chapter": 3,
        "formula_latex": r"\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]",
        "variables": {"F_max": "pre-relaxation maximum force in eV/angstrom"},
        "source_file": "research/q3_formal256.py",
        "source_symbol": "FORCE_HARM_EPSILON and gate mechanism analysis",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "The 1e-6 tolerance defines algorithmic harm/tie counts.",
    },
    {
        "formula_id": "F4_RESIDUAL",
        "chapter": 4,
        "formula_latex": r"r_{t,k}=s^{cond}_{t,k}-s^{uncond}_{t,k}",
        "variables": {"k": "cell, pos, or atomic_numbers", "t": "sampling call"},
        "source_file": "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "source_symbol": "score_residual_rms",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Residual remains field-specific until scalar RMS reduction.",
    },
    {
        "formula_id": "F4_FIELD_RMS",
        "chapter": 4,
        "formula_latex": r"\delta_{t,k}=\sqrt{\operatorname{mean}(r_{t,k}^{\,2})}",
        "variables": {"delta_tk": "scalar RMS for field k"},
        "source_file": "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "source_symbol": "score_residual_rms",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Residual is cast to float before square/mean.",
    },
    {
        "formula_id": "F4_FIELD_MEAN",
        "chapter": 4,
        "formula_latex": r"\delta_t=\frac{1}{|\mathcal K_t|}\sum_{k\in\mathcal K_t}\delta_{t,k}",
        "variables": {"K_t": "fields with finite valid RMS values"},
        "source_file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "source_symbol": "_mean_valid_deltas",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "One scalar controls the common guidance scale; there are not three field scales.",
    },
    {
        "formula_id": "F4_EMA",
        "chapter": 4,
        "formula_latex": r"m_{t,p}=\begin{cases}\delta_t,&m_{t-1,p}\ \mathrm{unset}\\ \beta m_{t-1,p}+(1-\beta)\delta_t,&\mathrm{otherwise}\end{cases}",
        "variables": {"p": "predictor or corrector", "beta": "0.95"},
        "source_file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "source_symbol": "GuidanceController.evaluate",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Predictor and corrector maintain independent EMA states.",
    },
    {
        "formula_id": "F4_RATIO",
        "chapter": 4,
        "formula_latex": r"q_t=\frac{\delta_t}{m_{t,p}+\epsilon}",
        "variables": {"epsilon": "1e-6"},
        "source_file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "source_symbol": "GuidanceController.evaluate",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Ratio is scalar.",
    },
    {
        "formula_id": "F4_MULTIPLIER",
        "chapter": 4,
        "formula_latex": r"u_t=\operatorname{clip}\!\left(1+\alpha(q_t-1),0.25,4\right)",
        "variables": {"alpha": "0.50"},
        "source_file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "source_symbol": "GuidanceController.evaluate",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Adaptive multiplier limit is distinct from final guidance limit.",
    },
    {
        "formula_id": "F4_GUIDANCE",
        "chapter": 4,
        "formula_latex": r"g_t=\operatorname{clip}(g_0u_t,g_{\min},g_{\max})",
        "variables": {"g0": "2.0", "g_min": "0.0", "g_max": "5.0"},
        "source_file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "source_symbol": "GuidanceController.evaluate",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "For schedule=adaptive, stage guidance equals base guidance g0.",
    },
    {
        "formula_id": "F4_CFG_FUSION",
        "chapter": 4,
        "formula_latex": r"s_t^{CFG}=s_t^{uncond}+g_t(s_t^{cond}-s_t^{uncond})",
        "variables": {"g_t": "one scalar used for all corrupted fields"},
        "source_file": "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "source_symbol": "GuidedPredictorCorrector._score_fn_unaccelerated",
        "source_commit": ADAPTIVE_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Implemented through torch.lerp(unconditional, conditional, g_t).",
    },
    {
        "formula_id": "F5_STANDARDIZE",
        "chapter": 5,
        "formula_latex": r"z_j=(x_j-\mu_j)/\sigma_j",
        "variables": {"x": "14-dimensional invariant feature vector"},
        "source_file": "research/postgen_fastgate/refiner_eval.py",
        "source_symbol": "build_network / StandardScaler",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "interpreted",
        "manual_confirmation_required": False,
        "notes": "Mathematical summary of scikit-learn StandardScaler.",
    },
    {
        "formula_id": "F5_GATE_NETWORK",
        "chapter": 5,
        "formula_latex": r"h=\tanh(W_1z+b_1),\qquad c=\sigma(W_2h+b_2)",
        "variables": {"z": "14 inputs", "h": "8 hidden units", "c": "positive-class probability"},
        "source_file": "research/postgen_fastgate/refiner_eval.py",
        "source_symbol": "build_network / MLPClassifier",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "interpreted",
        "manual_confirmation_required": False,
        "notes": "Library-level mathematical interpretation; inference calls predict_proba.",
    },
    {
        "formula_id": "F5_PARAMETER_COUNT",
        "chapter": 5,
        "formula_latex": r"14\times8+8+8\times1+1=129",
        "variables": {},
        "source_file": "research/postgen_fastgate/refiner_eval.py",
        "source_symbol": "train_gate network trainable_parameters",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "StandardScaler has no trainable neural-network parameters.",
    },
    {
        "formula_id": "F5_GATE_RULE",
        "chapter": 5,
        "formula_latex": r"a=\mathbb 1[c\ge 0.5]",
        "variables": {"a": "refinement activation flag"},
        "source_file": "research/q3_frozen64.py",
        "source_symbol": "refine",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Gate-off returns the original structure.",
    },
    {
        "formula_id": "F5_POSITION_PROPOSAL",
        "chapter": 5,
        "formula_latex": r"\Delta x_i^{(b)}=\operatorname{clipnorm}(\eta\,2^{-b}F_i,\ R_{step}2^{-b})",
        "variables": {"eta": "0.01", "b": "backtrack index 0,1,2", "R_step": "0.02 angstrom"},
        "source_file": "research/postgen_fastgate/refiner_eval.py",
        "source_symbol": "position_proposal and advance",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "Positions are updated and wrapped; atomic numbers and cell are not updated.",
    },
    {
        "formula_id": "F5_ACCEPTANCE",
        "chapter": 5,
        "formula_latex": r"\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}",
        "variables": {"X": "current structure", "X_prime": "position proposal"},
        "source_file": "research/postgen_fastgate/refiner_eval.py",
        "source_symbol": "finite_safe and advance",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "exact",
        "manual_confirmation_required": False,
        "notes": "finite_safe also requires volume >0.1 and minimum distance >=0.5 angstrom.",
    },
    {
        "formula_id": "F5_TRUST_BOUND",
        "chapter": 5,
        "formula_latex": r"\max_i\lVert x_i^{final}-x_i^{input}\rVert_{MIC}\le 5\times0.02=0.10\;\AA",
        "variables": {"MIC": "minimum-image wrapped displacement"},
        "source_file": "research/q3_frozen64.py",
        "source_symbol": "run_refinement_subset and refine postcondition",
        "source_commit": E3_COMMIT,
        "exact_or_interpreted": "interpreted",
        "manual_confirmation_required": False,
        "notes": "Bound follows per-step caps and is explicitly checked after refinement; it is not a lattice optimization.",
    },
]


SYMBOLS = [
    {
        "method": "Adaptive CFG",
        "file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "class": "GuidanceController",
        "function": "__init__, reset, stage_guidance, evaluate",
        "commit": ADAPTIVE_COMMIT,
        "inputs": "schedule, base guidance, progress, phase, field_deltas",
        "outputs": "GuidanceDecision",
        "purpose": "phase-local EMA and final adaptive guidance",
        "paper_section": "4.4–4.5",
        "important_lines_or_logic": "evaluate: first EMA initialization; beta update; ratio; [0.25,4] multiplier; [0,5] guidance",
    },
    {
        "method": "Adaptive CFG",
        "file": "mattergen/diffusion/sampling/guidance_schedule.py",
        "class": "",
        "function": "_mean_valid_deltas",
        "commit": ADAPTIVE_COMMIT,
        "inputs": "field RMS mapping",
        "outputs": "one scalar mean or None",
        "purpose": "combine differently shaped fields after scalar reduction",
        "paper_section": "4.3",
        "important_lines_or_logic": "finite valid field RMS values are averaged arithmetically",
    },
    {
        "method": "Adaptive CFG",
        "file": "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "class": "",
        "function": "score_residual_rms",
        "commit": ADAPTIVE_COMMIT,
        "inputs": "unconditional and conditional Diffusable scores",
        "outputs": "per-field RMS dictionary and error string",
        "purpose": "compute cell/pos/atomic_numbers residual summaries",
        "paper_section": "4.2–4.3",
        "important_lines_or_logic": "shape/finite checks; residual=conditional-unconditional; float RMS",
    },
    {
        "method": "Adaptive CFG",
        "file": "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "class": "GuidedPredictorCorrector",
        "function": "_score_fn_unaccelerated",
        "commit": ADAPTIVE_COMMIT,
        "inputs": "Diffusable state and timestep",
        "outputs": "guided score for every corrupted field",
        "purpose": "joint cond/uncond forward, adaptive decision and CFG lerp",
        "paper_section": "4.5–4.6",
        "important_lines_or_logic": "collate joint batch; split scores; controller.evaluate; torch.lerp with one final_guidance",
    },
    {
        "method": "Adaptive CFG",
        "file": "mattergen/diffusion/sampling/pc_sampler.py",
        "class": "PredictorCorrector",
        "function": "denoise",
        "commit": ADAPTIVE_COMMIT,
        "inputs": "initial batch and timesteps",
        "outputs": "sampled batch",
        "purpose": "invoke full corrector and predictor updates",
        "paper_section": "4.6",
        "important_lines_or_logic": "normal path calls all configured corrector steps then predictor at every sampling step",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/postgen_fastgate/refiner_eval.py",
        "class": "",
        "function": "build_network",
        "commit": E3_COMMIT,
        "inputs": "14-dimensional feature vector",
        "outputs": "StandardScaler + tanh MLPClassifier probability",
        "purpose": "learn whether refinement is likely beneficial",
        "paper_section": "5.3–5.4",
        "important_lines_or_logic": "hidden=(8,), alpha=.1, max_iter=2000, seed=20260728",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/postgen_fastgate/refiner_eval.py",
        "class": "",
        "function": "historical_training_data / train_gate",
        "commit": E3_COMMIT,
        "inputs": "A0 seeds 20000–20063, 14 features, force-improvement labels",
        "outputs": "129-parameter Gate checkpoint and OOF diagnostics",
        "purpose": "train the selective gate without training MatterGen or CHGNet",
        "paper_section": "5.4",
        "important_lines_or_logic": "label=refined max force < baseline; 8-fold StratifiedKFold; final fit on 64 rows",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/q3_frozen64.py",
        "class": "",
        "function": "extract_features",
        "commit": E3_COMMIT,
        "inputs": "generated structure and frozen CHGNet efsm prediction",
        "outputs": "14 invariant scalar features",
        "purpose": "construct Gate inputs",
        "paper_section": "5.3",
        "important_lines_or_logic": "FEATURE_COLUMNS exact order; no missing values allowed",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/postgen_fastgate/refiner_eval.py",
        "class": "",
        "function": "position_proposal",
        "commit": E3_COMMIT,
        "inputs": "structure, CHGNet force vectors, backtrack scale",
        "outputs": "wrapped position-only proposal",
        "purpose": "equivariant bounded position update",
        "paper_section": "5.5–5.6",
        "important_lines_or_logic": "eta=.01; per-atom norm cap=.02*scale; candidate.wrap",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/postgen_fastgate/refiner_eval.py",
        "class": "",
        "function": "minimum_distance / finite_safe / advance",
        "commit": E3_COMMIT,
        "inputs": "current structures and frozen CHGNet model",
        "outputs": "accepted structures, predictions and counters",
        "purpose": "geometry safety, energy backtracking and step fallback",
        "paper_section": "5.6–5.8",
        "important_lines_or_logic": "scales 1,.5,.25; min distance .5; positive finite volume; energy nonincrease; rejected item copied",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/q3_frozen64.py",
        "class": "",
        "function": "run_refinement_subset / refine",
        "commit": E3_COMMIT,
        "inputs": "C0 structures, Gate activation mask and frozen CHGNet",
        "outputs": "E3-G and E3-A structures plus trace manifest",
        "purpose": "execute five steps and enforce immutable species/cell/exact fallback",
        "paper_section": "5.5–5.8",
        "important_lines_or_logic": "five advance calls; atomic/cell equality; displacement <=.1000001; gate-off hash equality",
    },
    {
        "method": "Learned-Gated E3-PCR",
        "file": "research/q3_formal256.py",
        "class": "",
        "function": "validate_frozen_contract / force_robustness / audited_refinement",
        "commit": E3_COMMIT,
        "inputs": "frozen hashes, seeds 40000–40255 and three-arm outputs",
        "outputs": "formal reports, CI/tests and mechanism counters",
        "purpose": "ensure no retuning and calculate formal evidence",
        "paper_section": "5.9–5.11",
        "important_lines_or_logic": "20k paired bootstrap; Wilcoxon Pratt; Holm family size 2; seed intersections empty",
    },
]


CLAIMS = [
    {
        "claim_id": "C1_ADAPTIVE_CFG_DIRECTIONAL",
        "chapter": 4,
        "section": "4.8",
        "exact_wording": (
            "在20000–20255的256个配对样本中，Adaptive CFG相对C0使代理E-hull降低"
            "0.003435 eV/atom、Stable提高5.859 pp、NUS提高3.516 pp；总体方向正向，"
            "但三项配对统计均未达到显著性。"
        ),
        "source_data": ["S10_I1_DATA"],
        "source_code": ["S06_ADAPTIVE_CONTROLLER", "S07_ADAPTIVE_CFG"],
        "figure": "Figure 5",
        "table": "Table 02_innovation1",
        "seed_range": "20000–20255",
        "n": 256,
        "statistical_evidence": (
            "E-hull CI [-0.017926,0.011030], raw p=.357; Stable CI "
            "[-1.5625,13.2813] pp, p=.146; NUS CI [-2.7344,9.7656] pp, p=.342."
        ),
        "limitation": "MatterSim-5M surrogate; non-significant paired inference; no target-property verification.",
        "forbidden_variant": "Adaptive CFG统计显著改善真实热力学稳定性。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C2_E3PCR_FORCE",
        "chapter": 5,
        "section": "5.9",
        "exact_wording": (
            "在40000–40255的独立256个样本中，E3-G把预松弛最大力均值从"
            "0.342964降至0.263107 eV/Å，相对下降23.28%，配对均值差95% CI为"
            "[-0.144966,-0.032453]，Holm校正p=4.19e-10。"
        ),
        "source_data": ["S17_I2_DATA"],
        "source_code": ["S13_E3_REFINER", "S14_E3_FROZEN_CORE", "S15_E3_FORMAL_RUNNER"],
        "figure": "Figure 6",
        "table": "Table 03_innovation2",
        "seed_range": "40000–40255",
        "n": 256,
        "statistical_evidence": "20,000 paired bootstrap; Wilcoxon Pratt; Holm family size 2; raw W/T/L=163/0/93.",
        "limitation": "MatterSim-5M pre-relax force; not DFT or synthesizability evidence.",
        "forbidden_variant": "E3-PCR已通过DFT证明提升真实材料稳定性。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C3_GATE_HARM",
        "chapter": 5,
        "section": "5.10",
        "exact_wording": (
            "相对Always-on，Learned Gate把覆盖率从100%降至66.406%，harm从25.391%"
            "降至18.359%，低力子集harm从29.688%降至17.969%，并保留80.657%的平均"
            "降力收益；harm差异McNemar p=0.000534。"
        ),
        "source_data": ["S17_I2_DATA", "S19_GATE_MECHANISM"],
        "source_code": ["S14_E3_FROZEN_CORE", "S15_E3_FORMAL_RUNNER"],
        "figure": "Figure 7",
        "table": "Table 04_gate_ablation",
        "seed_range": "40000–40255",
        "n": 256,
        "statistical_evidence": "paired exact McNemar; E3-A-only harm=22, E3-G-only harm=4.",
        "limitation": "Always-on平均降力更大；E3-G仍有47个算法语义harm样本。",
        "forbidden_variant": "Learned Gate平均降力优于Always-on，且保证每个样本安全。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C4_COMBINATION_COHORT1",
        "chapter": 6,
        "section": "6.3",
        "exact_wording": (
            "第一组独立组合cohort（41000–41063，n=64）中，A0+E3-G把预松弛"
            "最大力从0.217302降至0.158416 eV/Å，相对下降27.10%，95% CI为"
            "[-0.092341,-0.029754]，p=7.74e-5。"
        ),
        "source_data": ["S21_COHORT1_DATA"],
        "source_code": ["S06_ADAPTIVE_CONTROLLER", "S13_E3_REFINER"],
        "figure": "Figure 9",
        "table": "Table 05_compatibility_cohort1",
        "seed_range": "41000–41063",
        "n": 64,
        "statistical_evidence": "raw W/T/L=45/0/19; algorithmic 1e-6 W/T/L=34/19/11.",
        "limitation": "独立64样本；不得替代E3-PCR正式256或与cohort 2事后合并。",
        "forbidden_variant": "预注册128-seed组合实验的前64个样本。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C5_COMBINATION_COHORT2",
        "chapter": 6,
        "section": "6.4",
        "exact_wording": (
            "第二组完全独立cohort（50000–50063，n=64）中，A0+E3-G把预松弛"
            "最大力从0.265280降至0.214830 eV/Å，相对下降19.02%，95% CI为"
            "[-0.102213,-0.010696]，p=0.000587；算法语义W/T/L=35/18/11。"
        ),
        "source_data": ["S23_COHORT2_DATA"],
        "source_code": ["S06_ADAPTIVE_CONTROLLER", "S13_E3_REFINER"],
        "figure": "Figure 9 and Figure 10",
        "table": "Table 06_compatibility_cohort2",
        "seed_range": "50000–50063",
        "n": 64,
        "statistical_evidence": "raw numeric W/T/L=46/0/18; Gate-off exact structure ties=18.",
        "limitation": "效应小于cohort 1；必须并列报告异质性。",
        "forbidden_variant": "只报告更有利的cohort或把两组pooled为预注册128。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C6_LEAKAGE_SAFETY",
        "chapter": 6,
        "section": "6.6",
        "exact_wording": (
            "训练重叠没有明显夸大平均最大力改善，但显著高估Gate安全性：overlap "
            "harm=0/64，held-out harm=31/192=16.15%，单侧Fisher p=6.87e-5。"
        ),
        "source_data": ["S25_LEAK_DATA"],
        "source_code": ["S15_E3_FORMAL_RUNNER"],
        "figure": "Figure 11",
        "table": "Table 08_leakage_diagnostic",
        "seed_range": "20000–20255",
        "n": 256,
        "statistical_evidence": "2x2 one-sided Fisher exact; held-out is supplementary only.",
        "limitation": "Mixed 256和training overlap不得作为独立正式结果。",
        "forbidden_variant": "Mixed 256独立验证或匿名化seed后的独立验证。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C7_MATTERSIM_SURROGATE",
        "chapter": 3,
        "section": "3.5",
        "exact_wording": "力、RMSD、E-hull、Stable与NUS均来自MatterSim-5M代理评价，可用于统一相对比较。",
        "source_data": ["S03_DATA_DICTIONARY", "S05_LIMITATIONS"],
        "source_code": ["S28_EVAL_ENERGY", "S29_EVAL_RMSD"],
        "figure": "Figure 4",
        "table": "Table 01_experiment_manifest",
        "seed_range": "all reported cohorts",
        "n": 0,
        "statistical_evidence": "not an effect claim",
        "limitation": "STABILITY_SOURCE=MatterSim-5M surrogate; cannot replace DFT or synthesis evidence.",
        "forbidden_variant": "MatterSim评价即DFT验证。",
        "evidence_complete": True,
    },
    {
        "claim_id": "C8_NO_DFT_OR_PROPERTY",
        "chapter": 3,
        "section": "3.3 and 3.8",
        "exact_wording": "DFT_VERIFIED=False且PROPERTY_TARGET_VERIFIED=False；dft_mag_density=0.1是条件输入，不是已验证命中结果。",
        "source_data": ["S01_MANIFEST", "S05_LIMITATIONS", "S31_BASE_CONDITION_CONFIG"],
        "source_code": [],
        "figure": "Figure 4",
        "table": "Table 01_experiment_manifest",
        "seed_range": "all reported cohorts",
        "n": 0,
        "statistical_evidence": "explicit negative evidence status",
        "limitation": "不能证明真实磁性、热力学稳定性或可合成性。",
        "forbidden_variant": "生成结构真实达到目标磁密度并通过DFT验证。",
        "evidence_complete": True,
    },
]


METRICS = [
    ("Pre-relaxation maximum force", "max atom-wise force norm before MatterSim relaxation", "eV/Å", "lower", "*_max_force / pre_relax_max_force_ev_ang", "S14_E3_FROZEN_CORE"),
    ("Relaxation RMSD", "RMSDStructureMatcher displacement between initial and relaxed structures", "Å", "lower", "*_rmsd / rmsd_from_relaxation", "S29_EVAL_RMSD,S30_RMSD_UTIL"),
    ("E-hull", "energy above the TRI2024correction convex hull", "eV/atom", "lower", "*_ehull / energy_above_hull_per_atom", "S28_EVAL_ENERGY"),
    ("Stable", "E-hull <= 0.1 eV/atom", "bool/rate", "higher", "*_stable / stable", "S28_EVAL_ENERGY"),
    ("Metastable", "E-hull <= 0.2 eV/atom in project reports", "bool/rate", "higher", "report aggregate", "S14_E3_FROZEN_CORE"),
    ("NUS", "Novel AND Unique AND Stable", "bool/rate", "higher", "*_nus / novel_unique_stable", "S28_EVAL_ENERGY"),
    ("MSUN", "Metastable AND Novel AND Unique", "bool/rate", "higher", "report aggregate msun", "S14_E3_FROZEN_CORE"),
    ("Novel", "no structure match in reference dataset", "bool/rate", "higher", "*_novel / novel", "S27_EVAL_STRUCTURE"),
    ("Unique", "unique within the generated sample set", "bool/rate", "higher", "*_unique / unique", "S27_EVAL_STRUCTURE"),
    ("Composition validity", "SMACT composition validity", "bool/rate", "higher", "*_composition_valid / comp_validity", "S27_EVAL_STRUCTURE"),
    ("Structure validity", "minimum distance >=0.5 Å and volume >=0.1 Å^3", "bool/rate", "higher", "*_structure_valid / structure_validity", "S27_EVAL_STRUCTURE"),
    ("Harm rate", "selected max force exceeds baseline by >1e-6 eV/Å", "rate", "lower", "refinement_harm", "S15_E3_FORMAL_RUNNER"),
    ("Refinement rate", "fraction with gate_applied=True", "rate", "context-dependent", "gate_on / gate_applied", "S14_E3_FROZEN_CORE"),
    ("Exact fallback", "output structure hash equals input hash for Gate-off/full rejection", "bool/rate", "higher", "exact_fallback / exact_baseline_fallback", "S14_E3_FROZEN_CORE"),
]


FIGURE_TABLE = [
    ("Figure 1", "full method architecture", "Chapter 6 overview", "thesis/figures/generated/pdf/fig01_full_method_architecture.pdf", "thesis/figures/source_data/fig01_full_method_architecture.csv", "conceptual; no pooled effect"),
    ("Figure 2", "Adaptive CFG mechanism", "Chapter 4 method", "thesis/figures/generated/pdf/fig02_adaptive_cfg_mechanism.pdf", "thesis/figures/source_data/fig02_adaptive_cfg_mechanism.csv", "one shared scale, separate phase EMA"),
    ("Figure 3", "E3-PCR mechanism", "Chapter 5 method", "thesis/figures/generated/pdf/fig03_e3pcr_mechanism.pdf", "thesis/figures/source_data/fig03_e3pcr_mechanism.csv", "14→8→1, position-only"),
    ("Figure 4", "experiment lineage", "Chapter 3/6 evidence control", "thesis/figures/generated/pdf/fig04_experiment_lineage.pdf", "thesis/figures/source_data/fig04_experiment_lineage.csv", "Mixed 256 is diagnostic"),
    ("Figure 5", "Adaptive formal results", "Chapter 4 results", "thesis/figures/generated/pdf/fig05_adaptive_cfg_results.pdf", "thesis/figures/source_data/fig05_adaptive_cfg_results.csv", "must show non-significant CI"),
    ("Figure 6", "E3-PCR three-arm formal", "Chapter 5 results", "thesis/figures/generated/pdf/fig06_e3pcr_force_formal256.pdf", "thesis/figures/source_data/fig06_e3pcr_force_formal256.csv", "Always-on mean effect is larger"),
    ("Figure 7", "Gate safety ablation", "Chapter 5 mechanism", "thesis/figures/generated/pdf/fig07_gate_safety_ablation.pdf", "thesis/figures/source_data/fig07_gate_safety_ablation.csv", "coverage/harm/retained gain together"),
    ("Figure 8", "confidence and force gain", "Chapter 5 descriptive", "thesis/figures/generated/pdf/fig08_gate_confidence_force_gain.pdf", "thesis/figures/source_data/fig08_gate_confidence_force_gain.csv", "descriptive, not causal/calibration proof"),
    ("Figure 9", "two independent cohorts", "Chapter 6 combination", "thesis/figures/generated/pdf/fig09_combination_replication_forest.pdf", "thesis/figures/source_data/fig09_combination_replication_forest.csv", "no pooled estimate"),
    ("Figure 10", "cohort 2 paired plot", "Chapter 6 replication", "thesis/figures/generated/pdf/fig10_independent64_pairplot.pdf", "thesis/figures/source_data/fig10_independent64_pairplot.csv", "algorithmic ties explained"),
    ("Figure 11", "leakage diagnostic", "Chapter 6 integrity", "thesis/figures/generated/pdf/fig11_leakage_diagnostic.pdf", "thesis/figures/source_data/fig11_leakage_diagnostic.csv", "diagnostic only"),
    ("Figure 12", "No-Go routes", "Chapter 6 discussion", "thesis/figures/generated/pdf/fig12_negative_routes_summary.pdf", "thesis/figures/source_data/fig12_negative_routes_summary.csv", "do not package No-Go as contribution"),
    ("Table 01", "experiment manifest", "Chapter 3", "thesis/tables/markdown/01_experiment_manifest.md", "thesis/tables/csv/01_experiment_manifest.csv", "seed and evidence qualification"),
    ("Table 02", "innovation 1", "Chapter 4", "thesis/tables/markdown/02_innovation1.md", "thesis/tables/csv/02_innovation1.csv", "non-significant"),
    ("Table 03", "innovation 2", "Chapter 5", "thesis/tables/markdown/03_innovation2.md", "thesis/tables/csv/03_innovation2.csv", "surrogate metrics"),
    ("Table 04", "Gate ablation", "Chapter 5", "thesis/tables/markdown/04_gate_ablation.md", "thesis/tables/csv/04_gate_ablation.csv", "Always-on vs learned"),
    ("Table 05", "cohort 1", "Chapter 6", "thesis/tables/markdown/05_compatibility_cohort1.md", "thesis/tables/csv/05_compatibility_cohort1.csv", "separate cohort"),
    ("Table 06", "cohort 2", "Chapter 6", "thesis/tables/markdown/06_compatibility_cohort2.md", "thesis/tables/csv/06_compatibility_cohort2.csv", "separate cohort"),
    ("Table 07", "combination summary", "Chapter 6", "thesis/tables/markdown/07_combination_summary.md", "thesis/tables/csv/07_combination_summary.csv", "no pooled p"),
    ("Table 08", "leakage diagnostic", "Chapter 6", "thesis/tables/markdown/08_leakage_diagnostic.md", "thesis/tables/csv/08_leakage_diagnostic.csv", "formal=false"),
    ("Table 09", "negative routes", "Chapter 6/appendix", "thesis/tables/markdown/09_negative_results.md", "thesis/tables/csv/09_negative_results.csv", "source recovery varies"),
]


CHAPTERS: dict[int, dict[str, Any]] = {
    3: {
        "title": "MatterGen基线、数据与评价体系",
        "goal": "定义条件晶体生成任务、C0基线、数据资格、代理评价和配对统计口径。",
        "questions": [
            "C0、A0、E3-A、E3-G和完整方法分别是什么？",
            "dft_mag_density=0.1是何种输入，当前是否有独立属性真值？",
            "MatterSim-5M指标可以支持哪些相对结论，不能支持哪些结论？",
            "各正式、补充、诊断seed如何隔离？",
        ],
        "outline": [
            "3.1 条件晶体生成任务定义",
            "3.2 MatterGen基线模型",
            "3.3 dft_mag_density条件生成设置",
            "3.4 实验数据与seed划分",
            "3.5 MatterSim-5M代理评价流程",
            "3.6 评价指标定义",
            "3.7 配对统计方法",
            "3.8 数据独立性与真实性控制",
        ],
        "relationship": "承接理论章节，为第4章Adaptive CFG、第5章E3-PCR及第6章组合/审计提供统一基线和评价口径。",
        "source_facts": [
            "C0为原始dft_mag_density MatterGen，constant CFG scale=2.0；完整Predictor/Corrector、FP32、batch_size=1。",
            "A0=C0+Multi-field Residual-driven Online Adaptive CFG。",
            "E3-A/E3-G从同一个C0结构分别执行Always-on或Learned-Gated位置精修。",
            "MatterGen是生成模型，MatterSim-5M是论文评价代理，CHGNet只用于Gate特征与E3-PCR局部更新；三者不得混同。",
            "条件目标为dft_mag_density=0.1，但PROPERTY_TARGET_VERIFIED=False。",
            "STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False。",
        ],
        "parameters": [
            "C0 guidance_scale=2.0",
            "Predictor/Corrector=full",
            "precision=FP32",
            "batch_size=1",
            "dft_mag_density target=0.1",
            "MatterSim stability threshold=0.1 eV/atom",
        ],
        "formula_ids": ["F3_MAX_FORCE", "F3_RMSD", "F3_STABLE", "F3_NUS", "F3_HARM"],
        "experiments": [
            "Gate training: 20000–20063。",
            "Adaptive CFG formal: 20000–20255，n=256；与Gate训练重叠不影响创新点一，因为A0不使用Q3 Gate。",
            "E3-PCR formal: 40000–40255，n=256，和Gate训练交集为0。",
            "组合cohort 1: 41000–41063，n=64，独立。",
            "组合cohort 2: 50000–50063，n=64，完全独立。",
            "Leakage overlap 20000–20063仅诊断；held-out 20064–20255仅补充；Mixed 256不得用于独立结论。",
        ],
        "results": ["本章不主张方法效果；只冻结实验身份、评价和证据资格。"],
        "figures": ["Figure 4"],
        "tables": ["Table 01"],
        "allowed": [
            "MatterSim用于相同流程下的方法间代理相对比较。",
            "预松弛最大力、RMSD、E-hull、Stable和NUS可按统一代理口径报告。",
            "两个64-seed cohort是两次独立证据。",
        ],
        "forbidden": [
            "MatterSim等价于DFT。",
            "条件输入0.1证明输出真实磁密度命中。",
            "代理Stable证明可合成。",
            "把Mixed 256或training overlap写成独立验证。",
        ],
        "limitations": [
            "无DFT、无实验合成验证、无目标属性独立验证。",
            "同一项目数据域、单一条件checkpoint和统一代理评价器。",
            "精确原始生成CLI未作为可移植归档的一部分；可使用冻结配置和manifest，不得编造命令。",
        ],
        "symbols": ["MetricsStructureSummary.rmsd_from_relaxation", "compute_rmsd_angstrom", "EnergyCapability.is_stable", "FracNovelUniqueStableStructures", "structure_validity"],
        "data": ["S01_MANIFEST", "S03_DATA_DICTIONARY", "S09_ADAPTIVE_CONFIG", "S10_I1_DATA", "S17_I2_DATA", "S21_COHORT1_DATA", "S23_COHORT2_DATA", "S25_LEAK_DATA"],
        "commits": [ADAPTIVE_COMMIT, E3_COMMIT],
        "unsupported": [
            "真实dft_mag_density命中率：NOT_SUPPORTED_BY_CURRENT_REPOSITORY",
            "DFT能量/力/声子/动力学稳定性：NOT_SUPPORTED_BY_CURRENT_REPOSITORY",
            "实验可合成性：NOT_SUPPORTED_BY_CURRENT_REPOSITORY",
        ],
        "chatgpt_notes": "先建立证据等级，再介绍指标；所有Stable/E-hull/RMSD旁保留surrogate限定。不要把现有旧提纲的章节编号自动带入本证据包。",
        "codex_checks": ["术语C0/A0/E3-A/E3-G一致", "每个seed范围和n一致", "代理/DFT/属性边界显式", "Mixed 256资格正确"],
    },
    4: {
        "title": "多字段残差驱动的在线Adaptive CFG",
        "goal": "从正式commit恢复Adaptive CFG的精确公式、伪代码、集成边界和256-seed结果。",
        "questions": [
            "如何把cell、pos、atomic_numbers三种不同形状残差变成稳定控制信号？",
            "EMA与guidance scale如何按predictor/corrector阶段更新？",
            "方法是否跳过任何采样步骤或改变物理forward数量？",
            "正式结果支持何种强度的结论？",
        ],
        "outline": [
            "4.1 固定CFG的局限",
            "4.2 条件与无条件分支",
            "4.3 三字段残差定义",
            "4.4 EMA残差状态",
            "4.5 在线Guidance更新",
            "4.6 完整算法流程",
            "4.7 计算开销",
            "4.8 正式实验结果",
            "4.9 讨论与限制",
        ],
        "relationship": "以第3章C0和评价口径为基础，输出A0；A0随后作为第6章完整组合方法的上游。",
        "source_facts": [
            "conditional与unconditional输入先collate为一次joint model forward，再拆分score。",
            "cell、pos、atomic_numbers残差分别计算RMS，只有在标量化后才求算术平均。",
            "predictor和corrector各自维护EMA；首个观测直接初始化EMA。",
            "当前实现产生一个全局guidance scale，三个字段共享，不是三套独立scale。",
            "invalid residual/EMA触发stage guidance fallback。",
            "Adaptive CFG不启用cfg acceleration或Corrector Gating；完整corrector和predictor流程保留。",
            "控制器增加三字段RMS归约和常数级标量运算，但不减少或增加MatterGen模型forward次数。",
        ],
        "parameters": ["g0=2.0", "alpha=0.50", "beta=0.95", "epsilon=1e-6", "multiplier clip=[0.25,4]", "guidance clip=[0,5]"],
        "formula_ids": ["F4_RESIDUAL", "F4_FIELD_RMS", "F4_FIELD_MEAN", "F4_EMA", "F4_RATIO", "F4_MULTIPLIER", "F4_GUIDANCE", "F4_CFG_FUSION"],
        "experiments": [
            "C0 vs A0严格配对，seeds 20000–20255，n=256。",
            "每个方法generation与MatterSim relaxation均256/256成功，initial-state配对通过，Determinism Level 1。",
            "未根据正式结果重新调参。",
        ],
        "results": [
            "E-hull C0=0.143667，A0=0.140232，差=-0.003435 eV/atom；CI跨0，p=.357。",
            "Stable C0=41.016%，A0=46.875%，差=+5.859 pp；CI跨0，p=.146。",
            "NUS C0=22.266%，A0=25.781%，差=+3.516 pp；CI跨0，p=.342。",
            "三项方向均正向，但均不得称统计显著。",
        ],
        "figures": ["Figure 2", "Figure 5"],
        "tables": ["Table 02"],
        "allowed": ["多字段在线反馈使三项代理指标呈总体正向趋势。", "算法保留完整Predictor/Corrector。", "控制器公式可标为代码精确等价。"],
        "forbidden": ["Adaptive CFG统计显著提升。", "该方法通过跳步或Corrector Gating加速。", "三个字段各使用独立guidance scale。", "代理结果证明真实磁稳定性。"],
        "limitations": ["配对统计未显著。", "单checkpoint、单目标和单采样配置。", "没有独立属性命中验证或DFT。", "精确控制开销没有作为正式主效果冻结。"],
        "symbols": ["GuidanceController", "_mean_valid_deltas", "score_residual_rms", "GuidedPredictorCorrector._score_fn_unaccelerated", "PredictorCorrector.denoise"],
        "data": ["S09_ADAPTIVE_CONFIG", "S10_I1_DATA", "S11_I1_REPORT"],
        "commits": [ADAPTIVE_COMMIT],
        "unsupported": ["其他条件字段/采样步数的泛化：NOT_SUPPORTED_BY_CURRENT_REPOSITORY", "真实属性命中：NOT_SUPPORTED_BY_CURRENT_REPOSITORY"],
        "chatgpt_notes": "先写固定scale问题，再按residual→RMS→mean→phase EMA→ratio→multiplier→shared scale→lerp展开；主结果紧跟非显著性。",
        "codex_checks": ["公式与GuidanceController一致", "首EMA初始化分支写明", "全局scale而非field-wise scale", "未混入Corrector Gating"],
    },
    5: {
        "title": "Learned-Gated E3-PCR后生成精修方法",
        "goal": "恢复选择Gate、安全有界等变位置精修、正式三臂实验和Always-on/Random Gate消融证据。",
        "questions": [
            "14维风险特征和129参数Gate如何构造？",
            "Gate、Refiner和Fallback各自承担什么功能？",
            "如何限制位置更新并拒绝不安全/升能proposal？",
            "平均降力和harm控制之间如何权衡？",
        ],
        "outline": [
            "5.1 生成后局部物理不一致问题",
            "5.2 方法总体框架",
            "5.3 14维风险特征",
            "5.4 129参数Learned Gate",
            "5.5 等变位置更新",
            "5.6 Trust region与位移限制",
            "5.7 Backtracking与安全检查",
            "5.8 Exact fallback",
            "5.9 正式256-seed实验",
            "5.10 Always-on与Random Gate消融",
            "5.11 机制分析与局限",
        ],
        "relationship": "第3章定义C0与评价；本章建立可连接C0或A0的独立后生成模块；第6章验证与A0组合。",
        "source_facts": [
            "14特征依次为num_atoms、volume_per_atom、mass_density、minimum_distance、atomic_number_mean/std、cell_condition、CHGNet energy/atom、force RMS/max/mean、stress RMS/maxabs和mag density。",
            "Gate为StandardScaler+MLPClassifier，14→8→1，tanh隐藏层，129个神经网络参数，阈值0.5。",
            "训练样本为A0 seeds 20000–20063；标签为5步位置精修后最大力是否低于基线；8折OOF只作开发诊断。",
            "Gate只判断是否执行；E3-PCR用CHGNet force-vector方向执行最多5步；Fallback返回原始结构。",
            "每步eta=.01、每原子0.02 Å cap，回溯scale为1、1/2、1/4；候选需finite、volume>0.1、min distance>=0.5 Å且CHGNet energy不升。",
            "原子种类和晶格不变；最终wrapped累计最大位移检查<=0.10 Å；Gate-off和全拒绝exact fallback。",
            "推理不训练MatterGen或CHGNet，且不改变原始MatterGen采样轨迹。",
        ],
        "parameters": ["features=14", "hidden=8", "output=1", "parameters=129", "threshold=.5", "steps=5", "eta=.01", "step cap=.02 Å", "cumulative cap=.10 Å", "backtracks=3", "min distance=.5 Å"],
        "formula_ids": ["F5_STANDARDIZE", "F5_GATE_NETWORK", "F5_PARAMETER_COUNT", "F5_GATE_RULE", "F5_POSITION_PROPOSAL", "F5_ACCEPTANCE", "F5_TRUST_BOUND", "F3_HARM"],
        "experiments": [
            "正式C0/E3-A/E3-G三臂严格配对：40000–40255，n=256，和Gate训练交集为0。",
            "C0每seed只生成一次；E3-A和E3-G从同一C0派生；MatterSim 768/768。",
            "正式主端点为预松弛最大力；20,000 paired bootstrap；Wilcoxon Pratt；两主臂Holm校正。",
            "Random Gate仅为frozen64补充消融：5个随机重复，每次42/64开启（65.625%），不是formal256。",
        ],
        "results": [
            "E3-G最大力0.342964→0.263107 eV/Å，-23.28%；CI [-0.144966,-0.032453]；Holm p=4.19e-10；raw W/T/L=163/0/93。",
            "RMSD 0.049390→0.045937 Å；E-hull基本不变；Stable/NUS/Novel/Unique保持。",
            "E3-A平均最大力-28.87%，大于E3-G的-23.28%。",
            "E3-G coverage 66.406%，harm 18.359%，low-force harm 17.969%；E3-A分别100%、25.391%、29.688%。",
            "E3-G保留80.657% Always-on平均降力收益；harm McNemar p=.000534。",
            "Random Gate frozen64五次平均相对变化-21.42%，范围[-30.00%,-13.05%]；Learned Gate frozen64为-33.56%，但该比较不是formal256主结论。",
        ],
        "figures": ["Figure 3", "Figure 6", "Figure 7", "Figure 8"],
        "tables": ["Table 03", "Table 04"],
        "allowed": ["独立formal256支持显著预松弛最大力下降。", "Learned Gate以较少覆盖降低总体和低力harm。", "位置更新安全有界且元素/晶胞保持。"],
        "forbidden": ["Learned Gate平均降力优于Always-on。", "Gate保证所有结构改善。", "E3-PCR是完整晶格/组成松弛器。", "CHGNet输出是真实磁性或DFT验证。"],
        "limitations": ["E3-G仍存在harm样本。", "Gate仅64个训练结构、129参数，对训练重叠敏感。", "只更新位置，不能修复组成或晶格错误。", "CHGNet是辅助代理，正式评价仍为MatterSim。", "Random Gate来自frozen64而非formal256。"],
        "symbols": ["FEATURE_COLUMNS", "build_network", "historical_training_data", "train_gate", "position_proposal", "finite_safe", "advance", "run_refinement_subset", "refine", "force_robustness"],
        "data": ["S16_E3_CONFIG", "S17_I2_DATA", "S18_I2_REPORT", "S19_GATE_MECHANISM", "S20_RANDOM_GATE"],
        "commits": [E3_COMMIT, E3_FROZEN_SOURCE_COMMIT, E3_FORMAL_CODE_COMMIT],
        "unsupported": ["外部材料体系泛化：NOT_SUPPORTED_BY_CURRENT_REPOSITORY", "真实DFT力下降：NOT_SUPPORTED_BY_CURRENT_REPOSITORY", "Gate概率严格校准：NOT_SUPPORTED_BY_CURRENT_REPOSITORY"],
        "chatgpt_notes": "把Gate和Refiner分开写；先报告E3-G主效果，再诚实报告Always-on更强平均降力和Gate的risk–coverage价值。",
        "codex_checks": ["14特征顺序完整", "129参数计算正确", "CHGNet与MatterSim角色区分", "raw与algorithmic W/T/L不混用", "Random Gate标注frozen64"],
    },
    6: {
        "title": "组合验证、消融、负面结果与讨论",
        "goal": "并列呈现两次独立组合验证、Gate风险消融、泄漏诊断、代表性No-Go和证据边界。",
        "questions": [
            "Adaptive CFG与E3-PCR能否串联且各自保持功能身份？",
            "两组独立cohort是否复现同方向效果，效应是否异质？",
            "训练重叠主要影响平均效果还是安全估计？",
            "失败路线揭示了哪些速度—质量、在线—离线和学习方向边界？",
        ],
        "outline": [
            "6.1 两个创新点的功能分工",
            "6.2 组合验证设计",
            "6.3 独立兼容性实验一",
            "6.4 独立复现实验二",
            "6.5 Gate消融与风险分析",
            "6.6 训练—测试泄漏诊断",
            "6.7 代表性No-Go路线",
            "6.8 计算开销",
            "6.9 真实性与可复现性",
            "6.10 局限性讨论",
        ],
        "relationship": "综合第4章采样模块与第5章后生成模块；为总结章节提供可重复性、失败边界和限制。",
        "source_facts": [
            "Adaptive CFG是完整组合方法的共享上游采样模块；E3-PCR是可接C0或A0的独立后生成模块。",
            "两个64-seed cohort分别预留并独立报告；不得事后pool为单个预注册128。",
            "Gate-off为结构级exact fallback；评价数值可能有<1e-6微差，因此报告需区分raw numeric与algorithmic counts。",
            "训练重叠诊断故意包含20000–20063；整个Mixed 256无独立资格。",
            "代表性No-Go不是创新贡献，而是停止证据和方法边界。",
        ],
        "parameters": ["cohort1=41000–41063", "cohort2=50000–50063", "harm epsilon=1e-6", "leak overlap=20000–20063", "leak held-out=20064–20255"],
        "formula_ids": ["F3_MAX_FORCE", "F3_HARM", "F5_GATE_RULE", "F5_ACCEPTANCE"],
        "experiments": [
            "Cohort 1：A0/A0+E3-G配对64；generation 64/64；relaxation 128/128。",
            "Cohort 2：A0/A0+E3-G配对64；全新seeds；generation 64/64；relaxation 128/128。",
            "Leakage：training overlap 64与held-out 192；single-sided Fisher exact。",
            "Corrector Gating正式256；RP-QTFG Gate 1八样本；CG-TDR V2八样本；其他路线证据等级见negative-results summary。",
        ],
        "results": [
            "Cohort 1最大力0.217302→0.158416，-27.10%；CI [-0.092341,-0.029754]；p=7.74e-5；raw 45/0/19，algorithmic 34/19/11。",
            "Cohort 2最大力0.265280→0.214830，-19.02%；CI [-0.102213,-0.010696]；p=.000587；algorithmic 35/18/11。",
            "两组方向一致但效应大小不同，不能声称固定幅度。",
            "Leakage overlap harm=0/64，held-out=31/192，Fisher p=6.87e-5；安全性明显被高估。",
            "Corrector Gating约1.506×，但E-hull +0.0224、Stable -9.77 pp、NUS -9.38 pp。",
            "RP-QTFG离线方向正向但在线RMSD系统恶化，延迟约+30%–49%。",
            "CG-TDR Gate可学但Teacher residual方向未可靠泛化，收益接近零或RMSD恶化。",
        ],
        "figures": ["Figure 1", "Figure 4", "Figure 7", "Figure 9", "Figure 10", "Figure 11", "Figure 12"],
        "tables": ["Table 05", "Table 06", "Table 07", "Table 08", "Table 09"],
        "allowed": ["两个完全独立cohort均复现正向降力方向。", "效应大小存在cohort异质性。", "泄漏显著高估Gate安全性。", "No-Go可用于讨论假设边界。"],
        "forbidden": ["预注册128-seed pooled实验。", "只报告cohort 1。", "Mixed 256独立验证。", "No-Go路线包装成正向创新。", "创新点一是所有历史分支公共代码。"],
        "limitations": ["两个组合cohort各n=64。", "同一数据域与MatterSim评价器。", "部分历史No-Go原始报告只留服务器/历史分支，归档仅完整保留总结。", "计算开销证据对E3-PCR组合以小样本/单环境为主，不宜外推部署成本。"],
        "symbols": ["a0_e3g_compat64 analysis", "a0_e3g_independent64 force_outcome_counts", "leakage statistics Fisher exact", "q3_formal256 mechanism"],
        "data": ["S21_COHORT1_DATA", "S22_COHORT1_REPORT", "S23_COHORT2_DATA", "S24_COHORT2_REPORT", "S25_LEAK_DATA", "S26_LEAK_REPORT", "S32_NEGATIVE_RESULTS"],
        "commits": ["ba2303c284210fdae0a35bb0153a8ef3af45a54c", "22e1db74a59476562f1f746cd4210b9420cbdf05", "01e9b2c30e5c58e05eaae908ba291c518b977d03"],
        "unsupported": ["两个cohort统一pooled效应：NOT_SUPPORTED_BY_CURRENT_REPOSITORY", "跨材料体系泛化：NOT_SUPPORTED_BY_CURRENT_REPOSITORY", "部分No-Go完整原始日志：NOT_FULLY_RECOVERED_FROM_ARCHIVE"],
        "chatgpt_notes": "两个cohort各自成节后讨论异质性；泄漏诊断写成可信性审计；No-Go按假设—观察—停止证据—认识组织。",
        "codex_checks": ["两cohort不pool", "raw与algorithmic计数标注", "Mixed资格正确", "No-Go来源恢复边界明确", "MatterSim/DFT边界保留"],
    },
}


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "- 无。"


def render_pack(chapter: int, item: dict[str, Any]) -> str:
    source_entries = [source for source in SOURCES if chapter in source["chapters"]]
    formula_entries = [formula for formula in FORMULAS if formula["formula_id"] in item["formula_ids"]]
    claim_entries = [claim for claim in CLAIMS if claim["chapter"] == chapter]
    source_table = table(
        ["source_id", "类型", "路径", "commit", "数据资格"],
        [[s["source_id"], s["type"], f"`{s['relative_path']}`", f"`{s['commit']}`", s["qualification"]] for s in source_entries],
    )
    formula_table = table(
        ["ID", "公式", "性质", "代码"],
        [[f["formula_id"], f"${f['formula_latex']}$", f["exact_or_interpreted"], f"`{f['source_file']}::{f['source_symbol']}`"] for f in formula_entries],
    )
    claim_text = render_list([claim["exact_wording"] for claim in claim_entries])
    return f"""# 第{chapter}章证据包：{item['title']}

> 本文件是写作证据，不是完整论文正文。任何项目事实必须回指 source_id；未支持内容不得由通用知识补齐。

## 1. 本章研究目标

{item['goal']}

## 2. 本章回答的核心问题

{render_list(item['questions'])}

## 3. 建议二级和三级标题

{render_list(item['outline'])}

## 4. 与前后章节的关系

{item['relationship']}

## 5. 可使用的源码事实

{render_list(item['source_facts'])}

## 6. 可使用的配置和参数

{render_list(item['parameters'])}

## 7. 公式与变量定义

{formula_table}

公式的完整变量、exact/interpreted资格见 `../FORMULA_REGISTRY.md`。

## 8. 实验设计

{render_list(item['experiments'])}

## 9. 正式结果

{render_list(item['results'])}

## 10. 对应图表

{render_list(item['figures'])}

## 11. 对应表格

{render_list(item['tables'])}

## 12. 允许写入正文的结论

{render_list(item['allowed'])}

冻结claim：

{claim_text}

## 13. 禁止夸大的结论

{render_list(item['forbidden'])}

## 14. 必须主动说明的限制

{render_list(item['limitations'])}

## 15. 对应源码文件和函数

{render_list(item['symbols'])}

## 16. 对应数据文件和字段

{render_list(item['data'])}

字段定义必须联合使用 `S03_DATA_DICTIONARY`、本章专用数据文件和 `metrics_definitions.md`/`experiment_evidence.md`。

## 17. 对应commit、分支和报告

{render_list([f'`{commit}`' for commit in item['commits']])}

{source_table}

## 18. 当前资料不支持的内容

{render_list(item['unsupported'])}

## 19. 网页ChatGPT写作注意事项

{item['chatgpt_notes']}

## 20. 写完后应由Codex核查的项目

{render_list(item['codex_checks'])}
"""


def render_chapter_json(chapter: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "chapter": chapter,
        "title": item["title"],
        "evidence_pack_version": "v1",
        "base_commit": BASE_COMMIT,
        "sections": {
            "1_research_goal": item["goal"],
            "2_core_questions": item["questions"],
            "3_section_outline": item["outline"],
            "4_relationship": item["relationship"],
            "5_source_facts": item["source_facts"],
            "6_parameters": item["parameters"],
            "7_formula_ids": item["formula_ids"],
            "8_experiment_design": item["experiments"],
            "9_formal_results": item["results"],
            "10_figures": item["figures"],
            "11_tables": item["tables"],
            "12_allowed_claims": item["allowed"],
            "13_forbidden_claims": item["forbidden"],
            "14_mandatory_limitations": item["limitations"],
            "15_code_symbols": item["symbols"],
            "16_data_sources": item["data"],
            "17_commits": item["commits"],
            "18_not_supported": item["unsupported"],
            "19_chatgpt_notes": item["chatgpt_notes"],
            "20_codex_audit": item["codex_checks"],
        },
        "claims": [claim["claim_id"] for claim in CLAIMS if claim["chapter"] == chapter],
        "sources": [source["source_id"] for source in SOURCES if chapter in source["chapters"]],
    }


def render_chatgpt_input(chapter: int, item: dict[str, Any]) -> str:
    formulas = [formula for formula in FORMULAS if formula["formula_id"] in item["formula_ids"]]
    claims = [claim for claim in CLAIMS if claim["chapter"] == chapter]
    target = {3: "6000–9000", 4: "6000–9000", 5: "7000–10000", 6: "6000–9000"}[chapter]
    formula_lines = [f"{formula['formula_id']}: ${formula['formula_latex']}$ ({formula['exact_or_interpreted']})" for formula in formulas]
    return f"""# 网页ChatGPT写作输入：第{chapter}章

你将撰写毕业论文第{chapter}章《{item['title']}》。只能依据以下证据，不得补造项目事实；通用理论若需加入，必须标成待补参考文献，不能冒充项目实现。

## 项目术语

C0=原始dft_mag_density MatterGen；A0=C0+Adaptive CFG；E3-A=Always-on E3-PCR；E3-G=Learned-Gated E3-PCR；完整方法=A0+E3-G。MatterGen是生成模型，MatterSim-5M是评价代理，CHGNet是E3-PCR辅助代理。

## 章节结构

{render_list(item['outline'])}

## 核心方法事实

{render_list(item['source_facts'])}

## 公式

{render_list(formula_lines)}

## 参数

{render_list(item['parameters'])}

## 实验结果

{render_list(item['results'])}

## 图表

图：{', '.join(item['figures']) or '无'}。

表：{', '.join(item['tables']) or '无'}。

## 允许结论

{render_list(item['allowed'])}

冻结claim原句：

{render_list([claim['exact_wording'] for claim in claims])}

## 禁止结论

{render_list(item['forbidden'])}

## 局限性

{render_list(item['limitations'])}

STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False；PROPERTY_TARGET_VERIFIED=False。

## 数据来源标识

{render_list(item['data'])}

## 正文风格

使用计算机专业学位论文的客观学术中文；先定义、再公式、再算法、再实验、再边界。所有效果注明baseline、seed、n、单位、统计口径和surrogate限制。非显著结果写“方向性趋势”，不写“证明无差异”。

## 目标字数

{target}字。当前任务只生成正文草稿；参考文献、学校模板编号和人工审阅标记保留待办。
"""


def render_section_outline(chapter: int, item: dict[str, Any]) -> str:
    return f"""# 第{chapter}章章节结构

标题：{item['title']}

{render_list(item['outline'])}

## 叙事顺序

研究问题 → 可追溯方法事实 → 公式/参数 → 冻结实验 → 效应大小与统计 → 不允许的推论 → 小结。

## 章节编号提醒

本证据包遵循本轮新编号。仓库旧版 `thesis/THESIS_OUTLINE.md` 曾将Adaptive CFG列为第3章；网页写作时不得把旧编号自动复制进本证据包。
"""


def render_source_map(chapter: int) -> str:
    entries = [source for source in SOURCES if chapter in source["chapters"]]
    return f"""# 第{chapter}章来源映射

{table(
        ['source_id', '类型', '仓库相对路径', 'commit', '支持内容', '资格'],
        [[s['source_id'], s['type'], f"`{s['relative_path']}`", f"`{s['commit']}`", '; '.join(s['claims']), s['qualification']] for s in entries],
    )}

正式commit内源码应使用 `git show <commit>:<relative_path>` 查看；不能只依赖当前工作树或README。
"""


def build_master_files() -> None:
    write_json("MASTER_SOURCE_INDEX.json", SOURCES)
    write_text(
        "MASTER_SOURCE_INDEX.md",
        "# Master Source Index\n\n"
        + table(
            ["source_id", "类型", "相对路径", "commit", "用于章节", "用于结论", "数据资格"],
            [
                [
                    s["source_id"],
                    s["type"],
                    f"`{s['relative_path']}`",
                    f"`{s['commit']}`",
                    ",".join(map(str, s["chapters"])),
                    "; ".join(s["claims"]),
                    s["qualification"],
                ]
                for s in SOURCES
            ],
        )
        + "\n\n所有路径均为仓库相对路径。正式源码事实必须同时使用路径和commit。\n",
    )

    write_json("CLAIM_TRACEABILITY.json", CLAIMS)
    write_text(
        "CLAIM_TRACEABILITY.md",
        "# Claim Traceability\n\n"
        + "\n\n".join(
            f"""## {claim['claim_id']}

- 章节/节：第{claim['chapter']}章，{claim['section']}
- 冻结表述：{claim['exact_wording']}
- 数据：{', '.join(claim['source_data'])}
- 源码：{', '.join(claim['source_code']) or '不适用'}
- 图/表：{claim['figure']}；{claim['table']}
- seeds/n：{claim['seed_range']}；n={claim['n'] if claim['n'] else '跨实验边界'}
- 统计：{claim['statistical_evidence']}
- 限制：{claim['limitation']}
- 禁止变体：{claim['forbidden_variant']}
- 证据完整：`{claim['evidence_complete']}`
"""
            for claim in CLAIMS
        ),
    )

    write_json("FORMULA_REGISTRY.json", FORMULAS)
    write_text(
        "FORMULA_REGISTRY.md",
        "# Formula Registry\n\n"
        + table(
            ["formula_id", "章", "公式", "源码符号", "commit", "性质", "人工确认"],
            [
                [
                    f["formula_id"],
                    f["chapter"],
                    f"${f['formula_latex']}$",
                    f"`{f['source_file']}::{f['source_symbol']}`",
                    f"`{f['source_commit']}`",
                    f["exact_or_interpreted"],
                    f["manual_confirmation_required"],
                ]
                for f in FORMULAS
            ],
        )
        + "\n\n`exact`仅表示与仓库代码数学等价；`interpreted`表示对库调用或实现流程的数学概括。\n",
    )

    write_json("CODE_SYMBOL_INDEX.json", SYMBOLS)
    write_text(
        "CODE_SYMBOL_INDEX.md",
        "# Code Symbol Index\n\n"
        + table(
            ["方法", "文件", "类", "函数/符号", "commit", "输入", "输出", "论文节", "关键逻辑"],
            [
                [
                    s["method"],
                    f"`{s['file']}`",
                    s["class"] or "—",
                    f"`{s['function']}`",
                    f"`{s['commit']}`",
                    s["inputs"],
                    s["outputs"],
                    s["paper_section"],
                    s["important_lines_or_logic"],
                ]
                for s in SYMBOLS
            ],
        ),
    )

    write_text(
        "FIGURE_TABLE_CROSSWALK.md",
        "# Figure–Table Crosswalk\n\n"
        + table(
            ["对象", "内容", "章节用途", "产物", "源数据", "写作防护"],
            [[*row] for row in FIGURE_TABLE],
        )
        + "\n\n重绘说明见 `thesis/figures/CORE_FIGURES_V2_REDRAW.md` 和 `thesis/figures/REDRAW_GUIDE.md`。\n",
    )

    write_text(
        "WRITING_GUARDRAILS.md",
        """# 写作防护规则

1. Adaptive CFG不得称统计显著；必须同时给出“方向正向”和“配对统计未显著”。
2. MatterSim-5M不得称DFT、真实热力学真值或实验可合成性。
3. CHGNet不得称真实磁性验证；它在E3-PCR中是辅助特征/更新代理。
4. `PROPERTY_TARGET_VERIFIED=False`；条件值0.1不能写成输出属性已命中。
5. Gate不得称绝对安全；formal256仍有算法语义harm样本。
6. Always-on平均最大力下降更多（−28.87% vs −23.28%），不得隐藏。
7. 两个64-seed cohort必须分别报告，不得包装为预注册128或事后pooled主结论。
8. Mixed 256不得用于独立正式结论；training overlap只用于诊断；held-out 192只作补充。
9. No-Go不得包装成贡献；只能用于说明假设边界、停止证据和可复用基础设施。
10. Q3只作为历史候选代号；论文正式名称使用Learned-Gated E3-PCR。
11. Win/Tie/Loss必须写明口径：formal E3-PCR主表用raw continuous；组合复现可用1e-6算法语义并将Gate-off计精确平局。
12. `displacement_mean`在统一归档逐seed CSV中缺失，不得从汇总值反推逐seed值。
13. 源码公式必须指明commit；解释性公式不得标为exact。
14. 未由当前仓库支持的事实必须写`NOT_SUPPORTED_BY_CURRENT_REPOSITORY`。
""",
    )


def build_chapters() -> None:
    for chapter, item in CHAPTERS.items():
        folder = f"chapter{chapter}"
        write_text(f"{folder}/CHAPTER{chapter}_EVIDENCE_PACK.md", render_pack(chapter, item))
        write_json(f"{folder}/CHAPTER{chapter}_EVIDENCE_PACK.json", render_chapter_json(chapter, item))
        write_text(f"{folder}/section_outline.md", render_section_outline(chapter, item))
        write_text(f"{folder}/source_map.md", render_source_map(chapter))
        write_text(f"{folder}/chatgpt_input.md", render_chatgpt_input(chapter, item))

    metric_rows = [[*metric] for metric in METRICS]
    write_text(
        "chapter3/metrics_definitions.md",
        "# 第3章评价指标定义\n\n"
        + table(["指标", "定义", "单位", "方向", "数据字段", "实现来源"], metric_rows)
        + """

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
""",
    )

    adaptive_formulas = [f for f in FORMULAS if f["chapter"] == 4]
    write_text(
        "chapter4/formula_notes.md",
        "# 第4章公式与伪代码说明\n\n"
        + "\n\n".join(
            f"## {f['formula_id']}\n\n$${f['formula_latex']}$$\n\n"
            f"- 代码：`{f['source_file']}::{f['source_symbol']}` @ `{f['source_commit']}`\n"
            f"- 性质：`{f['exact_or_interpreted']}`\n- 说明：{f['notes']}"
            for f in adaptive_formulas
        )
        + """

## 论文伪代码

```text
Input: x_t, t, base guidance g0; state m_predictor, m_corrector
1. Build unconditional and conditional inputs and execute the joint full-CFG model forward.
2. Split s_uncond and s_cond.
3. For k in {cell,pos,atomic_numbers}, compute residual r_k and scalar RMS delta_k.
4. Average valid field RMS values into scalar delta.
5. Select the EMA state belonging to the current predictor/corrector phase.
6. Initialize/update EMA, compute q, multiplier u and clipped shared guidance g.
7. For every corrupted field, return lerp(s_uncond, s_cond, g).
8. Continue the complete configured corrector/predictor update.
Output: guided score and updated phase-local EMA state.
```

明确：该算法不跳过Predictor、不跳过Corrector、不是Corrector Gating，也不是步数削减方法。
""",
    )
    write_text(
        "chapter4/experiment_evidence.md",
        """# 第4章正式实验证据

| 指标 | C0 | A0 | A0−C0 | 95% CI | raw p | W/T/L | 推论 |
|---|---:|---:|---:|---:|---:|---:|---|
| E-hull (eV/atom) | 0.143667 | 0.140232 | −0.003435 | [−0.017926,0.011030] | 0.357 | 137/1/118 | 正向、未显著 |
| Stable | 41.016% | 46.875% | +5.859 pp | [−1.563,+13.281] pp | 0.146 | 54/163/39 | 正向、未显著 |
| NUS | 22.266% | 25.781% | +3.516 pp | [−2.734,+9.766] pp | 0.342 | 40/185/31 | 正向、未显著 |

- 数据：`S10_I1_DATA`
- 报告：`S11_I1_REPORT`
- 图：Figure 5；表：Table 02。
- Holm-corrected p均为1.0。
- `FORMAL_INNOVATION1_CONFIRMED=True`是冻结工程/方向门槛结论，不等于统计显著。
""",
    )

    e3_formulas = [f for f in FORMULAS if f["chapter"] == 5]
    write_text(
        "chapter5/formula_notes.md",
        "# 第5章公式说明\n\n"
        + "\n\n".join(
            f"## {f['formula_id']}\n\n$${f['formula_latex']}$$\n\n"
            f"- 代码：`{f['source_file']}::{f['source_symbol']}` @ `{f['source_commit']}`\n"
            f"- 性质：`{f['exact_or_interpreted']}`\n- 说明：{f['notes']}"
            for f in e3_formulas
        ),
    )
    write_text(
        "chapter5/experiment_evidence.md",
        """# 第5章实验与消融证据

## 正式三臂256

| 方法 | max force | 相对C0 | RMSD | E-hull | Stable | NUS |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.342964 | — | 0.049390 | 0.156136 | 44.531% | 22.266% |
| E3-A | 0.243956 | −28.87% | 0.045057 | 0.156179 | 44.531% | 22.266% |
| E3-G | 0.263107 | −23.28% | 0.045937 | 0.156177 | 44.531% | 22.266% |

E3-G: CI=[−0.144966,−0.032453] eV/Å；raw Wilcoxon Holm-corrected p=4.19e-10；raw W/T/L=163/0/93；algorithmic 1e-6 W/T/L=127/82/47。

## Learned Gate vs Always-on

| 指标 | E3-A | E3-G |
|---|---:|---:|
| Refinement rate | 100% | 66.406% |
| Harm | 25.391% | 18.359% |
| Low-force harm | 29.688% | 17.969% |
| Mean displacement | 0.010968 Å | 0.007580 Å |
| Mean force-gain retention | 100% | 80.657% |

McNemar p=.000534。Always-on平均降力更大；Gate价值是risk–coverage折中。

## Random Gate

frozen64使用5个随机seed，每次固定42/64开启，匹配Learned Gate frozen64覆盖率65.625%。随机相对降力范围−30.00%至−13.05%，均值−21.42%；Learned Gate frozen64为−33.56%。该结果是补充消融，不替代formal256。
""",
    )

    write_text(
        "chapter6/combination_evidence.md",
        """# 第6章组合证据

| Cohort | Seeds | n | A0 | A0+E3-G | 相对变化 | 95% CI | p | raw W/T/L | algorithmic W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41000–41063 | 64 | 0.217302 | 0.158416 | −27.10% | [−0.092341,−0.029754] | 7.74e-5 | 45/0/19 | 34/19/11 |
| 2 | 50000–50063 | 64 | 0.265280 | 0.214830 | −19.02% | [−0.102213,−0.010696] | 0.000587 | 46/0/18 | 35/18/11 |

两个cohort均独立且方向一致，但不是一个预注册128-seed实验；不计算主文pooled p或pooled效应。
""",
    )
    write_text(
        "chapter6/ablation_evidence.md",
        """# 第6章消融与风险证据

- 正式E3-A vs E3-G：Always-on平均降力更大，Learned Gate覆盖更低且harm更低。
- Formal256 harm：25.391% vs 18.359%；low-force harm：29.688% vs 17.969%。
- Gain retention：80.657%；harm McNemar p=.000534。
- Random Gate只来自frozen64五次相同覆盖率重复，属于补充消融。
- Figure 8 confidence–gain只支持描述性相关，不能证明Gate已校准或因果。
- 组合cohort中Gate-off结构级exact fallback；评价器重载可产生小于1e-6的数值微差。
""",
    )
    write_text(
        "chapter6/leakage_evidence.md",
        """# 第6章训练—测试泄漏证据

| Cohort | Seeds | n | Harm | Harm rate | 数据资格 |
|---|---:|---:|---:|---:|---|
| Training overlap | 20000–20063 | 64 | 0 | 0% | diagnostic only |
| Held-out | 20064–20255 | 192 | 31 | 16.146% | supplementary only |
| Mixed | 20000–20255 | 256 | 31 | 12.109% | invalid for independent claims |

单侧Fisher exact p=6.8659e-5。overlap与held-out平均相对降力分别约−28.15%与−27.32%，但安全率明显不同。准确结论是：重叠没有明显夸大平均改善，却显著高估安全性。
""",
    )
    write_text(
        "chapter6/negative_results_evidence.md",
        """# 第6章负面结果证据

## 正文代表性三条

| 路线 | 核心观察 | 停止理由 | 论文认识 | 证据恢复 |
|---|---|---|---|---|
| Corrector Gating | 约1.506×；forward −35.37% | E-hull +0.0224、Stable −9.77 pp、NUS −9.38 pp | 减少物理forward产生速度—质量冲突 | 正式数值由S11完整恢复；原服务器专用报告未全部归档 |
| RP-QTFG | 离线单步方向正向 | 在线RMSD系统恶化，延迟约+30%–49% | 局部代理梯度不等于稳定生成轨迹引导 | `S32_NEGATIVE_RESULTS`; NOT_FULLY_RECOVERED_FROM_ARCHIVE |
| CG-TDR | Gate utility可学习 | Teacher residual方向不能泛化，收益近零/RMSD问题 | 安全选择不能修复错误修正方向 | `S32_NEGATIVE_RESULTS`; NOT_FULLY_RECOVERED_FROM_ARCHIVE |

## 表格/附录路线

Residual Reuse、Budget-aware Gating、FN-PRA、CrystalREPA、Q1 UQ-PQR、Q2 RFR、Q4 CPRC、Q5 CQPS、Q6 NS-SetRank和GPU acceleration routes统一引用`S32_NEGATIVE_RESULTS`与Table 09。部分原始日志/报告不在GitHub，必须保留`NOT_FULLY_RECOVERED_FROM_ARCHIVE`，不得据摘要扩写新数值。
""",
    )


def build_readme() -> None:
    write_text(
        "README.md",
        """# 第3–6章可追溯证据包

本目录面向后续网页ChatGPT论文写作。它将正式commit、源码符号、冻结配置、逐seed数据、报告、图表、公式和允许/禁止结论连接起来；不是新实验，也不是最终论文正文。

## 入口

- `MASTER_SOURCE_INDEX.md/json`：全部来源和数据资格。
- `CLAIM_TRACEABILITY.md/json`：核心论文claim到数据/代码/图表的映射。
- `FORMULA_REGISTRY.md/json`：公式exact/interpreted资格。
- `CODE_SYMBOL_INDEX.md/json`：正式源码符号。
- `FIGURE_TABLE_CROSSWALK.md`：图表、源数据和章节。
- `WRITING_GUARDRAILS.md`：不可违反的写作边界。
- `EVIDENCE_PACK_VALIDATION.md/json`：CPU重算和真实性验收。
- `chapter3`–`chapter6`：每章完整证据、来源、结构和`chatgpt_input.md`。

## 如何交给网页ChatGPT

1. 先复制`WRITING_GUARDRAILS.md`。
2. 再复制目标章的`chatgpt_input.md`。
3. 需要展开时补充同章`CHAPTERX_EVIDENCE_PACK.md`和`formula_notes.md`/`experiment_evidence.md`。
4. 要求网页ChatGPT保留source_id、NOT_SUPPORTED标记和所有限制。

## 如何把正文放回仓库

将返回正文保存为独立草稿文件，不覆盖本证据包；随后让Codex逐段检查claim、公式、seed、n、单位、图表和限制。旧版论文提纲的章节编号与本轮证据包不同，合稿前必须人工选择最终目录并统一交叉引用。

## CPU验证

```bash
python thesis/evidence_packs/validate_evidence_packs.py --write-report
```

验证只读取归档CSV/报告并写本目录验证报告，不启动MatterGen、MatterSim、DFT或GPU。

## 仍需人工确认

- 最终学校模板采用旧版章节编号还是本轮新编号。
- 参考文献和相关工作来源。
- 图表最终编号、排版和正文交叉引用。
- 部分No-Go原始服务器报告未完整进入GitHub，只能使用当前归档总结。
""",
    )


def main() -> None:
    build_master_files()
    build_chapters()
    build_readme()
    print(f"built chapter evidence packs in {PACK.relative_to(REPO)}")


if __name__ == "__main__":
    main()
