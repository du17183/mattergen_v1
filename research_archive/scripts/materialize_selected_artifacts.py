#!/usr/bin/env python3
"""Copy the small, high-value subset of untracked experiment artifacts.

This intentionally excludes raw structures, checkpoints, trajectories, large
per-step traces and bulk logs. Those are covered by external_data_manifest.csv.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_archive"


def files(source: str, target: str, names: list[tuple[str, str]]) -> list[tuple[Path, Path]]:
    base = Path(source)
    dest = ROOT / target
    return [(base / src, dest / dst) for src, dst in names]


SELECTIONS: list[tuple[Path, Path]] = []

# Innovation 1: original audit and the failed/mixed controller lineage.
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_innovation1_audit",
    "research_archive/innovation1/adaptive_cfg_v1",
    [
        ("minimal_ablation.py", "code/minimal_ablation.py"),
        ("final_report.md", "results/final_report.md"),
        ("minimal_ablation/preregistered_config.json", "config/preregistered_config.json"),
        ("minimal_ablation/pipeline_status.json", "results/pipeline_status.json"),
        ("minimal_ablation/p0_property_metrics.csv", "results/p0_property_metrics.csv"),
        ("minimal_ablation/per_structure_metrics.csv", "results/per_structure_metrics.csv"),
        ("minimal_ablation/quality_metrics.csv", "results/quality_metrics.csv"),
        ("minimal_ablation/seeds.json", "seeds/seeds.json"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_robust_adaptive_cfg/experiments/robust_adaptive_cfg_v2",
    "research_archive/innovation1/robust_v2",
    [
        ("analyze_experiment.py", "code/analyze_experiment.py"),
        ("run_frozen_pipeline.py", "code/run_frozen_pipeline.py"),
        ("run_generation_and_evaluation.py", "code/run_generation_and_evaluation.py"),
        ("implementation/algorithm_description.md", "config/algorithm_description.md"),
        ("implementation/frozen_manifest.json", "config/frozen_manifest.json"),
        ("implementation/robust_adaptive_cfg_v2_config.yaml", "config/robust_adaptive_cfg_v2_config.yaml"),
        ("calibration/residual_calibration.json", "results/residual_calibration.json"),
        ("calibration/residual_calibration.csv", "results/residual_calibration.csv"),
        ("final_report.md", "results/final_report.md"),
        ("final_status.json", "results/final_status.json"),
        ("p0/final_report.md", "results/p0_final_report.md"),
        ("p0/decision_summary.json", "results/p0_decision_summary.json"),
        ("p0/bootstrap_results.json", "results/bootstrap_results.json"),
        ("p0/harm_metrics.csv", "results/harm_metrics.csv"),
        ("p0/harm_per_seed.csv", "results/harm_per_seed.csv"),
        ("p0/paired_results.csv", "results/paired_results.csv"),
        ("p0/pipeline_status.json", "results/pipeline_status.json"),
        ("p0/seeds.json", "seeds/p0_seeds.json"),
        ("fresh_seed_audit.json", "seeds/fresh_seed_audit.json"),
        ("plots/harm_rate_comparison.png", "figures/harm_rate_comparison.png"),
        ("plots/worst_quartile_comparison.png", "figures/worst_quartile_comparison.png"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle/experiments/counterfactual_adaptive_cfg_v3",
    "research_archive/innovation1/oracle_v3",
    [
        ("analyze_oracle.py", "code/analyze_oracle.py"),
        ("run_predictability.py", "code/run_predictability.py"),
        ("frozen_manifest.json", "config/frozen_manifest.json"),
        ("oracle/oracle_config.yaml", "config/oracle_config.yaml"),
        ("final_report.md", "results/final_report.md"),
        ("final_status.json", "results/final_status.json"),
        ("final_integrity_audit.json", "results/final_integrity_audit.json"),
        ("oracle/oracle_summary.json", "results/oracle_summary.json"),
        ("oracle/oracle_headroom.csv", "results/oracle_headroom.csv"),
        ("oracle/oracle_bootstrap.json", "results/oracle_bootstrap.json"),
        ("oracle/all_outcome_metrics.csv", "results/all_outcome_metrics.csv"),
        ("oracle/oracle_seeds.json", "seeds/oracle_seeds.json"),
        ("fresh_seed_audit.json", "seeds/fresh_seed_audit.json"),
        ("plots/oracle_gain_distribution.png", "figures/oracle_gain_distribution.png"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg/experiments/risk_calibrated_adaptive_cfg",
    "research_archive/innovation1/risk_v4",
    [
        ("run_offline_analysis.py", "code/run_offline_analysis.py"),
        ("prepare_fresh_p0.py", "code/prepare_fresh_p0.py"),
        ("preregistered_protocol.yaml", "config/preregistered_protocol.yaml"),
        ("frozen_manifest.json", "config/frozen_manifest.json"),
        ("final_report.md", "results/final_report.md"),
        ("final_status.json", "results/final_status.json"),
        ("offline/decision_summary.json", "results/decision_summary.json"),
        ("offline/risk_coverage.csv", "results/risk_coverage.csv"),
        ("offline/timestep_comparison.csv", "results/timestep_comparison.csv"),
        ("models/ensemble_results.json", "results/ensemble_results.json"),
        ("splits/split_manifest.json", "seeds/split_manifest.json"),
        ("plots/coverage_vs_harm.png", "figures/coverage_vs_harm.png"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg/experiments/calibrated_safe_selection_cfg",
    "research_archive/innovation1/safe_v5",
    [
        ("run_offline_analysis.py", "code/run_offline_analysis.py"),
        ("analyze_fresh.py", "code/analyze_fresh.py"),
        ("preregistered_protocol.yaml", "config/preregistered_protocol.yaml"),
        ("frozen_manifest.json", "config/frozen_manifest.json"),
        ("calibration/conformal_quantiles.json", "results/conformal_quantiles.json"),
        ("calibration/harm_probability_calibration.json", "results/harm_probability_calibration.json"),
        ("validation/decision_summary.json", "results/decision_summary.json"),
        ("validation/operating_point_results.csv", "results/operating_point_results.csv"),
        ("validation/selected_operating_point.json", "results/selected_operating_point.json"),
        ("final_report.md", "results/final_report.md"),
        ("final_status.json", "results/final_status.json"),
        ("splits/split_manifest.json", "seeds/split_manifest.json"),
        ("plots/harm_probability_calibration.png", "figures/harm_probability_calibration.png"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_stage_cfg/experiments/stage_calibrated_bounded_cfg",
    "research_archive/innovation1/stage_cfg",
    [
        ("analyze_phase.py", "code/analyze_phase.py"),
        ("run_generation.py", "code/run_generation.py"),
        ("preregistered_protocol.yaml", "config/preregistered_protocol.yaml"),
        ("implementation/frozen_stage_cfg.yaml", "config/frozen_stage_cfg.yaml"),
        ("implementation/frozen_stage_cfg_manifest.json", "config/frozen_stage_cfg_manifest.json"),
        ("calibration/decision_summary.json", "results/calibration_decision_summary.json"),
        ("calibration/bootstrap_results.json", "results/calibration_bootstrap.json"),
        ("calibration/final_report.md", "results/calibration_final_report.md"),
        ("p0/decision_summary.json", "results/p0_decision_summary.json"),
        ("p0/bootstrap_results.json", "results/p0_bootstrap.json"),
        ("p0/final_report.md", "results/p0_final_report.md"),
        ("p0/harm_metrics.csv", "results/p0_harm_metrics.csv"),
        ("p0/paired_results.csv", "results/p0_paired_results.csv"),
        ("p0/tail_metrics.csv", "results/p0_tail_metrics.csv"),
        ("calibration/seeds.json", "seeds/calibration_seeds.json"),
        ("p0/seeds.json", "seeds/p0_seeds.json"),
        ("plots/p0_harm_comparison.png", "figures/p0_harm_comparison.png"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg/experiments/field_decoupled_adaptive_cfg",
    "research_archive/innovation1/field_cfg",
    [
        ("analyze_phase.py", "code/analyze_phase.py"),
        ("run_generation.py", "code/run_generation.py"),
        ("preregistered_protocol.yaml", "config/preregistered_protocol.yaml"),
        ("frozen_manifest.json", "config/frozen_manifest.json"),
        ("calibration/decision_summary.json", "results/decision_summary.json"),
        ("calibration/final_report.md", "results/final_report.md"),
        ("calibration/field_mechanism.csv", "results/field_mechanism.csv"),
        ("calibration/field_policy_ranking.csv", "results/field_policy_ranking.csv"),
        ("calibration/harm_metrics.csv", "results/harm_metrics.csv"),
        ("calibration/pareto_front.csv", "results/pareto_front.csv"),
        ("calibration/selected_policy.json", "results/selected_policy.json"),
        ("calibration/seeds.json", "seeds/calibration_seeds.json"),
        ("seed_registry.json", "seeds/seed_registry.json"),
        ("plots/global_vs_field_decoupled.png", "figures/global_vs_field_decoupled.png"),
    ],
)

# Innovation 1 branching lineage. Some small evidence is intentionally repeated
# across concept-level landing pages so each key negative result is standalone.
BRANCHING = "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/reference_preserved_budgeted_cfg"
SELECTIONS += files(BRANCHING, "research_archive/innovation1/branch_oracle", [
    ("run_branch_compatible_audit.py", "code/run_branch_compatible_audit.py"),
    ("branch_compatible_protocol.yaml", "config/branch_compatible_protocol.yaml"),
    ("offline_branchable/branchable_oracle_all_report.md", "results/branchable_oracle_all_report.md"),
    ("offline_branchable/branchable_oracle_all_metrics.csv", "results/branchable_oracle_all_metrics.csv"),
    ("offline_branchable/oracle_branchable_k_results.csv", "results/oracle_branchable_k_results.csv"),
    ("offline_branchable/phase_a_status.json", "results/phase_a_status.json"),
    ("offline_branchable/phase_a_source_manifest.json", "seeds/phase_a_source_manifest.json"),
])
SELECTIONS += files(BRANCHING, "research_archive/innovation1/reference_branching", [
    ("phase_b_sampler.py", "code/phase_b_sampler.py"),
    ("run_feature_generation.py", "code/run_feature_generation.py"),
    ("feature_study_protocol.yaml", "config/feature_study_protocol.yaml"),
    ("pipeline_status.json", "results/pipeline_status.json"),
    ("offline/final_offline_report.md", "results/final_offline_report.md"),
    ("offline/candidate_policy_summary.csv", "results/candidate_policy_summary.csv"),
    ("offline/fixed_k_results.csv", "results/fixed_k_results.csv"),
    ("offline/oracle_k_results.csv", "results/oracle_k_results.csv"),
    ("feature_study/feature_study_seed_manifest.json", "seeds/feature_study_seed_manifest.json"),
])
SELECTIONS += files(BRANCHING, "research_archive/innovation1/phase_b_allocator", [
    ("analyze_feature_study.py", "code/analyze_feature_study.py"),
    ("phase_b_sampler.py", "code/phase_b_sampler.py"),
    ("feature_study/frozen_manifest.json", "config/frozen_manifest.json"),
    ("feature_study/final_report.md", "results/final_report.md"),
    ("feature_study/phase_b_decision_summary.json", "results/phase_b_decision_summary.json"),
    ("feature_study/allocator_test_bootstrap_20k.json", "results/allocator_test_bootstrap_20k.json"),
    ("feature_study/allocator_test_metrics.csv", "results/allocator_test_metrics.csv"),
    ("feature_study/allocator_test_selections.csv", "results/allocator_test_selections.csv"),
    ("feature_study/allocator_validation_results.csv", "results/allocator_validation_results.csv"),
    ("feature_study/feature_study_seed_manifest.json", "seeds/feature_study_seed_manifest.json"),
])
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm/experiments/frozen_linear_k2_confirmation",
    "research_archive/innovation1/frozen_confirmation",
    [
        ("confirmatory_sampler.py", "code/confirmatory_sampler.py"),
        ("analyze_phase.py", "code/analyze_phase.py"),
        ("reconstruct_frozen_allocator.py", "code/reconstruct_frozen_allocator.py"),
        ("protocol/protocol.md", "config/protocol.md"),
        ("protocol/protocol_amendment.md", "config/protocol_amendment.md"),
        ("protocol/frozen_execution_manifest.json", "config/frozen_execution_manifest.json"),
        ("protocol/frozen_method_manifest.json", "config/frozen_method_manifest.json"),
        ("protocol/frozen_linear_k2_parameters.json", "config/frozen_linear_k2_parameters.json"),
        ("protocol/reconstruction_audit.json", "results/reconstruction_audit.json"),
        ("protocol/reconstruction_test_scores.csv", "results/reconstruction_test_scores.csv"),
        ("protocol/reconstruction_validation_scores.csv", "results/reconstruction_validation_scores.csv"),
        ("c1_128/final_report.md", "results/final_report.md"),
        ("c1_128/bootstrap_20k.json", "results/bootstrap_20k.json"),
        ("c1_128/continuation_decision.json", "results/continuation_decision.json"),
        ("c1_128/metrics.csv", "results/metrics.csv"),
        ("c1_128/paired_results.csv", "results/paired_results.csv"),
        ("c1_128/quality_guardrails.json", "results/quality_guardrails.json"),
        ("protocol/seed_manifest_256.json", "seeds/seed_manifest_256.json"),
    ],
)

# Innovation 2: early P0, the three confirmation stages, and mechanism studies.
I2ROOT = "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments"
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance/experiments/closed_loop_force_guidance_p0",
    "research_archive/innovation2/early_force_guidance",
    [
        ("closed_loop_sampler.py", "code/closed_loop_sampler.py"),
        ("config.yaml", "config/config.yaml"),
        ("implementation_lock.json", "config/implementation_lock.json"),
        ("final_report.md", "results/final_report.md"),
        ("decision_summary.json", "results/decision_summary.json"),
        ("paired_bootstrap.json", "results/paired_bootstrap.json"),
        ("paired_results.csv", "results/paired_results.csv"),
        ("quality_metrics.csv", "results/quality_metrics.csv"),
        ("seeds.json", "seeds/seeds.json"),
    ],
)
SELECTIONS += files(f"{I2ROOT}/mattersim_late_force_guidance_p0", "research_archive/innovation2/p0", [
    ("late_clean_sampler.py", "code/late_clean_sampler.py"),
    ("late_force_sampler.py", "code/late_force_sampler.py"),
    ("test_guidance_geometry.py", "code/test_guidance_geometry.py"),
    ("guidance_config.yaml", "config/guidance_config.yaml"),
    ("guidance_manifest.json", "config/guidance_manifest.json"),
    ("p0_config.yaml", "config/p0_config.yaml"),
    ("final_report.md", "results/final_report.md"),
    ("decision_summary.json", "results/decision_summary.json"),
    ("audit_results.json", "results/audit_results.json"),
    ("p0_bootstrap.csv", "results/p0_bootstrap.csv"),
    ("p0_paired_per_seed.csv", "results/p0_paired_per_seed.csv"),
    ("quality_metrics.csv", "results/quality_metrics.csv"),
    ("p0_structures/manifest.json", "seeds/structure_manifest.json"),
])
SELECTIONS += files(f"{I2ROOT}/mattersim_late_force_guidance_formal32", "research_archive/innovation2/formal32", [
    ("analyze_formal32.py", "code/analyze_formal32.py"),
    ("freeze_formal32_protocol.py", "code/freeze_formal32_protocol.py"),
    ("formal32_config.yaml", "config/formal32_config.yaml"),
    ("final_report.md", "results/final_report.md"),
    ("decision_summary.json", "results/decision_summary.json"),
    ("bootstrap_results.json", "results/bootstrap_results.json"),
    ("guardrail_results.json", "results/guardrail_results.json"),
    ("paired_results.csv", "results/paired_results.csv"),
    ("physics_metrics.csv", "results/physics_metrics.csv"),
    ("quality_metrics.csv", "results/quality_metrics.csv"),
    ("formal32_seeds.json", "seeds/formal32_seeds.json"),
    ("plots/maxf_paired_scatter.png", "figures/maxf_paired_scatter.png"),
])
SELECTIONS += files(f"{I2ROOT}/mattersim_late_force_guidance_formal256", "research_archive/innovation2/formal256", [
    ("formal256_config.yaml", "config/formal256_config.yaml"),
    ("implementation_lock.json", "config/implementation_lock.json"),
    ("final_report.md", "results/final_report.md"),
    ("decision_summary.json", "results/decision_summary.json"),
    ("audit_results.json", "results/audit_results.json"),
    ("baseline_maxF_quartiles.json", "results/baseline_maxF_quartiles.json"),
    ("paired_bootstrap.json", "results/paired_bootstrap.json"),
    ("paired_results.csv", "results/paired_results.csv"),
    ("per_structure_metrics.csv", "results/per_structure_metrics.csv"),
    ("quality_metrics.csv", "results/quality_metrics.csv"),
    ("formal256_seeds.json", "seeds/formal256_seeds.json"),
    ("plots/maxF_ecdf_paired.png", "figures/maxF_ecdf_paired.png"),
])
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_independent_mlip_validation",
    "research_archive/innovation2/chgnet_validation",
    [
        ("final_report.md", "results/final_report.md"),
        ("distribution_statistics.csv", "results/distribution_statistics.csv"),
        ("paired_statistics.csv", "results/paired_statistics.csv"),
        ("per_structure_metrics.csv", "results/per_structure_metrics.csv"),
    ],
)
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_thesis_final/thesis_release/innovation2/method",
    "research_archive/innovation2/chgnet_validation",
    [("independent_mlip.py", "code/independent_mlip.py")],
)

for dirname, target in [
    ("mattersim_force_direction_ablation", "direction_ablation"),
    ("mattersim_trust_region_ablation", "trust_region_ablation"),
    ("mattersim_equal_budget_post_generation", "equal_budget_post"),
    ("adaptive_cfg_force_guidance_compatibility", "compatibility"),
]:
    SELECTIONS += files(f"{I2ROOT}/{dirname}", f"research_archive/innovation2/{target}", [
        ("config.yaml", "config/config.yaml"),
        ("implementation_lock.json", "config/implementation_lock.json"),
        ("final_report.md", "results/final_report.md"),
        ("decision_summary.json", "results/decision_summary.json"),
        ("audit_results.json", "results/audit_results.json"),
        ("paired_bootstrap.json", "results/paired_bootstrap.json"),
        ("paired_results.csv", "results/paired_results.csv"),
        ("per_structure_metrics.csv", "results/per_structure_metrics.csv"),
        ("quality_metrics.csv", "results/quality_metrics.csv"),
        ("seeds.json", "seeds/seeds.json"),
    ])
SELECTIONS += files(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_bounded_correction_audit",
    "research_archive/innovation2/bounded_correction",
    [
        ("final_report.md", "results/final_report.md"),
        ("distribution_statistics.csv", "results/distribution_statistics.csv"),
        ("paired_statistics.csv", "results/paired_statistics.csv"),
        ("per_structure_metrics.csv", "results/per_structure_metrics.csv"),
    ],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}


def materialize(source: Path, destination: Path) -> str:
    data = source.read_bytes()
    if source.suffix.lower() in TEXT_SUFFIXES:
        line_normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        normalized = b"\n".join(line.rstrip(b" \t") for line in line_normalized.split(b"\n"))
        destination.write_bytes(normalized)
        return "line_endings_and_trailing_whitespace_normalized" if normalized != data else "none"
    destination.write_bytes(data)
    return "none"


def main() -> int:
    missing = [str(source) for source, _ in SELECTIONS if not source.is_file()]
    if missing:
        raise SystemExit("Missing selected artifacts:\n" + "\n".join(missing))
    rows = []
    for source, destination in SELECTIONS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256(source)
        source_size = source.stat().st_size
        transform = materialize(source, destination)
        dest_hash = sha256(destination)
        if transform == "none" and source_hash != dest_hash:
            raise SystemExit(f"Hash mismatch after copy: {source} -> {destination}")
        rows.append({
            "archive_path": destination.relative_to(ROOT).as_posix(),
            "original_path": str(source),
            "source_size_bytes": source_size,
            "source_sha256": source_hash,
            "transform": transform,
            "archive_size_bytes": destination.stat().st_size,
            "archive_sha256": dest_hash,
        })
    out = ARCHIVE / "selected_artifact_manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "archive_path",
                "original_path",
                "source_size_bytes",
                "source_sha256",
                "transform",
                "archive_size_bytes",
                "archive_sha256",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["archive_path"]))
    print(f"selected_artifacts={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
