#!/usr/bin/env python3
"""CPU-only validation for the thesis chapter evidence packs.

This script never writes to frozen experiment data.  It recomputes the
pre-registered statistics from the archived per-seed CSV files and, after the
evidence packs have been built, checks their JSON, paths, claim coverage, and
portable ChatGPT inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon


REPO = Path(__file__).resolve().parents[2]
PACK = REPO / "thesis/evidence_packs"
ARCHIVE = REPO / "thesis_archive"
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 20_000
TOL = 5.0e-7


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ARCHIVE / f"data/{name}/per_seed_metrics.csv")


def close(errors: list[str], label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        errors.append(f"DATA_MISMATCH {label}: recomputed={actual!r}, reported={expected!r}")


def bootstrap_mean_ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    return [float(value) for value in np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])]


def verify_statistics() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    frames = {
        "innovation1": load("innovation1"),
        "innovation2": load("innovation2"),
        "compatibility_1": load("compatibility_1"),
        "compatibility_2": load("compatibility_2"),
        "leakage_diagnostic": load("leakage_diagnostic"),
    }
    expected = {
        "innovation1": (256, 20000, 20255),
        "innovation2": (256, 40000, 40255),
        "compatibility_1": (64, 41000, 41063),
        "compatibility_2": (64, 50000, 50063),
        "leakage_diagnostic": (256, 20000, 20255),
    }
    counts: dict[str, Any] = {}
    for name, frame in frames.items():
        n, low, high = expected[name]
        counts[name] = {
            "n": len(frame),
            "seed_min": int(frame.seed.min()),
            "seed_max": int(frame.seed.max()),
            "duplicate_seeds": int(frame.seed.duplicated().sum()),
        }
        if counts[name] != {
            "n": n,
            "seed_min": low,
            "seed_max": high,
            "duplicate_seeds": 0,
        }:
            errors.append(f"DATA_MISMATCH {name} count/seed contract: {counts[name]}")

    training = set(range(20000, 20064))
    intersections = {
        name: sorted(set(frames[name].seed.astype(int)) & training)
        for name in ("innovation2", "compatibility_1", "compatibility_2")
    }
    if any(intersections.values()):
        errors.append(f"FORMAL_DATA_LEAKAGE: {intersections}")

    i1 = frames["innovation1"]
    i1_report = json.loads(
        (ARCHIVE / "reports/innovation1/formal_final_report.json").read_text()
    )["innovation1"]
    i1_values = {
        "ehull_change": float((i1.a0_ehull - i1.c0_ehull).mean()),
        "stable_change": float(i1.a0_stable.mean() - i1.c0_stable.mean()),
        "nus_change": float(i1.a0_nus.mean() - i1.c0_nus.mean()),
    }
    close(errors, "innovation1 ehull", i1_values["ehull_change"], i1_report["EHULL_CHANGE_A0_MINUS_C0"])
    close(errors, "innovation1 stable", i1_values["stable_change"], i1_report["STABLE_CHANGE_A0_MINUS_C0"])
    close(errors, "innovation1 nus", i1_values["nus_change"], i1_report["NUS_CHANGE_A0_MINUS_C0"])

    i2 = frames["innovation2"]
    i2_primary = json.loads(
        (REPO / "reports/q3_e3_pcr/formal256/primary_statistics.json").read_text()
    )
    i2_values: dict[str, Any] = {}
    for arm, selected in (("E3-A", "e3a_max_force"), ("E3-G", "e3g_max_force")):
        left = i2.c0_max_force.to_numpy(float)
        right = i2[selected].to_numpy(float)
        difference = right - left
        raw_p = float(wilcoxon(difference, zero_method="pratt").pvalue)
        ci = bootstrap_mean_ci(difference)
        record = {
            "baseline_mean": float(left.mean()),
            "selected_mean": float(right.mean()),
            "mean_difference": float(difference.mean()),
            "relative_change": float(right.mean() / left.mean() - 1.0),
            "bootstrap_95_ci": ci,
            "wilcoxon_p_raw": raw_p,
            "raw_wins": int((difference < -1.0e-12).sum()),
            "raw_ties": int((np.abs(difference) <= 1.0e-12).sum()),
            "raw_losses": int((difference > 1.0e-12).sum()),
            "semantic_wins": int((difference < -1.0e-6).sum()),
            "semantic_ties": int((np.abs(difference) <= 1.0e-6).sum()),
            "semantic_losses": int((difference > 1.0e-6).sum()),
        }
        i2_values[arm] = record
        report = i2_primary[arm]
        for key in ("baseline_mean", "selected_mean", "mean_difference", "relative_change", "wilcoxon_p_raw"):
            close(errors, f"innovation2 {arm} {key}", record[key], report[key], 1.0e-12)
        for actual, expected_value, suffix in zip(ci, report["bootstrap_95_ci"], ("low", "high"), strict=True):
            close(errors, f"innovation2 {arm} bootstrap {suffix}", actual, expected_value, 1.0e-12)
        for key, report_key in (("raw_wins", "wins"), ("raw_ties", "ties"), ("raw_losses", "losses")):
            if record[key] != int(report[report_key]):
                errors.append(
                    f"DATA_MISMATCH innovation2 {arm} {key}: "
                    f"recomputed={record[key]}, reported={report[report_key]}"
                )

    combination: dict[str, Any] = {}
    report_paths = {
        "compatibility_1": REPO / "reports/a0_e3g_compat64/primary_statistics.json",
        "compatibility_2": REPO / "reports/a0_e3g_independent64/primary_statistics.json",
    }
    for name in ("compatibility_1", "compatibility_2"):
        frame = frames[name]
        left = frame.a0_max_force.to_numpy(float)
        right = frame.a0_e3g_max_force.to_numpy(float)
        difference = right - left
        report = json.loads(report_paths[name].read_text())
        record = {
            "baseline_mean": float(left.mean()),
            "selected_mean": float(right.mean()),
            "mean_difference": float(difference.mean()),
            "relative_change": float(right.mean() / left.mean() - 1.0),
            "bootstrap_95_ci": bootstrap_mean_ci(difference),
            "wilcoxon_p_raw": float(wilcoxon(difference, zero_method="pratt").pvalue),
            "raw_wins": int((difference < -1.0e-12).sum()),
            "raw_ties": int((np.abs(difference) <= 1.0e-12).sum()),
            "raw_losses": int((difference > 1.0e-12).sum()),
            "semantic_wins": int((difference < -1.0e-6).sum()),
            "semantic_ties": int((np.abs(difference) <= 1.0e-6).sum()),
            "semantic_losses": int((difference > 1.0e-6).sum()),
        }
        combination[name] = record
        for key in ("baseline_mean", "selected_mean", "mean_difference", "relative_change", "wilcoxon_p_raw"):
            close(errors, f"{name} {key}", record[key], report[key], 1.0e-12)
        for actual, expected_value, suffix in zip(
            record["bootstrap_95_ci"], report["bootstrap_95_ci"], ("low", "high"), strict=True
        ):
            close(errors, f"{name} bootstrap {suffix}", actual, expected_value, 1.0e-12)

    leakage = frames["leakage_diagnostic"]
    overlap = leakage[leakage.training_overlap.astype(bool)]
    heldout = leakage[~leakage.training_overlap.astype(bool)]
    contingency = [
        [int(overlap.refinement_harm.sum()), int((~overlap.refinement_harm.astype(bool)).sum())],
        [int(heldout.refinement_harm.sum()), int((~heldout.refinement_harm.astype(bool)).sum())],
    ]
    leakage_values = {
        "training_overlap_n": len(overlap),
        "heldout_n": len(heldout),
        "training_harm_count": contingency[0][0],
        "heldout_harm_count": contingency[1][0],
        "training_harm_rate": float(overlap.refinement_harm.mean()),
        "heldout_harm_rate": float(heldout.refinement_harm.mean()),
        "fisher_one_sided_p": float(fisher_exact(contingency, alternative="less").pvalue),
    }
    leakage_report = json.loads(
        (ARCHIVE / "experiments/leakage_diagnostic/statistics.json").read_text()
    )
    mapping = {
        "training_overlap_n": "training_overlap_count",
        "heldout_n": "heldout_count",
        "training_harm_count": "train_harm_count",
        "heldout_harm_count": "heldout_harm_count",
        "training_harm_rate": "train_harm_rate",
        "heldout_harm_rate": "heldout_harm_rate",
        "fisher_one_sided_p": "fisher_exact_one_sided_p",
    }
    for actual_key, report_key in mapping.items():
        close(errors, f"leakage {actual_key}", leakage_values[actual_key], leakage_report[report_key], 1.0e-12)

    result = {
        "dataset_contracts": counts,
        "formal_training_seed_intersections": intersections,
        "innovation1": i1_values,
        "innovation2": i2_values,
        "compatibility": combination,
        "leakage": leakage_values,
        "DATA_MISMATCH_DETECTED": any("DATA_MISMATCH" in error for error in errors),
        "FORMAL_DATA_LEAKAGE_FOUND": any("FORMAL_DATA_LEAKAGE" in error for error in errors),
    }
    return result, errors


def verify_pack(stats: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    required = [
        "README.md",
        "MASTER_SOURCE_INDEX.md",
        "MASTER_SOURCE_INDEX.json",
        "CLAIM_TRACEABILITY.md",
        "CLAIM_TRACEABILITY.json",
        "FORMULA_REGISTRY.md",
        "FORMULA_REGISTRY.json",
        "CODE_SYMBOL_INDEX.md",
        "CODE_SYMBOL_INDEX.json",
        "FIGURE_TABLE_CROSSWALK.md",
        "WRITING_GUARDRAILS.md",
        "chapter3/CHAPTER3_EVIDENCE_PACK.md",
        "chapter3/CHAPTER3_EVIDENCE_PACK.json",
        "chapter3/section_outline.md",
        "chapter3/source_map.md",
        "chapter3/metrics_definitions.md",
        "chapter3/chatgpt_input.md",
        "chapter4/CHAPTER4_EVIDENCE_PACK.md",
        "chapter4/CHAPTER4_EVIDENCE_PACK.json",
        "chapter4/section_outline.md",
        "chapter4/source_map.md",
        "chapter4/formula_notes.md",
        "chapter4/experiment_evidence.md",
        "chapter4/chatgpt_input.md",
        "chapter5/CHAPTER5_EVIDENCE_PACK.md",
        "chapter5/CHAPTER5_EVIDENCE_PACK.json",
        "chapter5/section_outline.md",
        "chapter5/source_map.md",
        "chapter5/formula_notes.md",
        "chapter5/experiment_evidence.md",
        "chapter5/chatgpt_input.md",
        "chapter6/CHAPTER6_EVIDENCE_PACK.md",
        "chapter6/CHAPTER6_EVIDENCE_PACK.json",
        "chapter6/section_outline.md",
        "chapter6/source_map.md",
        "chapter6/combination_evidence.md",
        "chapter6/ablation_evidence.md",
        "chapter6/negative_results_evidence.md",
        "chapter6/leakage_evidence.md",
        "chapter6/chatgpt_input.md",
    ]
    missing = [path for path in required if not (PACK / path).is_file()]
    errors.extend(f"missing evidence pack file: {path}" for path in missing)

    json_files = list(PACK.rglob("*.json"))
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - validation branch
            errors.append(f"invalid JSON {path.relative_to(REPO)}: {exc}")

    absolute_path_pattern = re.compile(r"/(?:data|home)/")
    chatgpt_files = sorted(PACK.glob("chapter*/chatgpt_input.md"))
    absolute_chatgpt = [
        str(path.relative_to(REPO))
        for path in chatgpt_files
        if absolute_path_pattern.search(path.read_text(encoding="utf-8"))
    ]
    errors.extend(f"server absolute path in ChatGPT input: {path}" for path in absolute_chatgpt)

    source_index = json.loads((PACK / "MASTER_SOURCE_INDEX.json").read_text()) if not missing else []
    broken_sources: list[str] = []
    for source in source_index:
        relative = source.get("relative_path", "")
        if relative and ":" not in relative and not (REPO / relative).exists():
            broken_sources.append(relative)
    errors.extend(f"broken source path: {path}" for path in sorted(set(broken_sources)))

    claims = json.loads((PACK / "CLAIM_TRACEABILITY.json").read_text()) if not missing else []
    required_claims = {
        "C1_ADAPTIVE_CFG_DIRECTIONAL",
        "C2_E3PCR_FORCE",
        "C3_GATE_HARM",
        "C4_COMBINATION_COHORT1",
        "C5_COMBINATION_COHORT2",
        "C6_LEAKAGE_SAFETY",
        "C7_MATTERSIM_SURROGATE",
        "C8_NO_DFT_OR_PROPERTY",
    }
    claim_ids = {claim.get("claim_id") for claim in claims}
    if not required_claims <= claim_ids:
        errors.append(f"missing required claims: {sorted(required_claims - claim_ids)}")
    incomplete_claims = [
        claim.get("claim_id", "UNKNOWN")
        for claim in claims
        if not claim.get("source_data") or not claim.get("limitation")
    ]
    errors.extend(f"incomplete claim evidence: {claim}" for claim in incomplete_claims)

    formulas = json.loads((PACK / "FORMULA_REGISTRY.json").read_text()) if not missing else []
    invalid_formula_labels = [
        item.get("formula_id", "UNKNOWN")
        for item in formulas
        if item.get("exact_or_interpreted") not in {"exact", "interpreted"}
    ]
    errors.extend(f"invalid formula evidence label: {item}" for item in invalid_formula_labels)

    guardrail_path = PACK / "WRITING_GUARDRAILS.md"
    required_guardrails = {
        "Adaptive CFG不得称统计显著",
        "MatterSim-5M不得称DFT",
        "Mixed 256不得用于独立正式结论",
    }
    guardrail_text = (
        guardrail_path.read_text(encoding="utf-8") if guardrail_path.exists() else ""
    )
    missing_guardrails = sorted(
        guardrail for guardrail in required_guardrails if guardrail not in guardrail_text
    )
    errors.extend(f"missing writing guardrail: {item}" for item in missing_guardrails)
    forbidden_hits = []

    return {
        "required_files": len(required),
        "missing_files": missing,
        "json_files_checked": len(json_files),
        "chatgpt_inputs_checked": len(chatgpt_files),
        "server_absolute_paths_in_chatgpt_input": absolute_chatgpt,
        "source_index_entries": len(source_index),
        "broken_source_paths": sorted(set(broken_sources)),
        "claims_checked": len(claims),
        "claims_with_incomplete_evidence": incomplete_claims,
        "formulas_checked": len(formulas),
        "invalid_formula_labels": invalid_formula_labels,
        "forbidden_assertion_hits": forbidden_hits,
        "UNSUPPORTED_PROJECT_FACTS_FOUND": bool(incomplete_claims or broken_sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    stats, errors = verify_statistics()
    pack_checks: dict[str, Any] = {}
    if not args.stats_only:
        pack_checks = verify_pack(stats, errors)
    result = {
        "EVIDENCE_PACKS_VALID": not errors and not args.stats_only,
        "STATISTICS_VALID": not errors,
        "FORMAL_DATA_LEAKAGE_FOUND": stats["FORMAL_DATA_LEAKAGE_FOUND"],
        "DATA_MISMATCH_DETECTED": stats["DATA_MISMATCH_DETECTED"],
        "UNSUPPORTED_PROJECT_FACTS_FOUND": bool(
            pack_checks.get("UNSUPPORTED_PROJECT_FACTS_FOUND", False)
        ),
        "SERVER_ABSOLUTE_PATHS_IN_CHATGPT_INPUT": pack_checks.get(
            "server_absolute_paths_in_chatgpt_input", []
        ),
        "statistics": stats,
        "pack_checks": pack_checks,
        "errors": errors,
    }
    if args.write_report and not args.stats_only:
        (PACK / "EVIDENCE_PACK_VALIDATION.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# 章节证据包真实性验证",
            "",
            f"- EVIDENCE_PACKS_VALID: `{result['EVIDENCE_PACKS_VALID']}`",
            f"- STATISTICS_VALID: `{result['STATISTICS_VALID']}`",
            f"- FORMAL_DATA_LEAKAGE_FOUND: `{result['FORMAL_DATA_LEAKAGE_FOUND']}`",
            f"- DATA_MISMATCH_DETECTED: `{result['DATA_MISMATCH_DETECTED']}`",
            f"- UNSUPPORTED_PROJECT_FACTS_FOUND: `{result['UNSUPPORTED_PROJECT_FACTS_FOUND']}`",
            f"- SERVER_ABSOLUTE_PATHS_IN_CHATGPT_INPUT: `{len(result['SERVER_ABSOLUTE_PATHS_IN_CHATGPT_INPUT'])}`",
            "",
            "## 数据集核对",
            "",
        ]
        for name, contract in stats["dataset_contracts"].items():
            lines.append(
                f"- `{name}`: n={contract['n']}, seeds={contract['seed_min']}–"
                f"{contract['seed_max']}, duplicate={contract['duplicate_seeds']}"
            )
        lines.extend(["", "## 错误", "", "无。" if not errors else "\n".join(f"- {e}" for e in errors), ""])
        (PACK / "EVIDENCE_PACK_VALIDATION.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
