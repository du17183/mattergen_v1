#!/usr/bin/env python3
"""Generate the thesis workbook plus CSV, Markdown, and LaTeX tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.paths import ARCHIVE_ROOT, TABLE_ROOT, ensure_output_directories
from generate_statistical_figures import NEGATIVE_ROUTES


def load_csv(group: str) -> pd.DataFrame:
    return pd.read_csv(ARCHIVE_ROOT / "data" / group / "per_seed_metrics.csv")


def load_json(relative: str):
    with (ARCHIVE_ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(relative: str):
    with (ARCHIVE_ROOT / relative).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, rule, *rows]) + "\n"


def fmt_float(value, digits=6):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}g}"


def build_tables() -> dict[str, pd.DataFrame]:
    i1 = load_csv("innovation1")
    i2 = load_csv("innovation2")
    c1 = load_csv("compatibility_1")
    c2 = load_csv("compatibility_2")
    leakage = load_csv("leakage_diagnostic")
    cfg = load_yaml("configs/adaptive_cfg_final.yaml")
    e3 = load_yaml("configs/e3_pcr_final.yaml")
    evaluation = load_yaml("configs/evaluation_final.yaml")
    formal = load_json("reports/innovation2/final_summary.json")
    i1_report = load_json("reports/innovation1/formal_final_report.json")
    c1_stats = pd.read_csv(ARCHIVE_ROOT / "reports" / "compatibility" / "paired_statistics.csv")
    c2_stats = pd.read_csv(ARCHIVE_ROOT / "reports" / "replication" / "paired_statistics.csv")
    c1_force = c1_stats[c1_stats.metric == "pre_relax_max_force_ev_ang"].iloc[0]
    c2_force = c2_stats[c2_stats.metric == "pre_relax_max_force_ev_ang"].iloc[0]
    mech = formal["mechanism"]
    behavior = mech["behavior"]

    manifest = pd.DataFrame(
        [
            ["Innovation 1", "C0", "20000–20255", 256, "official dft_mag_density", cfg["code_commit"], evaluation["evaluator"], "Formal"],
            ["Innovation 2", "C0", "40000–40255", 256, e3["q3_checkpoint_sha256"], formal["formal_code_commit"], evaluation["evaluator"], "Formal independent"],
            ["Combination cohort 1", "A0", "41000–41063", 64, e3["q3_checkpoint_sha256"], "see source manifest", evaluation["evaluator"], "Independent replication"],
            ["Combination cohort 2", "A0", "50000–50063", 64, e3["q3_checkpoint_sha256"], "see source manifest", evaluation["evaluator"], "Independent replication"],
            ["Leakage diagnostic", "A0", "20000–20255", 256, e3["q3_checkpoint_sha256"], "diagnostic lineage", evaluation["evaluator"], "Diagnostic; mixed invalid"],
        ],
        columns=["method", "baseline", "seed_range", "n", "checkpoint", "commit", "evaluator", "data_qualification"],
    )

    i1_table = pd.DataFrame(
        [
            ["E-hull (eV/atom)", i1.c0_ehull.mean(), i1.a0_ehull.mean(), (i1.a0_ehull - i1.c0_ehull).mean(), "not significant"],
            ["Stable", i1.c0_stable.mean(), i1.a0_stable.mean(), (i1.a0_stable.astype(int) - i1.c0_stable.astype(int)).mean(), "not significant"],
            ["NUS", i1.c0_nus.mean(), i1.a0_nus.mean(), (i1.a0_nus.astype(int) - i1.c0_nus.astype(int)).mean(), "not significant"],
            ["Novel", i1.c0_novel.mean(), i1.a0_novel.mean(), (i1.a0_novel.astype(int) - i1.c0_novel.astype(int)).mean(), "descriptive"],
            ["Unique", i1.c0_unique.mean(), i1.a0_unique.mean(), (i1.a0_unique.astype(int) - i1.c0_unique.astype(int)).mean(), "descriptive"],
            ["Composition validity", i1.c0_composition_valid.mean(), i1.a0_composition_valid.mean(), (i1.a0_composition_valid.astype(int) - i1.c0_composition_valid.astype(int)).mean(), "descriptive"],
            ["Structure validity", i1.c0_structure_valid.mean(), i1.a0_structure_valid.mean(), (i1.a0_structure_valid.astype(int) - i1.c0_structure_valid.astype(int)).mean(), "descriptive"],
        ],
        columns=["metric", "C0", "Adaptive_CFG_A0", "A0_minus_C0", "inference"],
    )

    i2_table = pd.DataFrame(
        [
            [
                name,
                frame[maxf].mean(),
                frame[rmsd].mean(),
                frame[ehull].mean(),
                frame[stable].mean(),
                frame[nus].mean(),
                frame[novel].mean(),
                frame[unique].mean(),
            ]
            for name, frame, maxf, rmsd, ehull, stable, nus, novel, unique in [
                ("C0", i2, "c0_max_force", "c0_rmsd", "c0_ehull", "c0_stable", "c0_nus", "c0_novel", "c0_unique"),
                ("Always-on E3-A", i2, "e3a_max_force", "e3a_rmsd", "e3a_ehull", "e3a_stable", "e3a_nus", "e3a_novel", "e3a_unique"),
                ("Learned-gated E3-G", i2, "e3g_max_force", "e3g_rmsd", "e3g_ehull", "e3g_stable", "e3g_nus", "e3g_novel", "e3g_unique"),
            ]
        ],
        columns=["method", "max_force_ev_ang", "RMSD_ang", "E_hull_ev_atom", "Stable", "NUS", "Novel", "Unique"],
    )

    gate = pd.DataFrame(
        [
            ["Refinement rate", behavior["E3-A"]["refinement_rate"], behavior["E3-G"]["refinement_rate"], "fraction"],
            ["Harm rate", mech["e3a_harm_rate"], mech["e3g_harm_rate"], "fraction"],
            ["Low-force harm rate", mech["low_force_e3a_harm_rate"], mech["low_force_e3g_harm_rate"], "fraction"],
            ["Mean displacement", behavior["E3-A"]["mean_displacement_angstrom"], behavior["E3-G"]["mean_displacement_angstrom"], "angstrom"],
            ["P95 displacement", behavior["E3-A"]["p95_displacement_angstrom"], behavior["E3-G"]["p95_displacement_angstrom"], "angstrom"],
            ["Gain retention", 1.0, mech["gain_retention"], "fraction"],
        ],
        columns=["metric", "Always_on", "Learned_gated", "unit"],
    )

    def cohort_table(name, seeds, data, base_col, selected_col, row, semantic_wtl):
        return pd.DataFrame(
            [
                {
                    "cohort": name,
                    "seed_range": seeds,
                    "n": int(row.paired_count),
                    "A0_force": data[base_col].mean(),
                    "A0_plus_E3G_force": data[selected_col].mean(),
                    "mean_difference": row.mean_difference,
                    "relative_change": data[selected_col].mean() / data[base_col].mean() - 1,
                    "bootstrap_95_CI": f"[{row.bootstrap_95_ci_low:.6f}, {row.bootstrap_95_ci_high:.6f}]",
                    "p_value": row.p_value,
                    "Win_Tie_Loss_algorithmic": semantic_wtl,
                }
            ]
        )

    cohort1 = cohort_table("Cohort 1", "41000–41063", c1, "a0_max_force", "a0_e3g_max_force", c1_force, "34/19/11")
    cohort2 = cohort_table("Cohort 2", "50000–50063", c2, "a0_max_force", "a0_e3g_max_force", c2_force, "35/18/11")
    combo = pd.concat([cohort1, cohort2], ignore_index=True)

    leakage_table = (
        leakage.groupby("cohort", as_index=False)
        .agg(
            n=("seed", "size"),
            mean_force_difference=("force_difference", "mean"),
            harm_count=("refinement_harm", "sum"),
            harm_rate=("refinement_harm", "mean"),
            valid_for_formal_claims=("valid_for_formal_claims", "all"),
            valid_for_supplementary_claims=("valid_for_supplementary_claims", "all"),
        )
    )
    leakage_table["qualification"] = np.where(
        leakage_table.cohort == "training_overlap",
        "Diagnostic only; INVALID for independent claims",
        "Supplementary held-out only",
    )

    negative = pd.DataFrame(NEGATIVE_ROUTES, columns=["method", "goal", "No_Go_reason", "status"])
    negative["main_benefit"] = [
        "Small throughput gain",
        "Real ~1.5× speed",
        "Moderate speed",
        "RMSD/NUS direction",
        "Base reproduction attempt",
        "Positive offline direction",
        "Teacher infrastructure",
        "Quality predictor",
        "Force reduction",
        "Constrained correction",
        "Selection signal",
        "Ranking signal",
        "Persistent workers",
    ]
    negative["paper_usage"] = "Negative-result motivation / boundary"
    negative["branch"] = "See root README branch map"
    negative["commit"] = "See thesis_archive/EXPERIMENT_LINEAGE.md"

    qualifications = pd.DataFrame(
        [
            ["Adaptive CFG formal256", True, False, False, False, "Positive overall trends", "Statistically significant improvement"],
            ["E3-PCR formal256", True, False, False, False, "Independent force reduction with surrogate-quality preservation", "DFT-verified stability"],
            ["Compatibility cohort 1", True, False, False, False, "Independent positive combination result", "Pooled pre-registered 128-seed result"],
            ["Compatibility cohort 2", True, False, False, False, "Independent replication with smaller effect", "Hide cohort heterogeneity"],
            ["Held-out leakage cohort", False, True, True, False, "Leakage safety diagnostic", "Primary method validation"],
            ["Training-overlap cohort", False, False, True, True, "Shows safety inflation risk", "Independent validation"],
            ["Mixed 256", False, False, True, True, "Diagnostic aggregation only", "Any independent claim"],
        ],
        columns=["result", "formal", "supplementary", "diagnostic", "invalid", "allowed_claim", "forbidden_claim"],
    )

    claims = pd.DataFrame(
        [
            ["C1", "Adaptive CFG positive E-hull/Stable/NUS trends; paired tests not significant", 256, "Formal with limitation"],
            ["C2", "Learned-gated E3-PCR max-force reduction 23.28%", 256, "Formal independent"],
            ["C3", "Gate lowers harmful intervention while retaining 80.657% gain", 256, "Formal mechanism"],
            ["C4", "Combination cohort 1 max-force reduction 27.10%", 64, "Independent"],
            ["C5", "Combination cohort 2 max-force reduction 19.02%", 64, "Independent replication"],
            ["C6", "Training overlap inflates apparent gate safety", 256, "Diagnostic only"],
        ],
        columns=["claim_id", "claim_summary", "n", "qualification"],
    )

    return {
        "01_Experiment_Manifest": manifest,
        "02_Innovation1": i1_table,
        "03_Innovation2": i2_table,
        "04_Gate_Ablation": gate,
        "05_Compatibility_Cohort1": cohort1,
        "06_Compatibility_Cohort2": cohort2,
        "07_Combination_Summary": combo,
        "08_Leakage_Diagnostic": leakage_table,
        "09_Negative_Results": negative,
        "10_Paper_Claims": claims,
    }


CAPTIONS = {
    "01_Experiment_Manifest": (
        "表1 实验设置与证据资格。所有评价均使用 MatterSim-5M 代理势；未开展 DFT 验证。",
        "Table 1. Experimental settings and evidence qualification. All evaluation uses the MatterSim-5M surrogate; no DFT verification is included.",
    ),
    "02_Innovation1": (
        "表2 创新点一正式 256-seed 结果。正向变化未达到配对统计显著性。",
        "Table 2. Innovation 1 formal 256-seed results. Positive changes did not reach paired statistical significance.",
    ),
    "03_Innovation2": (
        "表3 创新点二三臂正式比较。力、松弛和稳定性均由 MatterSim-5M 代理势评价。",
        "Table 3. Three-arm formal comparison for Innovation 2. Force, relaxation, and stability use MatterSim-5M.",
    ),
    "04_Gate_Ablation": (
        "表4 Learned Gate 机制消融。Gate 降低干预覆盖和伤害率，但平均降力不优于 Always-on。",
        "Table 4. Learned Gate mechanism ablation. Gating reduces coverage and harm, but not mean force more than Always-on.",
    ),
    "05_Compatibility_Cohort1": (
        "表5a 第一组独立组合验证（41000–41063，n=64）。",
        "Table 5a. Independent combination cohort 1 (41000–41063, n=64).",
    ),
    "06_Compatibility_Cohort2": (
        "表5b 第二组独立组合复现（50000–50063，n=64）。",
        "Table 5b. Independent combination replication cohort 2 (50000–50063, n=64).",
    ),
    "07_Combination_Summary": (
        "表5 两次独立组合验证并列汇总；不合并为预注册 128-seed 实验。",
        "Table 5. Side-by-side independent combination cohorts; they are not pooled as a preregistered 128-seed experiment.",
    ),
    "08_Leakage_Diagnostic": (
        "表6 训练泄漏诊断。Mixed 256 明确不具备独立结论资格。",
        "Table 6. Training-leakage diagnostic. The mixed 256 cohort is explicitly invalid for independent claims.",
    ),
    "09_Negative_Results": (
        "表7 代表性负面路线及停止证据，不构造统一评分。",
        "Table 7. Representative negative routes and stopping evidence; no synthetic unified score is used.",
    ),
    "10_Paper_Claims": (
        "表8 最终论文结论与证据资格。所有稳定性结论均为 MatterSim-5M 代理结果。",
        "Table 8. Final paper claims and evidence qualification. All stability claims are MatterSim-5M surrogate results.",
    ),
}


def main() -> None:
    ensure_output_directories()
    tables = build_tables()
    workbook = TABLE_ROOT / "xlsx" / "thesis_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)
            worksheet = writer.sheets[sheet[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(55, max(12, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    for name, frame in tables.items():
        stem = name.lower()
        frame.to_csv(TABLE_ROOT / "csv" / f"{stem}.csv", index=False)
        (TABLE_ROOT / "markdown" / f"{stem}.md").write_text(markdown_table(frame), encoding="utf-8")
        (TABLE_ROOT / "latex" / f"{stem}.tex").write_text(
            frame.to_latex(index=False, escape=True, float_format=lambda x: fmt_float(x)),
            encoding="utf-8",
        )
        zh, en = CAPTIONS[name]
        (TABLE_ROOT / "captions" / f"{stem}_zh.md").write_text(zh + "\n", encoding="utf-8")
        (TABLE_ROOT / "captions" / f"{stem}_en.md").write_text(en + "\n", encoding="utf-8")
    print(f"generated {len(tables)} table families and {workbook}")


if __name__ == "__main__":
    main()
