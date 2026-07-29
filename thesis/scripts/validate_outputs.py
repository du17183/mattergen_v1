#!/usr/bin/env python3
"""Validate reproducibility, provenance, formats, and claim-safety constraints."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.paths import FIGURE_OUTPUTS, FIGURE_ROOT, REPO_ROOT, TABLE_ROOT
from generate_all import FIGURES


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    figure_records = []
    for stem, _ in FIGURES:
        record = {"stem": stem}
        for ext in ("pdf", "svg", "png"):
            path = FIGURE_OUTPUTS / ext / f"{stem}.{ext}"
            check(path.is_file() and path.stat().st_size > 1000, f"missing/empty figure: {path}", errors)
            record[ext] = str(path.relative_to(REPO_ROOT))
        png = FIGURE_OUTPUTS / "png" / f"{stem}.png"
        if png.is_file():
            with Image.open(png) as image:
                dpi = image.info.get("dpi", (0, 0))
                check(min(dpi) >= 590, f"PNG DPI below 600 tolerance: {stem} {dpi}", errors)
                check(image.mode in {"RGB", "RGBA"}, f"unexpected PNG mode: {stem} {image.mode}", errors)
        svg = FIGURE_OUTPUTS / "svg" / f"{stem}.svg"
        if svg.is_file():
            text = svg.read_text(encoding="utf-8")
            check("<image" not in text, f"raster image embedded in SVG: {stem}", errors)
            check("<svg" in text, f"invalid SVG root: {stem}", errors)
        pdf = FIGURE_OUTPUTS / "pdf" / f"{stem}.pdf"
        if pdf.is_file():
            check(pdf.read_bytes()[:5] == b"%PDF-", f"invalid PDF header: {stem}", errors)
        check((FIGURE_ROOT / "source_data" / f"{stem}.csv").is_file(), f"missing source-data CSV: {stem}", errors)
        check((FIGURE_ROOT / "source" / "python" / f"{stem}.py").is_file(), f"missing figure Python source: {stem}", errors)
        check((FIGURE_ROOT / "captions" / "zh" / f"{stem}.md").is_file(), f"missing zh caption: {stem}", errors)
        check((FIGURE_ROOT / "captions" / "en" / f"{stem}.md").is_file(), f"missing en caption: {stem}", errors)
        figure_records.append(record)

    for stem in (
        "fig01_full_method_architecture",
        "fig02_adaptive_cfg_mechanism",
        "fig03_e3pcr_mechanism",
        "fig04_experiment_lineage",
        "fig12_negative_routes_summary",
    ):
        check((FIGURE_ROOT / "source" / "graphviz" / f"{stem}.dot").is_file(), f"missing DOT: {stem}", errors)

    table_records = []
    for csv_path in sorted((TABLE_ROOT / "csv").glob("*.csv")):
        stem = csv_path.stem
        record = {"stem": stem}
        for folder, ext in (("csv", "csv"), ("markdown", "md"), ("latex", "tex")):
            path = TABLE_ROOT / folder / f"{stem}.{ext}"
            check(path.is_file() and path.stat().st_size > 20, f"missing/empty table: {path}", errors)
            record[ext] = str(path.relative_to(REPO_ROOT))
        check((TABLE_ROOT / "captions" / f"{stem}_zh.md").is_file(), f"missing table zh caption: {stem}", errors)
        check((TABLE_ROOT / "captions" / f"{stem}_en.md").is_file(), f"missing table en caption: {stem}", errors)
        table_records.append(record)
    check(len(table_records) == 10, f"expected 10 table families, found {len(table_records)}", errors)
    workbook = TABLE_ROOT / "xlsx" / "thesis_results.xlsx"
    check(workbook.is_file() and workbook.stat().st_size > 5000, "missing/empty thesis workbook", errors)

    claims_path = REPO_ROOT / "thesis" / "PAPER_CLAIMS_FINAL.json"
    check(claims_path.is_file(), "missing PAPER_CLAIMS_FINAL.json", errors)
    if claims_path.is_file():
        claims = json.loads(claims_path.read_text(encoding="utf-8"))
        check(len(claims["claims"]) == 6, "expected exactly six frozen claims", errors)
        blob = json.dumps(claims, ensure_ascii=False)
        check("DFT_VERIFIED" in blob and "false" in blob.lower(), "missing DFT limitation", errors)
        check("Mixed 256" in blob and "INVALID" in blob, "missing mixed-cohort invalidity", errors)

    source_files = [
        *SCRIPT_DIR.rglob("*.py"),
        *(FIGURE_ROOT / "source").rglob("*.py"),
        *(FIGURE_ROOT / "source").rglob("*.dot"),
    ]
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        check("".join(("/", "data", "/dxl")) not in text, f"server absolute path in source: {path}", errors)
        check("".join(("/", "home", "/ubuntu")) not in text, f"server username/path in source: {path}", errors)

    formal_leakage = False
    source_data_root = FIGURE_ROOT / "source_data"
    for path in source_data_root.glob("*.csv"):
        if path.name != "fig11_leakage_diagnostic.csv":
            text = path.read_text(encoding="utf-8")
            if "training_overlap" in text:
                formal_leakage = True
                errors.append(f"training-overlap leaked into non-diagnostic figure source: {path.name}")

    figure_validation = {
        "valid": not errors,
        "figures_expected": 12,
        "figures_generated": len(figure_records),
        "formats_each": ["pdf", "svg", "png"],
        "png_dpi_minimum": 600,
        "graphviz_sources": 5,
        "source_data_files": len(list(source_data_root.glob("*.csv"))),
        "formal_data_leakage_found": formal_leakage,
        "errors": errors,
        "warnings": warnings,
    }
    table_validation = {
        "valid": not errors,
        "table_families_expected": 10,
        "table_families_generated": len(table_records),
        "workbook": str(workbook.relative_to(REPO_ROOT)) if workbook.exists() else None,
        "errors": errors,
    }
    (FIGURE_ROOT / "validation" / "FIGURE_VALIDATION.json").write_text(
        json.dumps(figure_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (TABLE_ROOT / "validation" / "TABLE_VALIDATION.json").write_text(
        json.dumps(table_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figure_md = [
        "# Figure validation",
        "",
        f"- Valid: `{not errors}`",
        f"- Figures: `{len(figure_records)}/12` in PDF/SVG/PNG",
        f"- Source-data CSV files: `{len(list(source_data_root.glob('*.csv')))}`",
        "- PNG resolution: 600 dpi (590 dpi metadata tolerance)",
        "- SVG raster embedding: forbidden and checked",
        f"- Formal data leakage found: `{formal_leakage}`",
        "",
        "## Errors",
        "",
        *([f"- {item}" for item in errors] or ["- None"]),
    ]
    (FIGURE_ROOT / "validation" / "FIGURE_VALIDATION.md").write_text("\n".join(figure_md) + "\n", encoding="utf-8")
    table_md = [
        "# Table validation",
        "",
        f"- Valid: `{not errors}`",
        f"- Table families: `{len(table_records)}/10`",
        f"- Workbook present: `{workbook.exists()}`",
        "- Every table exported to CSV, Markdown, and LaTeX.",
        "",
        "## Errors",
        "",
        *([f"- {item}" for item in errors] or ["- None"]),
    ]
    (TABLE_ROOT / "validation" / "TABLE_VALIDATION.md").write_text("\n".join(table_md) + "\n", encoding="utf-8")
    if errors:
        for item in errors:
            print(f"ERROR: {item}")
        raise SystemExit(1)
    print("all thesis figures, tables, claims, and portability checks passed")


if __name__ == "__main__":
    main()
