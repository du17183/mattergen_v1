#!/usr/bin/env python3
"""One-command, CPU-only generation of all thesis figures and tables."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common.paths import FIGURE_OUTPUTS, FIGURE_ROOT, TABLE_ROOT, ensure_output_directories
import generate_architecture_figures
import generate_statistical_figures
import generate_tables


FIGURES = [
    ("fig01_full_method_architecture", "Full two-innovation method architecture"),
    ("fig02_adaptive_cfg_mechanism", "Adaptive CFG mechanism"),
    ("fig03_e3pcr_mechanism", "Learned-gated E3-PCR mechanism"),
    ("fig04_experiment_lineage", "Experiment and evidence lineage"),
    ("fig05_adaptive_cfg_results", "Adaptive CFG formal results"),
    ("fig06_e3pcr_force_formal256", "E3-PCR three-arm formal comparison"),
    ("fig07_gate_safety_ablation", "Learned Gate safety ablation"),
    ("fig08_gate_confidence_force_gain", "Gate confidence and force gain"),
    ("fig09_combination_replication_forest", "Independent combination cohorts"),
    ("fig10_independent64_pairplot", "Newest independent 64-seed paired result"),
    ("fig11_leakage_diagnostic", "Training-overlap diagnostic"),
    ("fig12_negative_routes_summary", "Representative No-Go routes"),
]

SURROGATE_ZH = "所有原子力、结构松弛和稳定性指标均由 MatterSim-5M 代理势评价，本文未开展 DFT 验证。"
SURROGATE_EN = "All force, relaxation, and stability metrics are evaluated using the MatterSim-5M surrogate potential. No DFT verification is included."

CAPTIONS_ZH = {
    "fig01_full_method_architecture": "图1 完整方法架构。目标 dft_mag_density 条件经过带完整 Predictor/Corrector 的 MatterGen 与多字段残差驱动 Adaptive CFG，生成晶体后由 Learned Gate 决定执行 E3-PCR 或精确回退。MatterSim-5M 仅用于代理评价。创新点一位于采样阶段，创新点二位于后处理阶段。" + SURROGATE_ZH,
    "fig02_adaptive_cfg_mechanism": "图2 Adaptive CFG 机制。条件与无条件分支形成 cell、position、atom 三字段残差，经 EMA 平滑、残差驱动尺度更新及 [0,5] 限幅后完成 CFG 融合；Predictor 和 Corrector 均不跳过。本方法不是 Corrector Gating。",
    "fig03_e3pcr_mechanism": "图3 Learned-Gated E3-PCR。14 维风险特征输入 129 参数 Gate，阈值为 0.5；Gate-on 时执行 5 步位置等变精修、逐步半径与累计 trust region、回溯和安全检查，Gate-off 或拒绝时精确回退。原子种类和晶胞均保持不变。",
    "fig04_experiment_lineage": "图4 实验与数据血缘。图区分正式 256-seed、两次独立 64-seed 复现、补充/诊断证据以及不具备独立结论资格的 Mixed 256；证据类别同时使用文字与线型编码，避免仅依赖颜色。",
    "fig05_adaptive_cfg_results": "图5 Adaptive CFG 正式效果（seeds 20000–20255，n=256）。E-hull 平均变化为 −0.003435 eV/atom，Stable 和 NUS 分别提高 5.859 与 3.516 个百分点；配对 bootstrap 95% CI 均跨越零，Holm 校正后未达到显著性，故只解释为总体正向趋势。" + SURROGATE_ZH,
    "fig06_e3pcr_force_formal256": "图6 E3-PCR 独立正式三臂比较（seeds 40000–40255，n=256）。Learned-gated E3-G 将平均预松弛最大力从 0.342964 降至 0.263107 eV/Å（−23.28%），配对均值差 bootstrap 95% CI 为 [−0.144966,−0.032453]，Holm 校正 p=4.19×10⁻¹⁰，原始差值 Win/Tie/Loss=163/0/93。" + SURROGATE_ZH,
    "fig07_gate_safety_ablation": "图7 Learned Gate 安全机制（n=256）。与 Always-on 相比，精修覆盖率由 100% 降至 66.406%，总体伤害率由 25.391% 降至 18.359%，低初始力子集伤害率由 29.688% 降至 17.969%（McNemar p=0.000534），同时保留 80.657% 平均降力收益。Gate 不保证每个结构均改善，Always-on 的平均降力更大。" + SURROGATE_ZH,
    "fig08_gate_confidence_force_gain": "图8 Gate confidence 与真实最大力改善（seeds 40000–40255，n=256）。横轴为真实逐 seed confidence，纵轴为 C0−E3-G 最大力；虚线为零改善，点型区分 Gate-on 与精确回退。Spearman ρ 与 p 值由归档逐 seed 数据直接计算，趋势线仅作描述，不作因果或完美校准声明。" + SURROGATE_ZH,
    "fig09_combination_replication_forest": "图9 两次独立组合验证的 forest plot。Cohort 1（41000–41063，n=64）最大力相对变化 −27.10%，绝对均值差 95% CI [−0.092341,−0.029754]，p=7.74×10⁻⁵；Cohort 2（50000–50063，n=64）为 −19.02%，95% CI [−0.102213,−0.010696]，p=0.000587。两组并列呈现，不合并为预注册 128-seed 实验。" + SURROGATE_ZH,
    "fig10_independent64_pairplot": "图10 最新独立组合配对结果（seeds 50000–50063，n=64）。连线显示 A0 与 A0+E3-G 的逐 seed 预松弛最大力，Gate-off 样本按算法语义标记为精确平局；算法语义 Win/Tie/Loss=35/18/11。平均最大力相对下降 19.02%。" + SURROGATE_ZH,
    "fig11_leakage_diagnostic": "图11 训练泄漏诊断。Training-overlap（n=64）与 Held-out（n=192）的平均降力未显示清晰夸大，但伤害率从 0/64 上升至 31/192=16.15%，单侧 Fisher p=6.87×10⁻⁵，说明重叠显著高估 Gate 安全性。Mixed 256 仅用于诊断，禁止作为独立验证。" + SURROGATE_ZH,
    "fig12_negative_routes_summary": "图12 代表性 No-Go 路线。根据冻结实验血缘汇总采样加速、表示对齐、训练自由引导、后处理和 GPU 路线的目标、潜在收益、实测失败原因及最终状态；不构造统一评分，不从汇总值反推逐 seed 分布。",
}

CAPTIONS_EN = {
    "fig01_full_method_architecture": "Figure 1. Full architecture. The target dft_mag_density condition enters MatterGen with full Predictor/Corrector and residual-driven Adaptive CFG; a Learned Gate then selects E3-PCR or exact fallback. Innovation 1 acts during sampling and Innovation 2 after generation. " + SURROGATE_EN,
    "fig02_adaptive_cfg_mechanism": "Figure 2. Adaptive CFG. Conditional and unconditional branches produce cell, position, and atom residuals followed by EMA smoothing, a residual-driven scale update, clamping to [0,5], and CFG fusion. Neither Predictor nor Corrector is skipped; this is not Corrector Gating.",
    "fig03_e3pcr_mechanism": "Figure 3. Learned-gated E3-PCR. Fourteen risk features enter a 129-parameter gate at threshold 0.5. Gate-on executes five bounded, equivariant position-refinement steps with trust regions, backtracking, and safety checks; gate-off or rejection exactly falls back. Species and cell are unchanged.",
    "fig04_experiment_lineage": "Figure 4. Experiment and evidence lineage. Formal 256-seed experiments, two independent 64-seed replications, supplementary/diagnostic evidence, and the mixed cohort invalid for independent claims are separated by both text and line style.",
    "fig05_adaptive_cfg_results": "Figure 5. Adaptive CFG formal results (seeds 20000–20255; n=256). Mean E-hull changed by −0.003435 eV/atom, while Stable and NUS increased by 5.859 and 3.516 percentage points. Paired bootstrap intervals crossed zero and Holm-corrected tests were not significant; these are positive trends, not significant improvements. " + SURROGATE_EN,
    "fig06_e3pcr_force_formal256": "Figure 6. Independent formal three-arm E3-PCR comparison (seeds 40000–40255; n=256). E3-G reduced mean pre-relax maximum force from 0.342964 to 0.263107 eV/Å (−23.28%); paired mean-difference bootstrap 95% CI [−0.144966,−0.032453], Holm-adjusted p=4.19×10⁻¹⁰, raw-difference Win/Tie/Loss=163/0/93. " + SURROGATE_EN,
    "fig07_gate_safety_ablation": "Figure 7. Learned Gate safety mechanism (n=256). Relative to Always-on, coverage fell from 100% to 66.406%, harm from 25.391% to 18.359%, and low-force harm from 29.688% to 17.969% (McNemar p=0.000534), while 80.657% of mean force gain was retained. Gating does not guarantee per-structure improvement, and Always-on has the larger mean force reduction. " + SURROGATE_EN,
    "fig08_gate_confidence_force_gain": "Figure 8. Gate confidence versus realized maximum-force gain (seeds 40000–40255; n=256). Confidence is genuine per-seed data; markers separate gate-on and exact fallback. Spearman rho and p are recomputed from the archive. The line is descriptive and does not imply causal or perfect calibration. " + SURROGATE_EN,
    "fig09_combination_replication_forest": "Figure 9. Two independent combination cohorts. Cohort 1 (41000–41063; n=64) changed maximum force by −27.10%, absolute-difference 95% CI [−0.092341,−0.029754], p=7.74×10⁻⁵. Cohort 2 (50000–50063; n=64) changed it by −19.02%, 95% CI [−0.102213,−0.010696], p=0.000587. Cohorts are not pooled. " + SURROGATE_EN,
    "fig10_independent64_pairplot": "Figure 10. Newest independent paired combination cohort (seeds 50000–50063; n=64). Lines connect A0 and A0+E3-G maximum forces; gate-off cases are algorithmic exact ties. Semantic Win/Tie/Loss=35/18/11, and mean maximum force fell by 19.02%. " + SURROGATE_EN,
    "fig11_leakage_diagnostic": "Figure 11. Training-leakage diagnostic. Mean force gain was not clearly inflated, but harm changed from 0/64 in training-overlap to 31/192=16.15% held out (one-sided Fisher p=6.87×10⁻⁵), showing inflated apparent gate safety. The mixed 256 cohort is diagnostic only and invalid for independent validation. " + SURROGATE_EN,
    "fig12_negative_routes_summary": "Figure 12. Representative No-Go routes. Frozen experiment lineage records goals, potential benefits, observed stopping evidence, and final states across sampling, representation, post-processing, and GPU routes. No synthetic unified score or reconstructed per-seed distribution is used.",
}


def write_captions() -> None:
    for lang, captions in (("zh", CAPTIONS_ZH), ("en", CAPTIONS_EN)):
        root = FIGURE_ROOT / "captions" / lang
        root.mkdir(parents=True, exist_ok=True)
        for stem, text in captions.items():
            (root / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def build_contact_sheet() -> None:
    thumbs = []
    target_w, target_h = 900, 520
    for stem, title in FIGURES:
        image = Image.open(FIGURE_OUTPUTS / "png" / f"{stem}.png").convert("RGB")
        image.thumbnail((target_w, target_h))
        tile = Image.new("RGB", (target_w, target_h + 55), "white")
        tile.paste(image, ((target_w - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((15, target_h + 12), f"{stem}: {title}", fill="black")
        thumbs.append(tile)
    sheet = Image.new("RGB", (target_w * 2, (target_h + 55) * 6), "white")
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % 2) * target_w, (i // 2) * (target_h + 55)))
    sheet.save(FIGURE_ROOT / "generated" / "figure_contact_sheet.png", dpi=(300, 300))


def write_indices() -> None:
    figure_lines = ["# Figure index", "", "All paths are repository-relative and all statistics originate in `thesis_archive/`.", ""]
    for idx, (stem, title) in enumerate(FIGURES, 1):
        figure_lines.extend(
            [
                f"## Figure {idx}: {title}",
                "",
                f"- [PDF](pdf/{stem}.pdf) · [SVG](svg/{stem}.svg) · [PNG](png/{stem}.png)",
                f"- [Source data](../source_data/{stem}.csv) · [Python](../source/python/{stem}.py)",
                f"- [中文图注](../captions/zh/{stem}.md) · [English caption](../captions/en/{stem}.md)",
                "",
            ]
        )
    (FIGURE_ROOT / "generated" / "figure_index.md").write_text("\n".join(figure_lines), encoding="utf-8")

    table_lines = ["# Table index", "", "- [Workbook](xlsx/thesis_results.xlsx)", ""]
    for path in sorted((TABLE_ROOT / "csv").glob("*.csv")):
        stem = path.stem
        table_lines.append(
            f"- `{stem}`: [CSV](csv/{stem}.csv) · [Markdown](markdown/{stem}.md) · [LaTeX](latex/{stem}.tex)"
        )
    (TABLE_ROOT / "table_index.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_directories()
    generate_architecture_figures.main()
    generate_statistical_figures.main()
    generate_tables.main()
    write_captions()
    build_contact_sheet()
    write_indices()
    print("generated all thesis figures, tables, captions, and indices")


if __name__ == "__main__":
    main()
