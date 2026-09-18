#!/usr/bin/env python3
"""Hash external experiment trees that are intentionally excluded from Git.

Directory hashes use ``tree-sha256-v1``: sorted relative POSIX path, NUL, file
size, NUL, then file bytes for every regular file. Symlinks contribute their
relative path and link target, never the target contents.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_archive" / "external_data_manifest.csv"


def item(experiment: str, artifact: str, path: str, artifact_type: str, needed: str, reason: str = "raw structures, trajectories, checkpoints, large traces or bulk logs are excluded from ordinary Git") -> dict[str, str]:
    return {
        "experiment": experiment,
        "artifact": artifact,
        "original_path": path,
        "type": artifact_type,
        "reason_not_in_git": reason,
        "recommended_storage": "immutable object storage or read-only server snapshot; retain this manifest beside it",
        "needed_for_reproduction": needed,
    }


ITEMS = [
    item("Adaptive CFG V1", "complete later audit tree", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_innovation1_audit", "directory_tree", "partial"),
    item("Robust Adaptive CFG V2", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_robust_adaptive_cfg/experiments/robust_adaptive_cfg_v2", "directory_tree", "yes"),
    item("Counterfactual Oracle V3", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle/experiments/counterfactual_adaptive_cfg_v3", "directory_tree", "yes"),
    item("Risk-Calibrated CFG V4", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg/experiments/risk_calibrated_adaptive_cfg", "directory_tree", "yes"),
    item("Safe-Selection CFG V5", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg/experiments/calibrated_safe_selection_cfg", "directory_tree", "yes"),
    item("Stage-Calibrated CFG", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_stage_cfg/experiments/stage_calibrated_bounded_cfg", "directory_tree", "yes"),
    item("Field-Decoupled CFG", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg/experiments/field_decoupled_adaptive_cfg", "directory_tree", "yes"),
    item("Adaptive Neighborhood P0", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_adaptive_neighbor/experiments/adaptive_neighborhood_p0", "directory_tree", "partial"),
    item("Field Async Freeze", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/field_async_freeze_p0", "directory_tree", "partial"),
    item("Shared Field Dynamics", "complete diagnostic tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/shared_field_dynamics", "directory_tree", "partial"),
    item("Site-Level Convergence", "complete diagnostic tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/site_level_convergence_diagnostic", "directory_tree", "partial"),
    item("Cell-Atom Fusion P0", "complete experiment tree including checkpoint", "/mnt/datasets-livsyn/dxl/mattergen_v1_cell_atom_fusion/experiments/cell_atom_fusion_p0", "directory_tree", "partial"),
    item("Field-Time Schedule P0", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_field_schedule/experiments/field_time_schedule_p0", "directory_tree", "partial"),
    item("FP-PC P0", "complete diagnostic tree including proposal trace", "/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0", "directory_tree", "partial"),
    item("Multifield Self-Conditioning P0", "complete experiment tree including checkpoint", "/mnt/datasets-livsyn/dxl/mattergen_v1_self_condition/experiments/multifield_self_conditioning_p0", "directory_tree", "partial"),
    item("Self-Correcting Search P0", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_self_correcting_search/experiments/self_correcting_search_p0", "directory_tree", "partial"),
    item("Reference-Preserved Budgeted CFG", "complete offline and Phase B tree", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/reference_preserved_budgeted_cfg", "directory_tree", "yes"),
    item("Frozen C1 Confirmation", "complete confirmatory tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm/experiments/frozen_linear_k2_confirmation", "directory_tree", "yes"),
    item("Early Closed-Loop Force Guidance", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance/experiments/closed_loop_force_guidance_p0", "directory_tree", "partial"),
    item("RC-NFGD P0", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0", "directory_tree", "yes"),
    item("RC-NFGD Formal32", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_formal32", "directory_tree", "yes"),
    item("RC-NFGD Formal256", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_formal256", "directory_tree", "yes"),
    item("Force-Direction Ablation", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_force_direction_ablation", "directory_tree", "yes"),
    item("Trust-Region Ablation", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_trust_region_ablation", "directory_tree", "yes"),
    item("Equal-Budget Post-Generation", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_equal_budget_post_generation", "directory_tree", "yes"),
    item("Adaptive-CFG Compatibility", "complete experiment tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/adaptive_cfg_force_guidance_compatibility", "directory_tree", "yes"),
    item("Innovation 2 Finalization", "complete finalization tree", "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/innovation2_finalization", "directory_tree", "partial"),
    item("Independent CHGNet Validation", "complete metric tree", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_independent_mlip_validation", "directory_tree", "yes"),
    item("Bounded-Correction Ablation", "complete metric tree", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_bounded_correction_audit", "directory_tree", "yes"),
    item("Final Interaction Audit", "complete metric tree", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_interaction_analysis", "directory_tree", "partial"),
    item("ALM-GEN P0", "residual generated structures", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/alm_gen_p0", "directory_tree", "unknown"),
    item("Composable Property Adapter", "adapter weights and logs", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/composable_property_adapter_p0", "directory_tree", "unknown"),
    item("MatterGen LoRA P0", "generation logs and bytecode remnants", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/mattergen_lora_p0", "directory_tree", "unknown"),
    item("MatterGen RL P0", "training logs", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/mattergen_rl_p0", "directory_tree", "unknown"),
    item("Final Thesis Build Source", "duplicate generated source package", "/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results", "directory_tree", "no", "duplicate final-build sources and logs are excluded; curated content is already in release/thesis-final-2026"),
    item("MatterGen Base", "official base checkpoint", "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/mattergen_base/checkpoints/last.ckpt", "checkpoint", "yes", "upstream/third-party model weight is not duplicated in the archive branch"),
    item("Independent CHGNet Validation", "CHGNet 0.3.0 checkpoint", "/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar", "third_party_checkpoint", "yes", "downloaded third-party model weight is not committed to Git"),
]


def hash_file(path: Path, digest: hashlib._Hash) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return size, 1


def hash_tree(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total = 0
    count = 0
    for child in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        rel = child.relative_to(path).as_posix().encode("utf-8", "surrogateescape")
        if child.is_symlink():
            target = os.readlink(child).encode("utf-8", "surrogateescape")
            digest.update(b"L\0" + rel + b"\0" + target + b"\0")
            count += 1
        elif child.is_file():
            size = child.stat().st_size
            digest.update(b"F\0" + rel + b"\0" + str(size).encode("ascii") + b"\0")
            file_size, file_count = hash_file(child, digest)
            total += file_size
            count += file_count
    return total, count, digest.hexdigest()


def hash_path(path: Path) -> tuple[int, int, str, str]:
    if path.is_dir():
        size, count, digest = hash_tree(path)
        return size, count, digest, "tree-sha256-v1"
    digest = hashlib.sha256()
    size, count = hash_file(path, digest)
    return size, count, digest.hexdigest(), "file-sha256"


def main() -> int:
    fields = [
        "experiment",
        "artifact",
        "original_path",
        "size_bytes",
        "file_count",
        "sha256",
        "hash_scheme",
        "type",
        "reason_not_in_git",
        "recommended_storage",
        "needed_for_reproduction",
    ]
    rows = []
    for source in ITEMS:
        row = dict(source)
        path = Path(row["original_path"])
        if not path.exists():
            raise SystemExit(f"External artifact is missing: {path}")
        size, count, digest, scheme = hash_path(path)
        row.update(size_bytes=str(size), file_count=str(count), sha256=digest, hash_scheme=scheme)
        rows.append(row)
        print(f"hashed {row['experiment']}: {size} bytes, {count} files")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"external_artifacts={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
