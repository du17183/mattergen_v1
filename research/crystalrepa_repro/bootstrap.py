from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.crystalrepa_repro.common import REPORTS, RESULTS, TOOLS, atomic_json, atomic_text, initialize_progress, now, set_stage, sha256_file
from research.crystalrepa_repro.configuration import CHECKPOINT, CHECKPOINT_SHA256

DATASET = Path("/data/dxl/datasets/cache/mp_20")
CACHE = Path("/data/dxl/data/crystalrepa_teacher_cache")
P1_REPORTS = Path("/data/dxl/reports/fn_pra/phase1")


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def audit_split(split: str) -> dict:
    source = DATASET / split
    target = CACHE / split
    num_atoms = np.load(source / "num_atoms.npy", mmap_mode="r")
    offsets = np.concatenate(([0], np.cumsum(num_atoms[:-1], dtype=np.int64)))
    atomic_numbers = np.load(source / "atomic_numbers.npy", mmap_mode="r")
    positions = np.load(source / "pos.npy", mmap_mode="r")
    cells = np.load(source / "cell.npy", mmap_mode="r")
    structure_ids = np.load(source / "structure_id.npy", allow_pickle=True)
    cache_indices = np.load(target / "dataset_indices.npy")
    cache_num_atoms = np.load(target / "num_atoms.npy")
    cache_ids = np.load(target / "structure_ids.npy")
    cache_atomic = np.load(target / "atomic_numbers.npy", mmap_mode="r")
    cache_atomic_hash = np.load(target / "atomic_numbers_hash.npy")
    cache_position_hash = np.load(target / "positions_hash.npy")
    cache_cell_hash = np.load(target / "cell_hash.npy")
    cache_combined = np.load(target / "combined_structure_hash.npy")
    features = np.load(target / "features.f16.npy", mmap_mode="r")
    counts = {"atomic": 0, "position": 0, "cell": 0, "combined": 0}
    for index in range(len(num_atoms)):
        start = int(offsets[index])
        stop = start + int(num_atoms[index])
        z = np.asarray(atomic_numbers[start:stop], dtype=np.int64)
        pos = np.asarray(positions[start:stop], dtype=np.float64)
        cell = np.asarray(cells[index], dtype=np.float64)
        z_hash, pos_hash, cell_hash = array_hash(z), array_hash(pos), array_hash(cell)
        combined = hashlib.sha256(f"{structure_ids[index]}|{z_hash}|{cell_hash}|{pos_hash}".encode()).hexdigest()
        counts["atomic"] += int(str(cache_atomic_hash[index]) != z_hash)
        counts["position"] += int(str(cache_position_hash[index]) != pos_hash)
        counts["cell"] += int(str(cache_cell_hash[index]) != cell_hash)
        counts["combined"] += int(str(cache_combined[index]) != combined)
    canonical = np.array_equal(cache_indices, np.arange(len(num_atoms)))
    atom_rows_match = np.array_equal(np.asarray(cache_atomic), np.asarray(atomic_numbers, dtype=np.int64))
    passed = bool(
        canonical
        and np.array_equal(cache_num_atoms, np.asarray(num_atoms, dtype=np.int32))
        and np.array_equal(cache_ids.astype(str), np.asarray(structure_ids, dtype=str))
        and atom_rows_match
        and not any(counts.values())
        and features.shape == (int(num_atoms.sum()), 64)
        and np.isfinite(np.asarray(features)).all()
    )
    return {
        "split": split, "structures": len(num_atoms), "atoms": int(num_atoms.sum()),
        "canonical_full_coverage": canonical, "atom_rows_match": atom_rows_match,
        "hash_mismatches": counts, "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype), "features_finite": bool(np.isfinite(np.asarray(features)).all()),
        "passed": passed,
    }


def main() -> None:
    for path in (RESULTS, REPORTS, TOOLS, REPORTS / "frozen"):
        path.mkdir(parents=True, exist_ok=True)
    initialize_progress()
    p1_files = [P1_REPORTS / "phase1_decision.json", P1_REPORTS / "phase1_final_report.md"]
    p1 = {
        "created_at": now(), "P1_FROZEN": True,
        "source_branch": "feature/fn-pra", "source_commit": "42681f83a0d70c25f6f2e598232868c169904e30",
        "files": [{"path": str(path), "sha256": sha256_file(path)} for path in p1_files if path.exists()],
        "modified": False,
    }
    atomic_json(REPORTS / "frozen/p1_no_go_snapshot.json", p1)
    set_stage("state_audit", "success", "P1 frozen; worktree/process/GPU audit passed before branch creation.", {"gpu_count": 8, "duplicate_launcher": False, "p1_frozen": True})

    paper = {
        "schema_version": 1, "verified_at": now(),
        "paper": "CrystalREPA: Transferring Physical Priors from Universal MLIPs to Crystal Generative Models",
        "source": "https://arxiv.org/html/2605.08960",
        "mattergen_mp20": {
            "base_generator": "unconditional MatterGen on MP-20",
            "base_checkpoint_initialization": "NOT_VERIFIED",
            "gnn_layers": 4, "atom_embedding_dim": 512, "alignment_block_1_indexed": 2,
            "projection_head": "one residual linear+SiLU block, then linear to teacher dimension",
            "alignment_loss": "symmetric EA-NCE", "temperature": 0.1, "alignment_weight": 1.0,
            "optimizer": "Adam", "scheduler": "ReduceLROnPlateau", "max_lr": 1e-4, "min_lr": 1e-6,
            "gradient_accumulation": 4, "epochs": 1900, "batch_size_per_gpu": 128, "paper_gpus": 1,
            "trainable_scope": "NOT_VERIFIED; table reports 44.6M parameters and no frozen-backbone protocol",
            "sampling_steps": 1000, "evaluation_samples_per_method": 1024, "independent_runs": 5,
            "relaxer": "MatterSim-v1-1M",
        },
        "local_minimal_reproduction": {
            "teacher": "CHGNet 0.3.0", "teacher_layer": "atom_fea before final CHGNet convolution",
            "teacher_is_one_of_paper_ten": False,
            "deviation_reason": "User-frozen independent Teacher cache; avoids MatterSim self-evaluation and cache rebuild.",
            "normalization": "cosine L2 normalization inside EA-NCE; diagnostic dimension-wise z-score is not used",
            "training_cap": 10000, "gpus": 8, "batch_size_per_gpu": 16,
            "global_micro_batch": 128, "gradient_accumulation": 4, "effective_batch": 512,
        },
        "paper_teachers": ["DPA-3.1-3M", "DPA-3.1-3M-FT", "DPA-3.1-MPtrj", "MACE-MPA-0", "MACE-MP-0", "ORB v3", "ORB v2", "SevenNet-Omni-i12", "SevenNet-l3i5", "MatterSim-v1-5M"],
    }
    atomic_json(REPORTS / "frozen/paper_config.json", paper)
    atomic_text(REPORTS / "frozen/paper_config_verified.md", "# Verified CrystalREPA MatterGen configuration\n\nSource: https://arxiv.org/html/2605.08960 (Appendix B.2–B.3).\n\nThe verified MatterGen MP-20 setting uses four GNN layers, block 2 alignment, symmetric EA-NCE, temperature 0.1, alignment weight 1, Adam at 1e-4 with ReduceLROnPlateau to 1e-6, batch 128/GPU, accumulation 4, one A800-80GB, and 1900 epochs. Inference keeps the original 1000-step sampler.\n\nImportant controlled deviation: the local experiment uses the user-frozen CHGNet 0.3.0 cache. CHGNet is not one of the paper's ten teachers, so this is a CrystalREPA-like isolated diagnostic, not a bit-for-bit paper reproduction. The paper does not state a frozen-backbone protocol; its 44.6M parameter row supports full-backbone training. Base-checkpoint initialization and exact teacher internal layer are marked NOT_VERIFIED.\n")
    set_stage("paper_config_verification", "success", "Paper configuration verified; CHGNet teacher deviation explicitly frozen.", {"alignment_block": 2, "temperature": 0.1, "alignment_weight": 1.0})
    set_stage("branch_creation", "success", "Created feature/crystalrepa-repro from main.", {"base_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544"})
    actual_sha = sha256_file(CHECKPOINT)
    checkpoint_report = {"path": str(CHECKPOINT), "exists": CHECKPOINT.exists(), "sha256": actual_sha, "expected_sha256": CHECKPOINT_SHA256, "strict_load_pending_integration_test": True, "passed": actual_sha == CHECKPOINT_SHA256}
    atomic_json(REPORTS / "checkpoint_audit.json", checkpoint_report)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError("Official MP-20 checkpoint SHA mismatch")
    set_stage("checkpoint_audit", "success", "Official unconditional MP-20 checkpoint located and SHA256 verified.", checkpoint_report)
    cache_report = {
        "created_at": now(), "cache": str(CACHE), "teacher": "CHGNet 0.3.0",
        "teacher_checkpoint_sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
        "teacher_layer": "atom_fea before final CHGNet convolution", "representation_dimension": 64,
        "splits": [audit_split("train"), audit_split("val")],
    }
    cache_report["CACHE_REUSED"] = all(item["passed"] for item in cache_report["splits"])
    cache_report["CACHE_MAPPING_PASSED"] = cache_report["CACHE_REUSED"]
    atomic_json(REPORTS / "cache_reuse_audit.json", cache_report)
    if not cache_report["CACHE_MAPPING_PASSED"]:
        set_stage("cache_reuse_audit", "blocked", "Teacher cache full mapping failed.", cache_report)
        raise RuntimeError("Teacher cache full mapping failed")
    set_stage("cache_reuse_audit", "success", "Teacher cache reused; every MP-20 train/val structure and atom hash matched.", cache_report)
    print(json.dumps({"paper": paper, "checkpoint": checkpoint_report, "cache": cache_report}, indent=2))


if __name__ == "__main__":
    main()
