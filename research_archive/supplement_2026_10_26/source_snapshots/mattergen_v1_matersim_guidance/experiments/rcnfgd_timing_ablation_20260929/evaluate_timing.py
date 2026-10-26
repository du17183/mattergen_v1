"""Reuse frozen A1 property, MatterSim relaxation, and quality evaluators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from ase.io import read, write


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "timing_formal32"
ALM_PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows or path.exists():
        raise RuntimeError(f"empty or existing output table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(protocol: dict, seeds: list[int]) -> None:
    structures = OUT / "structures"
    if structures.exists():
        raise FileExistsError(structures)
    structures.mkdir(parents=True)
    rows: list[dict] = []
    traces: list[dict] = []
    source_hashes: dict[str, str] = {}
    for method in protocol["methods"]:
        atoms = []
        for seed in seeds:
            run = OUT / "generation" / method / str(seed)
            summary = json.loads((run / "run_summary.json").read_text())
            if (summary["success"] is not True or summary["method"] != method
                    or summary["seed"] != seed or summary["mattergen_score_calls"] != 2000):
                raise RuntimeError(f"incomplete/invalid generation record: {method}/{seed}")
            expected = int(protocol["expected_predictor_force_opportunities"][method])
            if method != "C0" and summary["mattersim_guidance_eligible"] != expected:
                raise RuntimeError(f"incorrect force opportunities: {method}/{seed}")
            structure_path = run / "generated_crystals.extxyz"
            items = read(structure_path, index=":")
            if len(items) != 1:
                raise RuntimeError(f"wrong structure count: {method}/{seed}")
            atom = items[0]
            atom.info.update(sample_seed=seed, sample_index_within_seed=0, method=method)
            atoms.append(atom)
            source_hashes[str(structure_path)] = sha256(structure_path)
            rows.append({
                "method": method, "seed": seed, "success": True,
                "formula": atom.get_chemical_formula(), "num_atoms": len(atom),
                "elapsed_seconds": summary["elapsed_seconds"],
                "potential_load_seconds": summary["potential_load_seconds"],
                "end_to_end_seconds": summary["end_to_end_seconds"],
                "cuda_event_elapsed_seconds": summary["cuda_event_elapsed_seconds"],
                "physical_gpu": summary["physical_gpu"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "mattergen_score_calls": summary["mattergen_score_calls"],
                "guidance_eligible": summary.get("mattersim_guidance_eligible", 0),
                "guidance_accepted": summary.get("mattersim_guidance_accepted", 0),
                "guidance_fallbacks": summary.get("mattersim_guidance_fallbacks", 0),
                "guidance_seconds": summary.get("mattersim_guidance_seconds", 0.0),
                "actual_mattersim_calls": summary["actual_force_model_calls"],
                "failed_mattersim_calls": summary["failed_force_model_calls"],
            })
            if method != "C0":
                with (run / "late_force_trace.csv").open(newline="", encoding="utf-8") as stream:
                    events = list(csv.DictReader(stream))
                if len(events) != expected:
                    raise RuntimeError(f"trace count mismatch: {method}/{seed}")
                traces.extend([event | {"method": method} for event in events])
        write(structures / f"{method}_generated.extxyz", atoms)
    write_table(OUT / "generation_manifest.csv", rows)
    write_table(OUT / "guidance_trace.csv", traces)
    alias = OUT / "p0_structures"
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(alias)
    alias.symlink_to(structures, target_is_directory=True)
    (structures / "manifest.json").write_text(json.dumps({
        "methods": protocol["methods"], "paired_seeds": seeds,
        "n_per_method": len(seeds), "all_registered_seeds_included": True,
        "source_sha256": source_hashes,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "prepare", "rows": len(rows), "trace_rows": len(traces)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "properties", "relax", "quality"), required=True)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != ALM_PYTHON.resolve():
        raise RuntimeError("ALM interpreter required for frozen evaluators")
    protocol = json.loads((ROOT / "timing_protocol.json").read_text())
    seeds = list(range(protocol["seed_start"], protocol["seed_stop_exclusive"]))
    status = json.loads((OUT / "generation_status_attempt3.json").read_text())
    if status["status"] != "COMPLETE" or status["completed_paired_seeds"] != seeds:
        raise RuntimeError("all four generation arms must complete before evaluation")
    if args.stage == "prepare":
        prepare(protocol, seeds)
    elif args.stage == "properties":
        from experiments.mattersim_late_force_guidance_p0 import evaluate_p0_properties as frozen
        frozen.ROOT = OUT
        frozen.SEEDS = tuple(seeds)
        frozen.METHODS = tuple(protocol["methods"])
        frozen.main()
    elif args.stage == "relax":
        from experiments.mattersim_late_force_guidance_p0 import run_p0_relaxation as frozen
        frozen.ROOT = OUT
        frozen.JOBS = tuple(zip(protocol["methods"], (2, 3, 4, 5)))
        frozen.main()
    else:
        from experiments.mattersim_late_force_guidance_p0 import run_p0_quality as frozen
        frozen.ROOT = OUT
        frozen.METHODS = tuple(protocol["methods"])
        frozen.main()
    print(json.dumps({"stage": args.stage, "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
