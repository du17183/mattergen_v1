"""Run frozen MatterSim-5M quality evaluation only after PROPERTY_GO is true."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ase.io import read, write


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
POTENTIAL = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
REFERENCE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/reference_MP2020correction.gz")
METHODS = ("B0", "S0")
SEEDS = tuple(range(98200, 98208))


def aggregate_structures() -> None:
    output_root = ROOT / "search_structures"
    output_root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        output = output_root / f"{method}_generated.extxyz"
        if output.is_file():
            continue
        structures = []
        for seed in SEEDS:
            atoms = read(
                ROOT / "search_generation" / method / str(seed) / "generated_crystals.extxyz",
                index=":",
            )
            if len(atoms) != 1:
                raise RuntimeError(f"expected one structure: {method} {seed}")
            item = atoms[0]
            item.info.update(sample_seed=seed, sample_index_within_seed=0,
                             benchmark_method=method, dft_verified=False)
            structures.append(item)
        write(output, structures)


def run_relaxations(gpus: tuple[str, ...]) -> None:
    jobs = []
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for method, gpu in zip(METHODS, gpus):
        output = ROOT / "mattersim_relaxation" / method
        if (output / "relaxation_summary.json").is_file():
            continue
        if output.exists():
            raise RuntimeError(f"incomplete MatterSim relaxation retained: {output}")
        log_path = logs / f"mattersim_relaxation_{method}.log"
        stream = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable, "-m", "research.corrector_distillation.relax_many_individual",
            "--structures-path", str(ROOT / "search_structures" / f"{method}_generated.extxyz"),
            "--potential-path", str(POTENTIAL), "--output-dir", str(output),
            "--device", "cuda", "--fmax", "0.05", "--max-natoms-per-batch", "512",
            "--cuda-3x3-det-workaround",
        ]
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        jobs.append((method, process, stream, log_path))
    failures = []
    for method, process, stream, log_path in jobs:
        code = process.wait()
        stream.close()
        if code:
            failures.append((method, code, str(log_path)))
    if failures:
        raise RuntimeError(f"MatterSim relaxation failures: {failures}")


def run_official_quality() -> None:
    memory_fd = os.memfd_create("self_correcting_search_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(
        memory_fd, fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL,
    )
    jobs = []
    try:
        for method in METHODS:
            output = ROOT / "mattersim_quality" / method
            if (output / "quality_summary.json").is_file():
                continue
            if output.exists():
                raise RuntimeError(f"incomplete official quality output retained: {output}")
            log_path = ROOT / "logs" / f"mattersim_quality_{method}.log"
            stream = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable, "-m", "research.corrector_distillation.evaluate_quality",
                "--method", method,
                "--structures-path", str(ROOT / "search_structures" / f"{method}_generated.extxyz"),
                "--potential-path", str(POTENTIAL),
                "--reference-memory-fd", str(memory_fd),
                "--output-dir", str(output),
                "--temporary-dir", str(ROOT / "runtime_tmp"),
                "--device", "cpu",
                "--precomputed-relaxation-dir", str(ROOT / "mattersim_relaxation" / method),
            ]
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, pass_fds=(memory_fd,),
            )
            jobs.append((method, process, stream, log_path))
        failures = []
        for method, process, stream, log_path in jobs:
            code = process.wait()
            stream.close()
            if code:
                failures.append((method, code, str(log_path)))
        if failures:
            raise RuntimeError(f"official quality failures: {failures}")
    finally:
        os.close(memory_fd)


def merge_results() -> None:
    summaries = {
        method: json.loads((ROOT / "mattersim_quality" / method / "quality_summary.json").read_text())
        for method in METHODS
    }
    rows = []
    for method in METHODS:
        summary = summaries[method]
        rows.append({"method": method, "n": summary["n"],
                     "relaxation_success_rate": summary["relaxation_success_rate"],
                     "relaxation_steps_mean": summary["relaxation_steps_mean"],
                     "pre_relaxation_max_force_mean": summary["pre_relaxation_max_force_mean"],
                     **summary["official_metrics"], "surrogate_quality_eval": True,
                     "dft_verified": False})
    with (ROOT / "mattersim_quality_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    decision = json.loads((ROOT / "decision_summary.json").read_text())
    decision["mattersim_run"] = True
    decision["mattersim_model"] = "MatterSim-v1.0.0-5M"
    decision["mattersim_quality"] = {row["method"]: row for row in rows}
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,6")
    args = parser.parse_args()
    decision = json.loads((ROOT / "decision_summary.json").read_text())
    if not decision.get("property_go", False):
        print(json.dumps({"event": "mattersim_not_run", "reason": "PROPERTY_GO false"}))
        return
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) < 2:
        raise ValueError("two explicitly free GPUs are required for paired MatterSim evaluation")
    aggregate_structures()
    run_relaxations(gpus[:2])
    run_official_quality()
    merge_results()


if __name__ == "__main__":
    main()
