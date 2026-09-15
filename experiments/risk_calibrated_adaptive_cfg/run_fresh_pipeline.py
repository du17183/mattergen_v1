"""Execution supervisor enforcing offline→P0→Formal256 stop gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
GPUS = (0, 1, 3, 4, 5, 6, 7)


def update_status(**updates) -> None:
    path = ROOT / "pipeline_status.json"; values = json.loads(path.read_text()); values.update(updates)
    temporary = path.with_suffix(".json.partial"); temporary.write_text(json.dumps(values, indent=2, default=float) + "\n"); os.replace(temporary, path)


def run_logged(script: str, log: str, gpu: str | None = None) -> None:
    environment = dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(ROOT / ".matplotlib"))
    if gpu is not None: environment.update(CUDA_VISIBLE_DEVICES=gpu, RISK_CFG_PHYSICAL_GPU=gpu)
    with (ROOT / "logs" / log).open("x") as stream:
        result = subprocess.run([PYTHON, str(ROOT / script)], cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"{script} failed; see {log}")


def generate(stage: str, seeds: list[int], script: str) -> None:
    destination = ROOT / stage / "generation"
    pending = [seed for seed in seeds if not (destination / str(seed) / "run_summary.json").exists()]
    processes = []
    for rank, gpu in enumerate(GPUS):
        assigned = pending[rank::len(GPUS)]
        if not assigned: continue
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RISK_CFG_PHYSICAL_GPU=str(gpu),
                           PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", MPLCONFIGDIR=str(ROOT / ".matplotlib"))
        stream = (ROOT / "logs" / f"{stage}_generation_gpu_{gpu}.log").open("x")
        process = subprocess.Popen([PYTHON, str(ROOT / script), "--seeds", ",".join(map(str, assigned))],
                                   cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((gpu, process, stream))
    while processes:
        running = []
        for gpu, process, stream in processes:
            code = process.poll()
            if code is None: running.append((gpu, process, stream))
            else:
                stream.close()
                if code:
                    for _, other, other_stream in running:
                        other.terminate(); other_stream.close()
                    raise RuntimeError(f"{stage} generation GPU {gpu} failed")
        processes = running
        complete = sum((destination / str(seed) / "run_summary.json").exists() for seed in seeds)
        update_status(stage=f"{stage}_generation", fresh_generation_complete=complete, fresh_generation_total=len(seeds))
        print(json.dumps({"stage": stage, "complete": complete, "total": len(seeds)}), flush=True)
        if processes: time.sleep(30)
    if sum((destination / str(seed) / "run_summary.json").exists() for seed in seeds) != len(seeds):
        raise RuntimeError(f"{stage} generation incomplete")


def main() -> None:
    offline = json.loads((ROOT / "offline/decision_summary.json").read_text())
    if offline["OFFLINE_PREDICTABILITY"] != "GO": raise RuntimeError("fresh pipeline prohibited: offline gate not GO")
    reservations = json.loads((ROOT / "future_seed_reservations.json").read_text())
    p0_seeds = list(map(int, reservations["p0_seeds"]))
    generate("p0", p0_seeds, "run_fresh_p0_generation.py")
    update_status(stage="p0_property_evaluation"); run_logged("prepare_fresh_p0.py", "p0_prepare.log", "0")
    update_status(stage="p0_quality_evaluation"); run_logged("run_p0_quality.py", "p0_quality_driver.log")
    update_status(stage="p0_analysis"); run_logged("analyze_fresh_p0.py", "p0_analysis.log")
    p0 = json.loads((ROOT / "p0/decision_summary.json").read_text()); update_status(**p0, stage="p0_complete")
    if p0["RISK_CALIBRATED_CFG_P0"] != "GO":
        print(json.dumps({"event": "STOP", "reason": "P0 FAIL"}), flush=True); return
    formal_seeds = list(map(int, reservations["formal256_seeds"]))
    (ROOT / "formal256").mkdir(exist_ok=True)
    generate("formal256", formal_seeds, "run_fresh_formal_generation.py")
    update_status(stage="formal256_property_evaluation"); run_logged("prepare_fresh_formal.py", "formal_prepare.log", "0")
    update_status(stage="formal256_quality_evaluation"); run_logged("run_formal_quality.py", "formal_quality_driver.log")
    update_status(stage="formal256_analysis"); run_logged("analyze_fresh_formal.py", "formal_analysis.log")
    formal = json.loads((ROOT / "formal256/decision_summary.json").read_text()); update_status(**formal, stage="complete")
    print(json.dumps(formal, indent=2), flush=True)


if __name__ == "__main__": main()
