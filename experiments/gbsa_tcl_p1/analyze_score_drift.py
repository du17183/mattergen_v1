"""Frozen P1 score-drift probe on all C0 and TCL initial structures (64 total)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ase.io import read
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p1"
P0_SCRIPT = PROJECT_ROOT / "experiments/gbsa_tcl_p0/analyze_score_drift.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_p0_score_probe", P0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.CHECKPOINTS = {
        "TCL": PROJECT_ROOT / "experiments/tcl_dml_p0/checkpoints/TCL/model.pt",
        "GBSA-TCL": PROJECT_ROOT / "experiments/gbsa_tcl_p0/checkpoints/GBSA-TCL/model.pt",
    }
    module.NOISE_SEED = 20260921
    if not torch.cuda.is_available():
        raise RuntimeError("score-drift diagnostic requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda")
    atoms = []
    for source in ("C0", "TCL"):
        values = read(ROOT / "structures" / f"{source}_generated.extxyz", index=":")
        if len(values) != 32:
            raise RuntimeError(f"expected 32 {source} structures")
        atoms.extend(values)
    baseline, _ = module.run_model("C0", atoms, device)
    all_rows = []
    for method in ("TCL", "GBSA-TCL"):
        _, rows = module.run_model(method, atoms, device, baseline)
        all_rows.extend(rows)
    detail = pd.DataFrame(all_rows)
    detail.to_csv(ROOT / "score_drift_per_sample.csv", index=False)
    summary = detail.groupby(["comparison", "field", "stage"], as_index=False).agg(
        n=("seed", "count"), rms_diff_mean=("rms_diff", "mean"),
        relative_l2_diff_mean=("relative_l2_diff", "mean"),
        relative_l2_diff_median=("relative_l2_diff", "median"),
        cosine_to_c0_mean=("cosine_to_c0", "mean"),
        c0_rms_mean=("c0_rms", "mean"), candidate_rms_mean=("candidate_rms", "mean"),
    )
    summary.to_csv(ROOT / "score_drift_results.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
