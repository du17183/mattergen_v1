"""Matched-state CFG=2 score drift on all 24 newly generated P0 structures."""
from __future__ import annotations

import gc
import os
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.property_embeddings import SetConditionalEmbeddingType, SetUnconditionalEmbeddingType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
CHECKPOINTS = {
    "TCL": PROJECT_ROOT / "experiments/tcl_dml_p0/checkpoints/TCL/model.pt",
    "GBSA-TCL": ROOT / "checkpoints/GBSA-TCL/model.pt",
}
METHODS = ("TCL", "GBSA-TCL")
FIELDS = ("atomic_numbers", "pos", "cell")
CFG_SCALE = 2.0
PROGRESS_CENTERS = tuple(np.arange(0.05, 1.0, 0.1).round(2))
NOISE_SEED = 20260919


def graph_from_atoms(atoms) -> ChemGraph:
    graph = ChemGraph(
        pos=torch.tensor(atoms.get_scaled_positions(wrap=True), dtype=torch.float32),
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0),
        atomic_numbers=torch.tensor(atoms.numbers, dtype=torch.long),
        num_atoms=torch.tensor(len(atoms)),
        num_nodes=len(atoms),
        dft_mag_density=torch.tensor(0.1, dtype=torch.float32),
    )
    return symmetrize_lattice(graph)


def stage(progress: float) -> str:
    if progress < 0.3:
        return "early_0_30"
    if progress < 0.7:
        return "mid_30_70"
    return "late_70_100"


def split_cfg(output, node_count: int, graph_count: int):
    unconditional = {
        "atomic_numbers": output["atomic_numbers"][:node_count],
        "pos": output["pos"][:node_count],
        "cell": output["cell"][:graph_count],
    }
    conditional = {
        "atomic_numbers": output["atomic_numbers"][node_count:],
        "pos": output["pos"][node_count:],
        "cell": output["cell"][graph_count:],
    }
    return {
        field: torch.lerp(unconditional[field], conditional[field], CFG_SCALE)
        for field in FIELDS
    }


def vector_metrics(base: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    a = base.double().reshape(-1)
    b = candidate.double().reshape(-1)
    delta = b - a
    base_norm = float(torch.linalg.vector_norm(a))
    candidate_norm = float(torch.linalg.vector_norm(b))
    delta_norm = float(torch.linalg.vector_norm(delta))
    cosine = (
        float(torch.dot(a, b)) / (base_norm * candidate_norm)
        if base_norm and candidate_norm else float("nan")
    )
    return {
        "rms_diff": float(delta.square().mean().sqrt()),
        "relative_l2_diff": delta_norm / base_norm if base_norm else float("nan"),
        "cosine_to_c0": cosine,
        "c0_rms": float(a.square().mean().sqrt()),
        "candidate_rms": float(b.square().mean().sqrt()),
    }


def load_model(method: str, device):
    model = load_model_diffusion(
        MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve()))
    )
    if method != "C0":
        payload = torch.load(CHECKPOINTS[method], map_location="cpu", weights_only=False)
        if payload["method"] != method:
            raise RuntimeError(f"checkpoint mismatch for {method}")
        model.load_state_dict(payload["model_state_dict"], strict=True)
    gemnet = model.diffusion_module.model.gemnet
    for attribute in ("global_adapter", "cross_field_adapter", "quality_adapter"):
        setattr(gemnet, attribute, None)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_model(method, atoms, device, baseline=None):
    model = load_model(method, device)
    diffusion = model.diffusion_module
    clean = SetConditionalEmbeddingType()(
        collate([graph_from_atoms(item) for item in atoms])
    ).to(device)
    graph_count = len(atoms)
    node_count = int(clean.num_atoms.sum())
    atom_batch = clean.batch.detach().cpu()
    cache = {} if baseline is None else None
    rows = []
    with torch.inference_mode():
        for bin_index, progress in enumerate(PROGRESS_CENTERS):
            forward_t = 1.0 - float(progress)
            t = torch.full((graph_count,), forward_t, device=device)
            torch.manual_seed(NOISE_SEED + bin_index)
            torch.cuda.manual_seed_all(NOISE_SEED + bin_index)
            noisy = diffusion.corruption.sample_marginal(clean, t)
            joint = collate([
                SetUnconditionalEmbeddingType()(noisy),
                SetConditionalEmbeddingType()(noisy),
            ])
            combined = diffusion.model(joint, torch.cat([t, t]))
            guided = {
                key: value.detach().cpu()
                for key, value in split_cfg(combined, node_count, graph_count).items()
            }
            if baseline is None:
                cache[bin_index] = guided
                continue
            for index, item in enumerate(atoms):
                mask = atom_batch == index
                for field in FIELDS:
                    if field == "cell":
                        a = baseline[bin_index][field][index]
                        b = guided[field][index]
                    else:
                        a = baseline[bin_index][field][mask]
                        b = guided[field][mask]
                    rows.append({
                        "comparison": f"{method}_vs_C0",
                        "source_method": item.info["benchmark_method"],
                        "seed": int(item.info["sample_seed"]),
                        "field": field,
                        "sampling_progress": progress,
                        "forward_diffusion_t": forward_t,
                        "stage": stage(progress),
                        **vector_metrics(a, b),
                    })
            print(f"{method}: bin {bin_index + 1}/10", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cache, rows


def main() -> None:
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
    for source in ("C0", "TCL", "GBSA-TCL"):
        values = read(ROOT / "structures" / f"{source}_generated.extxyz", index=":")
        if len(values) != 8:
            raise RuntimeError(f"expected eight {source} structures")
        atoms.extend(values)
    baseline, _ = run_model("C0", atoms, device)
    all_rows = []
    for method in METHODS:
        _, rows = run_model(method, atoms, device, baseline)
        all_rows.extend(rows)
    detail = pd.DataFrame(all_rows)
    detail.to_csv(ROOT / "score_drift_per_sample.csv", index=False)
    summary = detail.groupby(["comparison", "field", "stage"], as_index=False).agg(
        n=("seed", "count"),
        rms_diff_mean=("rms_diff", "mean"),
        relative_l2_diff_mean=("relative_l2_diff", "mean"),
        relative_l2_diff_median=("relative_l2_diff", "median"),
        cosine_to_c0_mean=("cosine_to_c0", "mean"),
        c0_rms_mean=("c0_rms", "mean"),
        candidate_rms_mean=("candidate_rms", "mean"),
    )
    summary.to_csv(ROOT / "score_drift_results.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
