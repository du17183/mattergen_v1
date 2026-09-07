"""Matched-state score and hidden-representation drift for frozen checkpoints.

The probe uses all 256 already generated TCL pre-relaxation structures, applies
the checkpoint's real corruption process at ten fixed noise levels, and runs
the real C0/FT0/TCL models with the Formal CFG=2 arithmetic.  It does not train,
sample a reverse trajectory, or create new crystals.
"""
from __future__ import annotations

import gc
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
from scipy import stats
import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.property_embeddings import SetConditionalEmbeddingType, SetUnconditionalEmbeddingType


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
TRAINED = ROOT / "experiments/tcl_dml_p0/checkpoints"
STRUCTURES = ROOT / "experiments/tcl_formal256/relaxation/TCL/initial_with_properties.extxyz"
METHODS = ("C0", "FT0", "TCL")
FIELDS = ("atomic_numbers", "pos", "cell")
BATCH_SIZE = 64
CFG_SCALE = 2.0
PROGRESS_CENTERS = tuple(np.arange(0.05, 1.0, 0.1).round(2))
NOISE_SEED = 20260915


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
        return "early_0_30_high_to_mid_noise"
    if progress < 0.7:
        return "mid_30_70"
    return "late_70_100_mid_to_low_noise"


def split_fields(output, node_count: int, graph_count: int):
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


def vector_metrics(base: torch.Tensor, candidate: torch.Tensor):
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
        "mean_abs_diff": float(delta.abs().mean()),
        "rms_diff": float(delta.square().mean().sqrt()),
        "relative_l2_diff": delta_norm / base_norm if base_norm else float("nan"),
        "cosine_to_c0": cosine,
        "one_minus_cosine": 1.0 - cosine,
        "candidate_to_c0_norm_ratio": candidate_norm / base_norm if base_norm else float("nan"),
        "c0_rms": float(a.square().mean().sqrt()),
        "candidate_rms": float(b.square().mean().sqrt()),
    }


def load_model(method: str, device):
    model = load_model_diffusion(MatterGenCheckpointInfo(model_path=str(MODEL_ROOT.resolve())))
    if method != "C0":
        payload = torch.load(TRAINED / method / "model.pt", map_location="cpu", weights_only=False)
        if payload["method"] != method:
            raise RuntimeError("checkpoint method mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
    gemnet = model.diffusion_module.model.gemnet
    for attribute in ("global_adapter", "cross_field_adapter", "quality_adapter"):
        setattr(gemnet, attribute, None)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_model(method, atoms, device, baseline_cache=None):
    model = load_model(method, device)
    diffusion = model.diffusion_module
    hidden_capture = {}
    hooks = []
    for block_index, block in enumerate(diffusion.model.gemnet.int_blocks):
        def hook(_module, _inputs, output, index=block_index):
            hidden_capture[index] = output[0].detach()
        hooks.append(block.register_forward_hook(hook))

    cache = {} if baseline_cache is None else None
    rows = []
    with torch.inference_mode():
        for batch_start in range(0, len(atoms), BATCH_SIZE):
            batch_atoms = atoms[batch_start:batch_start + BATCH_SIZE]
            clean = SetConditionalEmbeddingType()(collate([graph_from_atoms(a) for a in batch_atoms])).to(device)
            graph_count = len(batch_atoms)
            node_count = int(clean.num_atoms.sum().item())
            atom_batch = clean.batch.detach().cpu()
            seeds = [int(a.info["sample_seed"]) for a in batch_atoms]
            for bin_index, progress in enumerate(PROGRESS_CENTERS):
                forward_t = 1.0 - float(progress)
                t = torch.full((graph_count,), forward_t, device=device)
                seed_value = NOISE_SEED + batch_start * 100 + bin_index
                torch.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value)
                noisy = diffusion.corruption.sample_marginal(clean, t)
                unconditional = SetUnconditionalEmbeddingType()(noisy)
                conditional = SetConditionalEmbeddingType()(noisy)
                joint = collate([unconditional, conditional])
                combined = diffusion.model(joint, torch.cat([t, t]))
                guided = {k: v.detach().cpu() for k, v in split_fields(combined, node_count, graph_count).items()}
                hidden = {
                    block: value[node_count:].detach().cpu()
                    for block, value in hidden_capture.items()
                }
                key = (batch_start, bin_index)
                if baseline_cache is None:
                    cache[key] = {"guided": guided, "hidden": hidden, "atom_batch": atom_batch, "seeds": seeds}
                    continue
                base = baseline_cache[key]
                if base["seeds"] != seeds or not torch.equal(base["atom_batch"], atom_batch):
                    raise RuntimeError("baseline probe alignment mismatch")
                for local_index, seed in enumerate(seeds):
                    node_mask = atom_batch == local_index
                    for field in FIELDS:
                        if field == "cell":
                            a = base["guided"][field][local_index]
                            b = guided[field][local_index]
                        else:
                            a = base["guided"][field][node_mask]
                            b = guided[field][node_mask]
                        rows.append({
                            "comparison": f"{method}_vs_C0",
                            "kind": "cfg2_score",
                            "field_or_block": field,
                            "seed": seed,
                            "sampling_progress": progress,
                            "forward_diffusion_t": forward_t,
                            "stage": stage(progress),
                            **vector_metrics(a, b),
                        })
                    for block in range(4):
                        a = base["hidden"][block][node_mask]
                        b = hidden[block][node_mask]
                        rows.append({
                            "comparison": f"{method}_vs_C0",
                            "kind": "conditional_hidden",
                            "field_or_block": f"gemnet_h{block + 1}",
                            "seed": seed,
                            "sampling_progress": progress,
                            "forward_diffusion_t": forward_t,
                            "stage": stage(progress),
                            **vector_metrics(a, b),
                        })
                print(f"{method}: batch {batch_start // BATCH_SIZE + 1}/4, bin {bin_index + 1}/10", flush=True)

    for hook in hooks:
        hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cache, rows


def aggregate(frame, grouping):
    metrics = [
        "mean_abs_diff", "rms_diff", "relative_l2_diff", "cosine_to_c0",
        "one_minus_cosine", "candidate_to_c0_norm_ratio", "c0_rms", "candidate_rms",
    ]
    rows = []
    for keys, group in frame.groupby(grouping, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        row["n"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_median"] = float(np.nanmedian(values))
            row[f"{metric}_p95"] = float(np.nanquantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_correlations(frame):
    outcomes = pd.read_csv(OUT / "structural_diagnostics.csv")
    records = []
    for comparison in ("FT0_vs_C0", "TCL_vs_C0"):
        method = comparison.split("_")[0]
        target = outcomes[outcomes.method == method].set_index("seed")
        sub = frame[frame.comparison == comparison]
        for (kind, field, probe_stage), group in sub.groupby(["kind", "field_or_block", "stage"]):
            per_seed = group.groupby("seed")["relative_l2_diff"].mean()
            for outcome in ("initial_max_force", "atomic_force_mean", "rmsd", "relaxation_steps", "e_hull", "severe"):
                aligned = target.loc[per_seed.index, outcome].astype(float)
                rho, p = stats.spearmanr(per_seed, aligned)
                records.append({
                    "comparison": comparison,
                    "kind": kind,
                    "field_or_block": field,
                    "stage": probe_stage,
                    "outcome": outcome,
                    "spearman_rho": float(rho),
                    "two_sided_p": float(p),
                    "n": len(per_seed),
                })
    return pd.DataFrame(records)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("real checkpoint forward diagnostics require CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    atoms = read(STRUCTURES, index=":")
    if len(atoms) != 256:
        raise RuntimeError("expected all 256 existing TCL structures")
    baseline, _ = run_model("C0", atoms, device)
    all_rows = []
    for method in ("FT0", "TCL"):
        _, rows = run_model(method, atoms, device, baseline)
        all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT / "score_hidden_drift_per_seed_bin.csv", index=False)
    aggregate(
        frame,
        ["comparison", "kind", "field_or_block", "sampling_progress", "forward_diffusion_t", "stage"],
    ).to_csv(OUT / "score_hidden_drift_by_10pct_bin.csv", index=False)
    aggregate(
        frame,
        ["comparison", "kind", "field_or_block", "stage"],
    ).to_csv(OUT / "score_hidden_drift_by_stage.csv", index=False)
    correlations = outcome_correlations(frame)
    correlations.to_csv(OUT / "score_hidden_outcome_correlations.csv", index=False)
    print("completed rows", len(frame), "correlations", len(correlations), flush=True)


if __name__ == "__main__":
    main()
