from __future__ import annotations

import argparse
import csv
import os
import traceback
from pathlib import Path

import torch

from research.spg_static_mvp.common import RESULTS, atomic_json, now
from research.spg_static_mvp.generation import (
    build_c0_generator,
    build_recording_sampler,
    configure_determinism,
    find_gemnet,
    singleton_condition,
)
from research.spg_static_mvp.reference_graph import (
    build_reference_graph,
    cell_repetitions,
)


OUTPUT = RESULTS / "shape_states"
CSV_FIELDS = (
    "seed",
    "state_index",
    "sampling_step",
    "phase",
    "progress",
    "num_atoms",
    "rep_a1",
    "rep_a2",
    "rep_a3",
    "candidate_periodic_images",
    "candidate_atom_pairs",
    "candidate_pair_images",
    "raw_edge_count",
    "edge_count",
    "max_neighbors",
    "mean_neighbors",
    "triplet_count",
    "cell_volume",
    "cell_condition_number",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_states(path: Path, states: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(states, temporary)
    os.replace(temporary, path)


def state_statistics(gemnet, states: list[dict]) -> list[dict]:
    device = next(gemnet.parameters()).device
    rows = []
    with torch.no_grad():
        for state in states:
            frac_positions = state["pos"].to(device)
            cell = state["cell"].to(device)
            num_atoms = state["num_atoms"].to(device)
            graph = build_reference_graph(
                gemnet,
                frac_positions=frac_positions,
                cell=cell,
                num_atoms=num_atoms,
            )
            n_atoms = int(num_atoms.sum())
            repetitions = cell_repetitions(
                cell,
                float(gemnet.cutoff),
                int(gemnet.max_cell_images_per_dim),
            )
            periodic_images = int(
                (2 * repetitions[0] + 1)
                * (2 * repetitions[1] + 1)
                * (2 * repetitions[2] + 1)
            )
            neighbor_counts = torch.bincount(
                graph["edge_index"][1],
                minlength=n_atoms,
            )
            rows.append(
                {
                    "seed": state["seed"],
                    "state_index": state["state_index"],
                    "sampling_step": state["sampling_step"],
                    "phase": state["phase"],
                    "progress": state["progress"],
                    "num_atoms": n_atoms,
                    "rep_a1": repetitions[0],
                    "rep_a2": repetitions[1],
                    "rep_a3": repetitions[2],
                    "candidate_periodic_images": periodic_images,
                    "candidate_atom_pairs": n_atoms * n_atoms,
                    "candidate_pair_images": n_atoms * n_atoms * periodic_images,
                    "raw_edge_count": int(graph["raw_edge_index"].shape[1]),
                    "edge_count": int(graph["edge_index"].shape[1]),
                    "max_neighbors": int(neighbor_counts.max()),
                    "mean_neighbors": float(neighbor_counts.float().mean()),
                    "triplet_count": int(graph["id3_ba"].numel()),
                    "cell_volume": float(torch.abs(torch.linalg.det(cell[0])).cpu()),
                    "cell_condition_number": float(torch.linalg.cond(cell[0]).cpu()),
                }
            )
    return rows


def process_seed(generator, gemnet, seed: int) -> dict:
    final_dir = OUTPUT / f"seed_{seed}"
    status_path = final_dir / "status.json"
    if status_path.is_file():
        status = __import__("json").loads(status_path.read_text(encoding="utf-8"))
        if status.get("success") is True:
            return status
    if final_dir.exists():
        raise RuntimeError(f"refusing to overwrite incomplete shape state directory: {final_dir}")
    temporary = OUTPUT / f".seed_{seed}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        sampler = build_recording_sampler(generator, seed)
        condition = singleton_condition(seed)
        sampler.sample(condition, None)
        states = sampler.recorded_states
        expected = sampler.N * (sampler._n_steps_corrector + 1)
        if len(states) != expected:
            raise RuntimeError(f"recorded {len(states)} states, expected {expected}")
        save_states(temporary / "states.pt", states)
        rows = state_statistics(gemnet, states)
        write_csv(temporary / "shape_statistics.csv", rows)
        summary = {
            "success": True,
            "seed": seed,
            "physical_gpu": int(os.environ.get("SPG_PHYSICAL_GPU", "-1")),
            "num_states": len(states),
            "num_atoms": int(states[0]["num_atoms"].sum()),
            "finished_at": now(),
        }
        atomic_json(temporary / "status.json", summary)
        os.replace(temporary, final_dir)
        return summary
    except BaseException:
        atomic_json(
            temporary / "status.json",
            {
                "success": False,
                "seed": seed,
                "finished_at": now(),
                "error": traceback.format_exc(),
            },
        )
        raise


def main() -> int:
    args = parse_args()
    os.environ["SPG_PHYSICAL_GPU"] = str(args.physical_gpu)
    configure_determinism()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generator = build_c0_generator()
    gemnet = find_gemnet(generator.model)
    summaries = [process_seed(generator, gemnet, seed) for seed in args.seeds]
    atomic_json(
        OUTPUT / f"worker_gpu{args.physical_gpu}.json",
        {
            "success": True,
            "physical_gpu": args.physical_gpu,
            "seeds": args.seeds,
            "summaries": summaries,
            "finished_at": now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
