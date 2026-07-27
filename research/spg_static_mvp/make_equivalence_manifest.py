from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.spg_static_mvp.common import RESULTS, atomic_json, read_json


def main() -> int:
    bucket = read_json(RESULTS / "selected_bucket.json")
    frames = []
    for path in sorted((RESULTS / "shape_states").glob("seed_*/shape_statistics.csv")):
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    mask = frame["num_atoms"].between(
        bucket["num_atoms_min"],
        bucket["num_atoms_max"],
    )
    mask &= (
        frame[["rep_a1", "rep_a2", "rep_a3"]]
        <= [
            bucket["max_rep_a1"],
            bucket["max_rep_a2"],
            bucket["max_rep_a3"],
        ]
    ).all(axis=1)
    selected = frame.loc[mask].sort_values(["seed", "state_index"]).reset_index(
        drop=True
    )
    if len(selected) < 10_000:
        raise RuntimeError(f"bucket has only {len(selected)} validation states")
    # Evenly sample the full eligible trajectory pool rather than taking the
    # first 10k rows. This preserves all seeds and covers the full noise
    # schedule (including both predictor and corrector states) deterministically.
    selection_index = np.linspace(
        0,
        len(selected) - 1,
        num=10_000,
        dtype=np.int64,
    )
    selected = selected.iloc[selection_index].copy()
    manifest = [
        {"seed": int(row.seed), "state_index": int(row.state_index)}
        for row in selected.itertuples(index=False)
    ]
    output = RESULTS / "equivalence/manifest.json"
    atomic_json(output, manifest)
    metadata = {
        "states": len(manifest),
        "eligible_states": int(mask.sum()),
        "seed_count": int(selected["seed"].nunique()),
        "seeds": sorted(int(value) for value in selected["seed"].unique()),
        "num_atoms": {
            str(int(key)): int(value)
            for key, value in selected["num_atoms"].value_counts().sort_index().items()
        },
        "state_index_min": int(selected["state_index"].min()),
        "state_index_max": int(selected["state_index"].max()),
        "sampling_step_min": int(selected["sampling_step"].min()),
        "sampling_step_max": int(selected["sampling_step"].max()),
        "phase_counts": {
            str(key): int(value)
            for key, value in selected["phase"].value_counts().sort_index().items()
        },
        "path": str(output),
    }
    atomic_json(RESULTS / "equivalence/manifest_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
