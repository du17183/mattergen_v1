from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from research.crystalrepa_repro.common import REPORTS, RESULTS, atomic_json, now, sha256_file
from research.crystalrepa_repro.configuration import PROJECTION_KEY, load_r1_as_inference_model

TRAINING = RESULTS / "training/r1/checkpoints"
INFERENCE = RESULTS / "inference"


def main() -> None:
    candidates = []
    for path in TRAINING.glob("best-*.ckpt"):
        match = re.search(r"loss_val=([0-9.]+)\.ckpt$", path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("No best validation checkpoint exists")
    val_loss, best = min(candidates)
    INFERENCE.mkdir(parents=True, exist_ok=True)
    output = INFERENCE / "r1_inference.ckpt"
    trained = torch.load(best, map_location="cpu", weights_only=False)
    state = {key: value for key, value in trained["state_dict"].items() if not key.startswith(PROJECTION_KEY)}
    torch.save({"state_dict": state, "source_checkpoint": str(best), "source_val_loss": val_loss}, output)
    model = load_r1_as_inference_model(output, torch.device("cpu"))
    del model
    manifest = {
        "created_at": now(), "best_training_checkpoint": str(best),
        "best_validation_loss": val_loss, "inference_checkpoint": str(output),
        "inference_checkpoint_sha256": sha256_file(output), "projection_removed": True,
        "teacher_absent": True, "strict_official_architecture_load": True,
    }
    atomic_json(REPORTS / "inference_checkpoint_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
