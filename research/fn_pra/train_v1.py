from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from mattergen.common.data.datamodule import CrystDataModule
from mattergen.common.data.dataset_transform import filter_sparse_properties
from mattergen.common.data.transform import set_chemical_system_string, symmetrize_lattice
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.fn_pra.data import RepaCrystalDataset
from research.fn_pra.phase1_common import REPORTS, RESULTS, atomic_json, now, set_stage
from research.fn_pra.validate_v1_integration import CHECKPOINT_ROOT, is_new_parameter, repa_config


DATASET_ROOT = Path("/data/dxl/datasets/cache/mp_20")
TEACHER_ROOT = Path("/data/dxl/data/fn_pra_teacher_cache")
TRAINING_ROOT = RESULTS / "training"


class ExactStepCheckpoint(Callback):
    def __init__(self, output_dir: Path, steps: list[int]) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.steps = set(steps)
        self.saved: set[int] = set()

    def _save(self, trainer: pl.Trainer, step: int) -> None:
        if step not in self.steps or step in self.saved:
            return
        path = self.output_dir / "checkpoints" / f"step={step:06d}.ckpt"
        trainer.save_checkpoint(path)
        self.saved.add(step)
        if trainer.is_global_zero:
            manifest = {
                "updated_at": now(),
                "saved_steps": sorted(self.saved),
                "paths": {
                    str(value): str(self.output_dir / "checkpoints" / f"step={value:06d}.ckpt")
                    for value in sorted(self.saved)
                },
            }
            atomic_json(self.output_dir / "checkpoint_progress.json", manifest)

    def on_train_start(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        self._save(trainer, 0)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        _outputs,
        _batch,
        _batch_idx: int,
    ) -> None:
        step = trainer.global_step
        self._save(trainer, step)
        if trainer.is_global_zero and (step <= 10 or step % 50 == 0):
            set_stage(
                "v1_smoke_training",
                "running",
                f"Static REPA V1 smoke training step {step}/{trainer.max_steps}.",
                {"step": step, "max_steps": trainer.max_steps},
            )

    def on_train_end(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        self._save(trainer, trainer.global_step)


def build_datasets() -> tuple[RepaCrystalDataset, RepaCrystalDataset]:
    common = {
        "properties": ["dft_mag_density"],
        "transforms": [symmetrize_lattice, set_chemical_system_string],
        "dataset_transforms": [filter_sparse_properties],
    }
    train = RepaCrystalDataset.from_cache_path(
        cache_path=str(DATASET_ROOT / "train"),
        teacher_cache_path=str(TEACHER_ROOT / "train"),
        **common,
    )
    val = RepaCrystalDataset.from_cache_path(
        cache_path=str(DATASET_ROOT / "val"),
        teacher_cache_path=str(TEACHER_ROOT / "val"),
        **common,
    )
    return train, val


def initialize_model(learning_rate: float):
    info = MatterGenCheckpointInfo(str(CHECKPOINT_ROOT))
    base_config = info.config
    lightning_config = repa_config(base_config)
    with open_dict(lightning_config.optimizer_partial):
        lightning_config.optimizer_partial.lr = learning_rate
    model = instantiate(lightning_config)
    checkpoint = torch.load(info.checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.unexpected_keys or len(incompatible.missing_keys) != 6:
        raise RuntimeError(f"Unexpected A0→V1 state loading result: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if is_new_parameter(name):
            parameter.requires_grad_(True)
    return model, info, base_config, lightning_config, checkpoint


def verify_frozen_checkpoint(trained_path: Path, official_state: dict[str, torch.Tensor]) -> dict:
    trained = torch.load(trained_path, map_location="cpu")["state_dict"]
    mismatches = []
    for name, value in official_state.items():
        if name not in trained or not torch.equal(value.cpu(), trained[name].cpu()):
            mismatches.append(name)
    return {
        "official_keys": len(official_state),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": not mismatches,
    }


def training_summary(output: Path, max_steps: int, frozen_report: dict) -> dict:
    metric_paths = sorted((output / "csv").glob("version_*/metrics.csv"))
    metrics_path = metric_paths[-1] if metric_paths else None
    summary = {
        "schema_version": 1,
        "created_at": now(),
        "max_steps": max_steps,
        "metrics_csv": str(metrics_path) if metrics_path else None,
        "frozen_backbone": frozen_report,
    }
    if metrics_path is not None:
        frame = pd.read_csv(metrics_path)
        for column in (
            "loss_train_step",
            "loss_diffusion_train_step",
            "loss_repa_alignment_train_step",
            "loss_val",
            "loss_diffusion_val",
            "loss_repa_alignment_val",
        ):
            if column in frame:
                values = frame[column].dropna()
                if not values.empty:
                    summary[column] = {
                        "first": float(values.iloc[0]),
                        "last": float(values.iloc[-1]),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "finite": bool(torch.isfinite(torch.tensor(values.to_numpy())).all()),
                    }
    summary["passed"] = bool(
        frozen_report["passed"]
        and all(item.get("finite", True) for item in summary.values() if isinstance(item, dict))
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--run-name", default="v1_smoke")
    args = parser.parse_args()
    if args.max_steps > 1000 and args.run_name == "v1_smoke":
        raise ValueError("V1-S may not exceed 1000 steps")
    pl.seed_everything(20260725, workers=True)
    torch.set_float32_matmul_precision("high")
    output = TRAINING_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        set_stage(
            "v1_smoke_training",
            "running",
            f"Launching {args.max_steps}-step V1-S with {args.devices} GPUs.",
            vars(args),
        )
    model, info, base_config, lightning_config, checkpoint = initialize_model(args.learning_rate)
    train_dataset, val_dataset = build_datasets()
    datamodule = CrystDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=OmegaConf.create({"train": args.num_workers, "val": args.num_workers}),
        batch_size=OmegaConf.create(
            {"train": args.train_batch_size, "val": args.val_batch_size}
        ),
    )
    full_config = deepcopy(base_config)
    with open_dict(full_config):
        full_config.lightning_module = lightning_config
        full_config.fn_pra = OmegaConf.create(
            {
                "phase": "V1-S",
                "teacher": "CHGNet 0.3.0",
                "teacher_checkpoint_sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
                "adapter_rank": 16,
                "projection_dim": 128,
                "alignment_weight": 0.1,
                "alignment_temperature": 0.07,
            }
        )
    OmegaConf.save(full_config, output / "config.yaml", resolve=True)
    milestones = sorted({0, 100, 250, 500, 1000, args.max_steps} & set(range(args.max_steps + 1)))
    exact = ExactStepCheckpoint(output, milestones)
    best = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best-{step:06d}-{loss_val:.6f}",
        monitor="loss_val",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=250,
    )
    logger = CSVLogger(save_dir=output, name="csv")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=32,
        max_steps=args.max_steps,
        max_epochs=100000,
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        callbacks=[exact, best],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=min(250, args.max_steps),
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )
    trainer.fit(model=model, datamodule=datamodule)
    if trainer.is_global_zero:
        final_path = output / "checkpoints" / f"step={args.max_steps:06d}.ckpt"
        if not final_path.exists():
            trainer.save_checkpoint(final_path)
        frozen = verify_frozen_checkpoint(final_path, checkpoint["state_dict"])
        summary = training_summary(output, args.max_steps, frozen)
        summary.update(
            {
                "run_name": args.run_name,
                "devices": args.devices,
                "checkpoint": str(final_path),
                "official_checkpoint": info.checkpoint_path,
            }
        )
        atomic_json(output / "training_summary.json", summary)
        atomic_json(REPORTS / "v1_smoke_training_summary.json", summary)
        status = "success" if summary["passed"] else "failed"
        set_stage(
            "v1_smoke_training",
            status,
            f"V1-S reached {args.max_steps} steps; frozen backbone passed={frozen['passed']}.",
            summary,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
