from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from mattergen.common.data.datamodule import CrystDataModule
from mattergen.common.data.dataset_transform import filter_sparse_properties
from mattergen.common.data.transform import set_chemical_system_string, symmetrize_lattice
from mattergen.crystalrepa.data import RepaCrystalDataset
from research.crystalrepa_repro.common import REPORTS, RESULTS, atomic_json, now, set_stage
from research.crystalrepa_repro.configuration import initialize_training_model

DATASET_ROOT = Path("/data/dxl/datasets/cache/mp_20")
TEACHER_ROOT = Path("/data/dxl/data/crystalrepa_teacher_cache")
TRAINING_ROOT = RESULTS / "training/r1"
MILESTONES = (0, 1000, 2500, 5000, 7500, 10000)


class ExactStepCheckpoint(Callback):
    def __init__(self, output_dir: Path, max_steps: int) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.steps = {step for step in MILESTONES if step <= max_steps}
        self.saved = {
            step for step in self.steps
            if (output_dir / "checkpoints" / f"step={step:06d}.ckpt").exists()
        }

    def _save(self, trainer: pl.Trainer, step: int) -> None:
        if step not in self.steps or step in self.saved:
            return
        path = self.output_dir / "checkpoints" / f"step={step:06d}.ckpt"
        trainer.save_checkpoint(path)
        self.saved.add(step)
        if trainer.is_global_zero:
            atomic_json(self.output_dir / "checkpoint_progress.json", {
                "updated_at": now(),
                "saved_steps": sorted(self.saved),
                "paths": {str(value): str(self.output_dir / "checkpoints" / f"step={value:06d}.ckpt") for value in sorted(self.saved)},
            })

    def on_train_start(self, trainer: pl.Trainer, _module: pl.LightningModule) -> None:
        self._save(trainer, trainer.global_step)

    def on_train_batch_end(self, trainer: pl.Trainer, _module: pl.LightningModule, _outputs: Any, _batch: Any, _batch_idx: int) -> None:
        self._save(trainer, trainer.global_step)
        step = trainer.global_step
        if trainer.is_global_zero and (step <= 10 or step % 25 == 0):
            stage = "training_smoke" if trainer.max_steps <= 1000 else "training_decision"
            set_stage(stage, "running", f"R1 full-backbone training step {step}/{trainer.max_steps}.", {"step": step, "max_steps": trainer.max_steps})

    def on_train_end(self, trainer: pl.Trainer, _module: pl.LightningModule) -> None:
        self._save(trainer, trainer.global_step)


class RuntimeTelemetry(Callback):
    def __init__(self, output: Path, local_batch_size: int) -> None:
        super().__init__()
        self.output = output
        self.local_batch_size = local_batch_size
        self.started: float | None = None
        self.batch_seconds: list[float] = []
        self.samples = 0
        self.ddp_comm_seconds = 0.0
        self._hook_registered = False

    def on_train_start(self, trainer: pl.Trainer, _module: pl.LightningModule) -> None:
        torch.cuda.reset_peak_memory_stats()
        wrapped = trainer.strategy.model
        if isinstance(wrapped, torch.nn.parallel.DistributedDataParallel):
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            callback = self
            def timed_allreduce(process_group, bucket):
                started = time.perf_counter()
                future = default_hooks.allreduce_hook(process_group, bucket)
                def record(result):
                    callback.ddp_comm_seconds += time.perf_counter() - started
                    return result.value()
                return future.then(record)
            wrapped.register_comm_hook(dist.group.WORLD, timed_allreduce)
            self._hook_registered = True

    def on_train_batch_start(self, _trainer: pl.Trainer, _module: pl.LightningModule, _batch: Any, _index: int) -> None:
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer: pl.Trainer, _module: pl.LightningModule, _outputs: Any, _batch: Any, _index: int) -> None:
        if self.started is not None:
            self.batch_seconds.append(time.perf_counter() - self.started)
            self.samples += self.local_batch_size * trainer.world_size
        if trainer.is_global_zero and trainer.global_step % 25 == 0:
            self.write(trainer)

    def write(self, trainer: pl.Trainer) -> None:
        elapsed = sum(self.batch_seconds)
        atomic_json(self.output / "runtime_telemetry.json", {
            "updated_at": now(),
            "global_step": trainer.global_step,
            "microbatches_observed": len(self.batch_seconds),
            "samples_observed_global": self.samples,
            "training_batch_wall_seconds": elapsed,
            "samples_per_second": self.samples / elapsed if elapsed else None,
            "mean_microbatch_seconds": elapsed / len(self.batch_seconds) if self.batch_seconds else None,
            "peak_vram_bytes_local_rank": torch.cuda.max_memory_allocated(),
            "ddp_communication_time_seconds_local_rank": self.ddp_comm_seconds,
            "ddp_comm_hook_registered": self._hook_registered,
        })

    def on_train_end(self, trainer: pl.Trainer, _module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            self.write(trainer)


def build_datasets() -> tuple[RepaCrystalDataset, RepaCrystalDataset]:
    common = {
        "properties": [],
        "transforms": [symmetrize_lattice, set_chemical_system_string],
        "dataset_transforms": [filter_sparse_properties],
    }
    return (
        RepaCrystalDataset.from_cache_path(cache_path=str(DATASET_ROOT / "train"), teacher_cache_path=str(TEACHER_ROOT / "train"), **common),
        RepaCrystalDataset.from_cache_path(cache_path=str(DATASET_ROOT / "val"), teacher_cache_path=str(TEACHER_ROOT / "val"), **common),
    )


def summarize(output: Path, max_steps: int, trainable_parameters: int) -> dict[str, Any]:
    metric_paths = sorted((output / "csv").glob("version_*/metrics.csv"))
    metrics_path = metric_paths[-1] if metric_paths else None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now(),
        "max_steps": max_steps,
        "trainable_parameters": trainable_parameters,
        "metrics_csv": str(metrics_path) if metrics_path else None,
    }
    if metrics_path:
        frame = pd.read_csv(metrics_path)
        for column in (
            "loss_train_step", "loss_diffusion_train_step", "loss_repa_alignment_train_step", "repa_positive_cosine_train_step",
            "loss_val", "loss_diffusion_val", "loss_repa_alignment_val", "repa_positive_cosine_val",
        ):
            if column in frame:
                values = frame[column].dropna()
                if not values.empty:
                    tensor = torch.as_tensor(values.to_numpy())
                    summary[column] = {
                        "first": float(values.iloc[0]), "last": float(values.iloc[-1]),
                        "min": float(values.min()), "max": float(values.max()),
                        "finite": bool(torch.isfinite(tensor).all()),
                    }
    summary["passed"] = all(item.get("finite", True) for item in summary.values() if isinstance(item, dict))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, choices=(1000, 10000), required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-val-batches", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.max_steps == 1000 and args.resume is not None:
        raise ValueError("The smoke run must start from the official checkpoint")
    if args.max_steps == 10000 and args.resume is None:
        default = TRAINING_ROOT / "checkpoints/step=001000.ckpt"
        if not default.exists():
            raise FileNotFoundError(f"Passed smoke checkpoint missing: {default}")
        args.resume = default
    pl.seed_everything(20260726, workers=True)
    torch.set_float32_matmul_precision("high")
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    (TRAINING_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    stage = "training_smoke" if args.max_steps == 1000 else "training_decision"
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        stage_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        set_stage(stage, "running", f"Launching R1 to step {args.max_steps}.", stage_args)
    model, info, _official, incompatible = initialize_training_model(args.learning_rate)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    train, val = build_datasets()
    datamodule = CrystDataModule(
        train_dataset=train, val_dataset=val,
        num_workers=OmegaConf.create({"train": args.num_workers, "val": args.num_workers}),
        batch_size=OmegaConf.create({"train": args.train_batch_size, "val": args.val_batch_size}),
    )
    config = {
        "created_at": now(), "base_checkpoint": info.checkpoint_path, "alignment_block": 2,
        "teacher": "CHGNet 0.3.0", "teacher_feature_dim": 64,
        "alignment_weight": 1.0, "alignment_temperature": 0.1,
        "training_mode": "full unconditional MatterGen backbone plus student projection",
        "devices": args.devices, "micro_batch_per_gpu": args.train_batch_size,
        "global_micro_batch": args.train_batch_size * args.devices,
        "gradient_accumulation": args.accumulate_grad_batches,
        "effective_batch": args.train_batch_size * args.devices * args.accumulate_grad_batches,
        "learning_rate": args.learning_rate, "max_steps": args.max_steps,
        "trainable_parameters": trainable_parameters, "total_parameters": total_parameters,
        "missing_projection_keys": list(incompatible.missing_keys),
    }
    atomic_json(TRAINING_ROOT / "training_config.json", config)
    callbacks: list[Callback] = [
        ExactStepCheckpoint(TRAINING_ROOT, args.max_steps),
        RuntimeTelemetry(TRAINING_ROOT, args.train_batch_size),
        ModelCheckpoint(
            dirpath=TRAINING_ROOT / "checkpoints", filename="best-{step:06d}-{loss_val:.6f}",
            monitor="loss_val", mode="min", save_top_k=1, save_last=True,
            every_n_train_steps=250,
        ),
    ]
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True), precision=32,
        max_steps=args.max_steps, max_epochs=100000,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=0.5, gradient_clip_algorithm="value",
        callbacks=callbacks, logger=CSVLogger(save_dir=TRAINING_ROOT, name="csv"),
        log_every_n_steps=10, val_check_interval=min(250, args.max_steps),
        check_val_every_n_epoch=None,
        limit_val_batches=args.limit_val_batches, num_sanity_val_steps=2,
        enable_progress_bar=True, enable_model_summary=True, deterministic=False,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(args.resume) if args.resume else None)
    if trainer.is_global_zero:
        final_path = TRAINING_ROOT / "checkpoints" / f"step={args.max_steps:06d}.ckpt"
        if not final_path.exists():
            trainer.save_checkpoint(final_path)
        summary = summarize(TRAINING_ROOT, args.max_steps, trainable_parameters)
        summary.update({"checkpoint": str(final_path), "official_checkpoint": info.checkpoint_path, "resume": str(args.resume) if args.resume else None})
        atomic_json(TRAINING_ROOT / f"training_summary_{args.max_steps}.json", summary)
        atomic_json(REPORTS / f"training_summary_{args.max_steps}.json", summary)
        status = "success" if summary["passed"] else "failed"
        set_stage(stage, status, f"R1 reached step {args.max_steps}; finite={summary['passed']}.", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
