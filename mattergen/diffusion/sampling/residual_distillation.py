# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Teacher recording and safe runtime control for residual score distillation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch

from mattergen.diffusion.data.batched_data import BatchedData, SimpleBatchedData
from mattergen.diffusion.sampling.residual_adapter import (
    RISK_FEATURE_NAMES,
    AdapterPrediction,
    FieldwiseResidualAdapter,
    ResidualAdapterArchitecture,
    all_prediction_tensors_finite,
)


TEACHER_SCHEMA_VERSION = 1
ADAPTER_CHECKPOINT_VERSION = 1
CONTROLLER_MODES = ("teacher", "reuse", "adapter")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_copy(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    detached = value.detach().to(device="cpu")
    if detached.is_floating_point():
        detached = detached.to(dtype=dtype)
    return detached.contiguous()


class TeacherShardWriter:
    """Write compact, ragged teacher pairs without serializing model batches."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        seed: int,
        split: str,
        shard_size: int = 64,
        storage_dtype: str = "bfloat16",
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.seed = int(seed)
        self.split = str(split)
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("teacher shard_size must be positive")
        dtype_by_name = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        if storage_dtype not in dtype_by_name:
            raise ValueError(f"storage_dtype must be one of {tuple(dtype_by_name)}")
        self.storage_dtype_name = storage_dtype
        self.storage_dtype = dtype_by_name[storage_dtype]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        if self.manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite teacher manifest: {self.manifest_path}")
        if list(self.output_dir.glob("shard_*.pt")):
            raise FileExistsError(f"refusing to mix teacher shards in {self.output_dir}")
        self._buffer: list[dict[str, Any]] = []
        self._shards: list[dict[str, Any]] = []
        self._record_count = 0
        self._closed = False

    def add(
        self,
        *,
        sampling_step: int,
        t: torch.Tensor,
        progress: float,
        x_before: BatchedData,
        score_before: BatchedData,
        x_after: BatchedData,
        score_after: BatchedData,
    ) -> None:
        if self._closed:
            raise RuntimeError("teacher writer is already closed")
        batch_size = x_after.get_batch_size()
        atom_index = x_after.get_batch_idx("pos")
        if atom_index is None:
            atom_index = torch.arange(batch_size, device=x_after["pos"].device)
        record = {
            "sampling_step": int(sampling_step),
            "progress": float(progress),
            "t": _storage_copy(t.reshape(-1), torch.float32),
            "num_atoms": _storage_copy(x_after["num_atoms"].reshape(-1), torch.float32),
            "atom_structure_index": _storage_copy(atom_index.long(), torch.float32),
            "x_before_pos": _storage_copy(x_before["pos"], self.storage_dtype),
            "x_after_pos": _storage_copy(x_after["pos"], self.storage_dtype),
            "x_before_cell": _storage_copy(x_before["cell"], self.storage_dtype),
            "x_after_cell": _storage_copy(x_after["cell"], self.storage_dtype),
            "x_before_atomic_numbers": _storage_copy(
                x_before["atomic_numbers"].long(), torch.float32
            ),
            "x_after_atomic_numbers": _storage_copy(
                x_after["atomic_numbers"].long(), torch.float32
            ),
        }
        for field in ("pos", "cell", "atomic_numbers"):
            before = score_before[field]
            after = score_after[field]
            record[f"score_before_{field}"] = _storage_copy(before, self.storage_dtype)
            record[f"score_after_{field}"] = _storage_copy(after, self.storage_dtype)
            record[f"score_residual_{field}"] = _storage_copy(
                after.float() - before.float(), self.storage_dtype
            )
        if record["t"].numel() not in (1, batch_size):
            raise ValueError("teacher timestep shape does not match batch size")
        self._buffer.append(record)
        self._record_count += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        atom_offsets = [0]
        structure_offsets = [0]
        atom_structure_parts = []
        atom_fields = (
            "x_before_pos",
            "x_after_pos",
            "x_before_atomic_numbers",
            "x_after_atomic_numbers",
            "score_before_pos",
            "score_after_pos",
            "score_residual_pos",
            "score_before_atomic_numbers",
            "score_after_atomic_numbers",
            "score_residual_atomic_numbers",
        )
        structure_fields = (
            "t",
            "num_atoms",
            "x_before_cell",
            "x_after_cell",
            "score_before_cell",
            "score_after_cell",
            "score_residual_cell",
        )
        for record in self._buffer:
            atom_offsets.append(atom_offsets[-1] + record["x_after_pos"].shape[0])
            structure_offsets.append(structure_offsets[-1] + record["x_after_cell"].shape[0])
            atom_structure_parts.append(
                record["atom_structure_index"].long() + structure_offsets[-2]
            )
        payload: dict[str, Any] = {
            "schema_version": TEACHER_SCHEMA_VERSION,
            "seed": self.seed,
            "split": self.split,
            "record_ptr_atoms": torch.tensor(atom_offsets, dtype=torch.long),
            "record_ptr_structures": torch.tensor(structure_offsets, dtype=torch.long),
            "sampling_step": torch.tensor(
                [record["sampling_step"] for record in self._buffer], dtype=torch.long
            ),
            "progress": torch.tensor(
                [record["progress"] for record in self._buffer], dtype=torch.float32
            ),
            "atom_structure_index": torch.cat(atom_structure_parts),
        }
        for field in atom_fields:
            payload[field] = torch.cat([record[field] for record in self._buffer], dim=0)
        for field in structure_fields:
            values = [record[field] for record in self._buffer]
            if field == "t":
                values = [
                    value.expand(record["x_after_cell"].shape[0])
                    if value.numel() == 1
                    else value
                    for value, record in zip(values, self._buffer)
                ]
            payload[field] = torch.cat(values, dim=0)

        shard_index = len(self._shards)
        path = self.output_dir / f"shard_{shard_index:05d}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite teacher shard: {path}")
        torch.save(payload, path)
        self._shards.append(
            {
                "path": path.name,
                "records": len(self._buffer),
                "atoms": atom_offsets[-1],
                "structures": structure_offsets[-1],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        self._buffer = []

    def close(self, *, error: BaseException | None = None) -> None:
        if self._closed:
            return
        self._flush()
        manifest = {
            "schema_version": TEACHER_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "split": self.split,
            "storage_dtype": self.storage_dtype_name,
            "records": self._record_count,
            "shard_size": self.shard_size,
            "shards": self._shards,
            "completed": error is None,
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "stored_fields": [
                "x_before_corrector(pos,cell,atomic_numbers)",
                "x_after_corrector(pos,cell,atomic_numbers)",
                "score_before(pos,cell,atomic_numbers)",
                "score_after(pos,cell,atomic_numbers)",
                "score_residual(pos,cell,atomic_numbers)",
                "seed/sampling_step/t/progress/num_atoms",
            ],
        }
        with self.manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        self._closed = True


@dataclass(frozen=True)
class LoadedAdapter:
    model: FieldwiseResidualAdapter
    calibration: Mapping[str, Any]
    metadata: Mapping[str, Any]


def load_adapter_checkpoint(path: str | Path, device: torch.device) -> LoadedAdapter:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != ADAPTER_CHECKPOINT_VERSION:
        raise ValueError(f"unsupported adapter checkpoint: {checkpoint_path}")
    architecture = ResidualAdapterArchitecture(**payload["architecture"])
    model = FieldwiseResidualAdapter(architecture=architecture)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device)
    return LoadedAdapter(
        model=model,
        calibration=payload.get("calibration", {}),
        metadata=payload.get("metadata", {}),
    )


def save_adapter_checkpoint(
    path: str | Path,
    *,
    model: FieldwiseResidualAdapter,
    calibration: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite adapter checkpoint: {output_path}")
    payload = {
        "checkpoint_version": ADAPTER_CHECKPOINT_VERSION,
        "architecture": model.architecture.as_dict(),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "calibration": dict(calibration or {}),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output_path)
    return output_path


def _calibrated_risk(
    risk_features: torch.Tensor, calibration: Mapping[str, Any]
) -> torch.Tensor | None:
    required = ("risk_feature_mean", "risk_feature_std", "risk_linear_weights")
    if not all(key in calibration for key in required):
        return None
    device = risk_features.device
    dtype = risk_features.dtype
    mean = torch.as_tensor(calibration["risk_feature_mean"], device=device, dtype=dtype)
    std = torch.as_tensor(calibration["risk_feature_std"], device=device, dtype=dtype)
    weights = torch.as_tensor(calibration["risk_linear_weights"], device=device, dtype=dtype)
    if mean.shape != (len(RISK_FEATURE_NAMES),) or std.shape != mean.shape:
        return None
    if weights.shape != mean.shape:
        return None
    bias = float(calibration.get("risk_linear_bias", 0.0))
    return ((risk_features - mean) / std.clamp_min(1.0e-6)) @ weights + bias


class ResidualDistillationController:
    """Choose Adapter/reuse or an exact score using an auditable fallback."""

    def __init__(self, config: Mapping[str, Any], *, device: torch.device) -> None:
        self.config = dict(config)
        self.mode = str(self.config.get("mode", "adapter")).lower()
        if self.mode not in CONTROLLER_MODES:
            raise ValueError(f"residual controller mode must be one of {CONTROLLER_MODES}")
        self.device = device
        self.coverage_target = float(self.config.get("coverage_target", 1.0))
        if not 0.0 <= self.coverage_target <= 1.0:
            raise ValueError("coverage_target must be in [0, 1]")
        self.force_fallback = bool(self.config.get("force_fallback", False))
        self.trace_path = (
            Path(str(self.config["trace_path"])).expanduser().resolve()
            if self.config.get("trace_path")
            else None
        )
        self.adapter: FieldwiseResidualAdapter | None = None
        self.calibration: Mapping[str, Any] = {}
        if self.mode == "adapter":
            checkpoint_path = self.config.get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError("adapter mode requires checkpoint_path")
            loaded = load_adapter_checkpoint(checkpoint_path, device=device)
            self.adapter = loaded.model
            self.calibration = loaded.calibration
        self.writer: TeacherShardWriter | None = None
        if self.mode == "teacher":
            if not self.config.get("teacher_output_dir"):
                raise ValueError("teacher mode requires teacher_output_dir")
            if self.config.get("sample_seed") is None:
                raise ValueError("teacher mode requires sample_seed")
            self.writer = TeacherShardWriter(
                output_dir=self.config["teacher_output_dir"],
                seed=int(self.config["sample_seed"]),
                split=str(self.config.get("teacher_split", "unspecified")),
                shard_size=int(self.config.get("teacher_shard_size", 64)),
                storage_dtype=str(self.config.get("teacher_storage_dtype", "bfloat16")),
            )
        self.reset()

    def reset(self) -> None:
        self.score_opportunities = 0
        self.adapter_calls = 0
        self.fallback_calls = 0
        self.saved_score_calls = 0
        self.adapter_seconds = 0.0
        self.exact_fallback_seconds = 0.0
        self._trace_rows: list[dict[str, Any]] = []

    @property
    def metrics(self) -> dict[str, float | int]:
        coverage = self.saved_score_calls / max(self.score_opportunities, 1)
        return {
            "score_opportunities": self.score_opportunities,
            "adapter_calls": self.adapter_calls,
            "fallback_calls": self.fallback_calls,
            "saved_score_calls": self.saved_score_calls,
            "adapter_coverage": coverage,
            "adapter_seconds": self.adapter_seconds,
            "exact_fallback_seconds": self.exact_fallback_seconds,
        }

    def _coverage_threshold(self) -> float | None:
        if self.coverage_target >= 1.0:
            return math.inf
        thresholds = self.calibration.get("coverage_thresholds", {})
        if not thresholds:
            return None
        candidates = [(abs(float(key) - self.coverage_target), float(value)) for key, value in thresholds.items()]
        return min(candidates, key=lambda item: item[0])[1]

    def _exact(
        self,
        exact_score_fn: Callable[[BatchedData, torch.Tensor], BatchedData],
        x_after: BatchedData,
        t: torch.Tensor,
    ) -> BatchedData:
        start = time.perf_counter()
        score = exact_score_fn(x_after, t)
        self.exact_fallback_seconds += time.perf_counter() - start
        return score

    def score_after_corrector(
        self,
        *,
        exact_score_fn: Callable[[BatchedData, torch.Tensor], BatchedData],
        x_before: BatchedData,
        score_before: BatchedData,
        x_after: BatchedData,
        t: torch.Tensor,
        sampling_step: int,
        progress: float,
    ) -> BatchedData:
        self.score_opportunities += 1
        risk_value: float | None = None
        threshold: float | None = None
        fallback_reason: str | None = None
        prediction: AdapterPrediction | None = None

        if self.mode == "teacher":
            score_after = self._exact(exact_score_fn, x_after, t)
            assert self.writer is not None
            self.writer.add(
                sampling_step=sampling_step,
                t=t,
                progress=progress,
                x_before=x_before,
                score_before=score_before,
                x_after=x_after,
                score_after=score_after,
            )
            fallback_reason = "teacher_exact"
        elif self.force_fallback:
            self.fallback_calls += 1
            score_after = self._exact(exact_score_fn, x_after, t)
            fallback_reason = "forced_exact_fallback"
        elif self.mode == "reuse":
            score_after = score_before
            self.saved_score_calls += 1
        else:
            assert self.adapter is not None
            start = time.perf_counter()
            prediction = self.adapter(
                x_before=x_before,
                x_after=x_after,
                score_before=score_before,
                t=t,
                progress=progress,
            )
            self.adapter_seconds += time.perf_counter() - start
            self.adapter_calls += 1
            if not all_prediction_tensors_finite(prediction):
                fallback_reason = "non_finite_adapter_output"
            else:
                risks = _calibrated_risk(prediction.risk_features, self.calibration)
                threshold = self._coverage_threshold()
                if risks is None and self.coverage_target < 1.0:
                    fallback_reason = "missing_or_invalid_calibration"
                elif risks is not None and not bool(torch.isfinite(risks).all().item()):
                    fallback_reason = "non_finite_uncertainty"
                else:
                    risk_value = None if risks is None else float(risks.max().item())
                    if threshold is not None and risk_value is not None and risk_value > threshold:
                        fallback_reason = "risk_above_threshold"
            if fallback_reason is None:
                score_after = prediction.score
                self.saved_score_calls += 1
            else:
                self.fallback_calls += 1
                score_after = self._exact(exact_score_fn, x_after, t)

        self._trace_rows.append(
            {
                "sampling_step": sampling_step,
                "progress": progress,
                "t": float(t.detach().reshape(-1)[0].cpu().item()),
                "mode": self.mode,
                "coverage_target": self.coverage_target,
                "risk": risk_value,
                "threshold": threshold,
                "used_adapter": fallback_reason is None and self.mode in ("adapter", "reuse"),
                "fallback_reason": fallback_reason,
            }
        )
        return score_after

    def close(self, *, error: BaseException | None = None) -> None:
        if self.writer is not None:
            self.writer.close(error=error)
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trace_path.exists():
            raise FileExistsError(f"refusing to overwrite residual trace: {self.trace_path}")
        with self.trace_path.open("x", newline="", encoding="utf-8") as stream:
            field_names = (
                "sampling_step",
                "progress",
                "t",
                "mode",
                "coverage_target",
                "risk",
                "threshold",
                "used_adapter",
                "fallback_reason",
            )
            writer = csv.DictWriter(stream, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(self._trace_rows)


def build_residual_controller(
    config: Mapping[str, Any] | None, *, device: torch.device
) -> ResidualDistillationController | None:
    if config is None or not bool(config.get("enabled", False)):
        return None
    return ResidualDistillationController(config, device=device)


def load_teacher_shard(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != TEACHER_SCHEMA_VERSION:
        raise ValueError(f"unsupported teacher shard: {path}")
    return payload


def iter_teacher_records(
    manifest_paths: list[str | Path], *, verify_sha256: bool = True
) -> Iterator[dict[str, Any]]:
    """Yield reconstructed lightweight batches one record at a time."""

    for manifest_value in manifest_paths:
        manifest_path = Path(manifest_value).expanduser().resolve()
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not manifest.get("completed"):
            raise ValueError(f"teacher run is incomplete: {manifest_path}")
        for shard_info in manifest["shards"]:
            shard_path = manifest_path.parent / shard_info["path"]
            if verify_sha256 and _sha256(shard_path) != shard_info["sha256"]:
                raise ValueError(f"teacher shard checksum mismatch: {shard_path}")
            shard = load_teacher_shard(shard_path)
            atom_ptr = shard["record_ptr_atoms"]
            structure_ptr = shard["record_ptr_structures"]
            for record_index in range(len(atom_ptr) - 1):
                atom_start, atom_end = int(atom_ptr[record_index]), int(atom_ptr[record_index + 1])
                structure_start = int(structure_ptr[record_index])
                structure_end = int(structure_ptr[record_index + 1])
                atom_index = (
                    shard["atom_structure_index"][atom_start:atom_end].long() - structure_start
                )
                num_atoms = shard["num_atoms"][structure_start:structure_end].long()
                x_batch_index = {
                    "pos": atom_index,
                    "atomic_numbers": atom_index,
                    "cell": None,
                    "num_atoms": None,
                }
                x_before = SimpleBatchedData(
                    data={
                        "pos": shard["x_before_pos"][atom_start:atom_end].float(),
                        "cell": shard["x_before_cell"][structure_start:structure_end].float(),
                        "atomic_numbers": shard["x_before_atomic_numbers"][atom_start:atom_end].long(),
                        "num_atoms": num_atoms,
                    },
                    batch_idx=x_batch_index,
                )
                x_after = SimpleBatchedData(
                    data={
                        "pos": shard["x_after_pos"][atom_start:atom_end].float(),
                        "cell": shard["x_after_cell"][structure_start:structure_end].float(),
                        "atomic_numbers": shard["x_after_atomic_numbers"][atom_start:atom_end].long(),
                        "num_atoms": num_atoms,
                    },
                    batch_idx=x_batch_index,
                )
                score_batch_index = {
                    "pos": atom_index,
                    "atomic_numbers": atom_index,
                    "cell": None,
                }
                scores = {}
                for prefix in ("before", "after", "residual"):
                    scores[prefix] = SimpleBatchedData(
                        data={
                            "pos": shard[f"score_{prefix}_pos"][atom_start:atom_end].float(),
                            "cell": shard[f"score_{prefix}_cell"][structure_start:structure_end].float(),
                            "atomic_numbers": shard[f"score_{prefix}_atomic_numbers"][atom_start:atom_end].float(),
                        },
                        batch_idx=score_batch_index,
                    )
                yield {
                    "seed": int(shard["seed"]),
                    "split": str(shard["split"]),
                    "sampling_step": int(shard["sampling_step"][record_index]),
                    "progress": float(shard["progress"][record_index]),
                    "t": shard["t"][structure_start:structure_end].float(),
                    "x_before": x_before,
                    "x_after": x_after,
                    "score_before": scores["before"],
                    "score_after": scores["after"],
                    "score_residual": scores["residual"],
                }
