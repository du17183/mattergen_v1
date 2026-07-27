from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate

from mattergen.common.data.collate import collate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.pc_sampler import _sample_prior
from mattergen.generator import CrystalGenerator
from research.mps_fastgate.common import (
    LOGS,
    MPS_LOG,
    MPS_PIPE,
    RESULTS,
    atomic_json,
    configure_environment,
    now,
)


CHECKPOINT = Path("/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")


def tensor_digest(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(json.dumps(list(cpu.shape)).encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def rng_digest() -> str:
    payload = {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(
            repr(
                (
                    np.random.get_state()[0],
                    np.random.get_state()[1].tolist(),
                    np.random.get_state()[2:],
                )
            ).encode()
        ).hexdigest(),
        "torch_cpu": tensor_digest(torch.get_rng_state()),
        "torch_cuda": [tensor_digest(value) for value in torch.cuda.get_rng_state_all()],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serialize_sample(sample, *, include_arrays: bool) -> dict[str, Any]:
    graph = sample.to_data_list()[0]
    fields = {
        "atomic_numbers": graph.atomic_numbers.detach().contiguous().cpu(),
        "pos": graph.pos.detach().contiguous().cpu(),
        "cell": graph.cell.detach().contiguous().cpu(),
    }
    hashes = {name: tensor_digest(value) for name, value in fields.items()}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    output: dict[str, Any] = {
        "hashes": hashes | {"final_structure": combined},
        "num_atoms": int(fields["atomic_numbers"].numel()),
    }
    if include_arrays:
        output["atomic_numbers"] = fields["atomic_numbers"].tolist()
        output["positions"] = fields["pos"].tolist()
        output["cell"] = fields["cell"].tolist()
    return output


def build_runtime():
    checkpoint = MatterGenCheckpointInfo(
        model_path=CHECKPOINT,
        load_epoch="last",
        strict_checkpoint_loading=True,
    )
    generator = CrystalGenerator(
        checkpoint_info=checkpoint,
        properties_to_condition_on={"dft_mag_density": 0.10},
        batch_size=1,
        num_batches=1,
        diffusion_guidance_factor=2.0,
        deterministic=True,
        guidance_schedule="constant",
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)
    return generator, sampling_config, sampler


def warm_runtime(generator, sampling_config, sampler) -> None:
    seed_everything(26999)
    condition_loader = generator.get_condition_loader(sampling_config)
    conditioning, mask = next(iter(condition_loader))
    conditioning = conditioning.to(generator.model.device)
    mask = {key: value.to(generator.model.device) for key, value in (mask or {}).items()}
    batch = _sample_prior(sampler._multi_corruption, conditioning, mask=mask)
    timestep = torch.full((batch.get_batch_size(),), 0.5, device=generator.model.device)
    unconditional = sampler._remove_conditioning_fn(batch)
    conditional = sampler._keep_conditioning_fn(batch)
    joint = collate([unconditional, conditional])
    for attribute, value in unconditional.items():
        if isinstance(value, list):
            joint[attribute] = unconditional[attribute] + conditional[attribute]
    with torch.inference_mode():
        sampler.diffusion_module.score_fn(joint, torch.cat([timestep, timestep], dim=0))
    torch.cuda.synchronize()


def persistent_worker(
    physical_gpu: int,
    slot: int,
    config_id: str,
    mps_enabled: bool,
    active_thread_percentage: int | None,
    task_queue,
    result_queue,
    ready_queue,
) -> None:
    worker_id = f"gpu{physical_gpu}_slot{slot}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if mps_enabled:
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(MPS_PIPE)
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(MPS_LOG)
        assert active_thread_percentage is not None
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(active_thread_percentage)
    else:
        os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
        os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    configure_environment()
    log_path = LOGS / config_id / f"{worker_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_float32_matmul_precision("highest")
            torch.cuda.set_device(0)
            load_started = time.monotonic()
            generator, sampling_config, sampler = build_runtime()
            warm_runtime(generator, sampling_config, sampler)
            ready_queue.put(
                {
                    "ready": True,
                    "worker_id": worker_id,
                    "gpu": physical_gpu,
                    "slot": slot,
                    "load_warmup_seconds": time.monotonic() - load_started,
                    "mps_enabled": mps_enabled,
                    "active_thread_percentage": active_thread_percentage,
                }
            )
        except BaseException:
            ready_queue.put(
                {
                    "ready": False,
                    "worker_id": worker_id,
                    "gpu": physical_gpu,
                    "slot": slot,
                    "error": traceback.format_exc(),
                }
            )
            return

        while True:
            task = task_queue.get()
            if task is None:
                return
            seed = int(task["seed"])
            round_index = int(task["round"])
            started = time.monotonic()
            cpu_started = time.process_time()
            try:
                seed_everything(seed)
                random_tape_hash = rng_digest()
                if hasattr(sampler, "_sample_seed"):
                    sampler._sample_seed = seed
                if hasattr(sampler, "_run_id"):
                    sampler._run_id = f"{config_id}_round{round_index}_seed{seed}"
                condition_loader = generator.get_condition_loader(sampling_config)
                conditioning, mask = next(iter(condition_loader))
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    _sample, mean = sampler.sample(conditioning, mask)
                torch.cuda.synchronize()
                result_queue.put(
                    {
                        "success": True,
                        "config_id": config_id,
                        "seed": seed,
                        "round": round_index,
                        "worker_id": worker_id,
                        "gpu": physical_gpu,
                        "slot": slot,
                        "random_tape_hash": random_tape_hash,
                        "elapsed_seconds": time.monotonic() - started,
                        "cpu_seconds": time.process_time() - cpu_started,
                        "finished_monotonic": time.monotonic(),
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "sample": serialize_sample(
                            mean,
                            include_arrays=bool(task["record_output"]),
                        ),
                    }
                )
            except BaseException:
                result_queue.put(
                    {
                        "success": False,
                        "config_id": config_id,
                        "seed": seed,
                        "round": round_index,
                        "worker_id": worker_id,
                        "gpu": physical_gpu,
                        "slot": slot,
                        "elapsed_seconds": time.monotonic() - started,
                        "cpu_seconds": time.process_time() - cpu_started,
                        "finished_monotonic": time.monotonic(),
                        "error": traceback.format_exc(),
                    }
                )


class Telemetry:
    def __init__(self, gpus: tuple[int, ...]) -> None:
        self.gpus = set(gpus)
        self.rows: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                timestamp = time.time()
                for line in result.stdout.splitlines():
                    gpu, utilization, memory, power = [part.strip() for part in line.split(",")]
                    if int(gpu) in self.gpus:
                        self.rows.append(
                            {
                                "time": timestamp,
                                "gpu": int(gpu),
                                "utilization_percent": float(utilization),
                                "memory_used_mib": float(memory),
                                "power_w": float(power),
                            }
                        )
            except Exception:
                pass
            self.stop_event.wait(1.0)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"samples": 0}
        return {
            "samples": len(self.rows),
            "gpu_utilization_mean_percent": statistics.mean(row["utilization_percent"] for row in self.rows),
            "gpu_utilization_max_percent": max(row["utilization_percent"] for row in self.rows),
            "peak_memory_mib": max(row["memory_used_mib"] for row in self.rows),
            "power_mean_w": statistics.mean(row["power_w"] for row in self.rows),
        }


def stop_workers(processes: list[mp.Process], task_queue) -> None:
    for _ in processes:
        task_queue.put(None)
    for process in processes:
        process.join(timeout=180)
    lingering = [process.pid for process in processes if process.is_alive()]
    if lingering:
        raise RuntimeError(f"workers did not exit after cooperative sentinel: {lingering}")


def worker_completion_skew_seconds(rows: list[dict[str, Any]]) -> float:
    """Return the difference between the last completion time of each worker."""
    last_finish: dict[str, float] = {}
    for row in rows:
        worker_id = row["worker_id"]
        finished = float(row["finished_monotonic"])
        last_finish[worker_id] = max(finished, last_finish.get(worker_id, finished))
    if len(last_finish) < 2:
        return 0.0
    finishes = tuple(last_finish.values())
    return max(finishes) - min(finishes)


def run_configuration(
    *,
    config_id: str,
    mps_enabled: bool,
    workers_per_gpu: int,
    active_thread_percentage: int | None,
    gpus: tuple[int, ...],
    seeds: tuple[int, ...],
    rounds: int,
) -> dict[str, Any]:
    output_dir = RESULTS / "runs" / config_id
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("success_count") == len(seeds) * rounds:
            return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    ready_queue = ctx.Queue()
    processes = []
    for gpu in gpus:
        for slot in range(workers_per_gpu):
            process = ctx.Process(
                target=persistent_worker,
                args=(gpu, slot, config_id, mps_enabled, active_thread_percentage, task_queue, result_queue, ready_queue),
                name=f"mps_fastgate_{config_id}_g{gpu}_s{slot}",
            )
            process.start()
            processes.append(process)
    ready_rows = [ready_queue.get(timeout=900) for _ in processes]
    failures = [row for row in ready_rows if not row["ready"]]
    if failures:
        stop_workers(processes, task_queue)
        raise RuntimeError(f"worker readiness failure: {failures}")

    telemetry = Telemetry(gpus)
    telemetry.start()
    rows = []
    round_summaries = []
    try:
        for round_index in range(1, rounds + 1):
            for seed in seeds:
                task_queue.put(
                    {
                        "seed": seed,
                        "round": round_index,
                        "record_output": round_index == 1,
                    }
                )
            round_started = time.monotonic()
            round_rows = [result_queue.get(timeout=1800) for _ in seeds]
            round_wall = time.monotonic() - round_started
            rows.extend(round_rows)
            for row in round_rows:
                if round_index == 1:
                    atomic_json(output_dir / f"seed_{row['seed']}.json", row)
            success_rows = [row for row in round_rows if row["success"]]
            round_summaries.append(
                {
                    "round": round_index,
                    "wall_seconds": round_wall,
                    "samples_per_hour": len(success_rows) * 3600.0 / round_wall,
                    "success_count": len(success_rows),
                    "failure_count": len(round_rows) - len(success_rows),
                    "p50_latency_seconds": statistics.median(row["elapsed_seconds"] for row in success_rows),
                    "p95_latency_seconds": float(np.percentile([row["elapsed_seconds"] for row in success_rows], 95)),
                    "worker_finish_spread_seconds": worker_completion_skew_seconds(success_rows),
                    "cpu_utilization_equivalent_percent": sum(row["cpu_seconds"] for row in success_rows) / round_wall * 100.0,
                }
            )
            atomic_json(output_dir / f"round_{round_index:02d}.json", round_summaries[-1])
    finally:
        telemetry.stop()
        stop_workers(processes, task_queue)

    success_rows = [row for row in rows if row["success"]]
    task_counts = Counter(row["worker_id"] for row in success_rows)
    first_round = [row for row in success_rows if row["round"] == 1]
    reference = {row["seed"]: row for row in first_round}
    within_config_bitwise = all(
        row["random_tape_hash"] == reference[row["seed"]]["random_tape_hash"]
        and row["sample"]["hashes"] == reference[row["seed"]]["sample"]["hashes"]
        for row in success_rows
    )
    summary = {
        "completed_at": now(),
        "config_id": config_id,
        "mps_enabled": mps_enabled,
        "workers_per_gpu": workers_per_gpu,
        "active_thread_percentage": active_thread_percentage,
        "gpus": list(gpus),
        "seeds": list(seeds),
        "round_count": rounds,
        "success_count": len(success_rows),
        "failure_count": len(rows) - len(success_rows),
        "rounds": round_summaries,
        "throughput_median_samples_per_hour": statistics.median(row["samples_per_hour"] for row in round_summaries),
        "p50_latency_median_seconds": statistics.median(row["p50_latency_seconds"] for row in round_summaries),
        "p95_latency_median_seconds": statistics.median(row["p95_latency_seconds"] for row in round_summaries),
        "cpu_utilization_median_percent": statistics.median(row["cpu_utilization_equivalent_percent"] for row in round_summaries),
        "worker_finish_spread_median_seconds": statistics.median(row["worker_finish_spread_seconds"] for row in round_summaries),
        "worker_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in success_rows),
        "worker_peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in success_rows),
        "within_config_bitwise": within_config_bitwise,
        "task_counts_per_worker": dict(sorted(task_counts.items())),
        "telemetry": telemetry.summary(),
        "ready": ready_rows,
        "result_index": [
            {
                "seed": row["seed"],
                "round": row["round"],
                "worker_id": row["worker_id"],
                "random_tape_hash": row["random_tape_hash"],
                "final_structure_hash": row["sample"]["hashes"]["final_structure"],
                "atomic_numbers_hash": row["sample"]["hashes"]["atomic_numbers"],
                "positions_hash": row["sample"]["hashes"]["pos"],
                "cell_hash": row["sample"]["hashes"]["cell"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for row in success_rows
        ],
    }
    atomic_json(summary_path, summary)
    return summary
