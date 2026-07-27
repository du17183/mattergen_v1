"""Representative PyTorch Profiler and Nsight Systems probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mattergen.common.data.collate import collate
from research.spg_fastgate.common import (
    LOGS,
    PROJECT,
    PYTHON,
    RESULTS,
    atomic_json,
    base_environment,
    now,
    set_stage,
    sha256_file,
)
from research.spg_fastgate.generation import (
    build_generator,
    build_sampler,
    configure_determinism,
    make_condition,
)


TRACE_DIR = LOGS / "profiler"
RESULT = RESULTS / "profiler_breakdown.json"


@contextmanager
def scoped_model(model):
    """Attach temporary profiler scopes without changing model numerics."""

    restorers = []

    def wrap_method(module, name: str, scope: str) -> None:
        original = getattr(module, name)

        def wrapped(self, *args, __original=original, __scope=scope, **kwargs):
            with record_function(__scope):
                return __original(*args, **kwargs)

        setattr(module, name, types.MethodType(wrapped, module))
        restorers.append(lambda: setattr(module, name, original))

    gemnet = None
    for module in model.modules():
        if module.__class__.__name__ in {"GemNetT", "GemNetTCtrl"}:
            gemnet = module
            break
    if gemnet is None:
        raise RuntimeError("GemNet model not found")

    wrap_method(gemnet, "forward", "spg::gemnet_forward")
    wrap_method(gemnet, "generate_interaction_graph", "spg::interaction_graph_total")
    wrap_method(gemnet, "get_triplets", "spg::triplet_construction")
    for block in gemnet.int_blocks:
        wrap_method(block, "forward", "spg::gemnet_interaction_block")

    import mattergen.common.gemnet.gemnet as gemnet_module

    original_radius = gemnet_module.radius_graph_pbc

    def radius_wrapper(*args, **kwargs):
        with record_function("spg::periodic_neighbor_construction"):
            return original_radius(*args, **kwargs)

    gemnet_module.radius_graph_pbc = radius_wrapper
    restorers.append(lambda: setattr(gemnet_module, "radius_graph_pbc", original_radius))

    scatter_modules = []
    import mattergen.common.gemnet.layers.atom_update_block as atom_update_module

    scatter_modules.extend([gemnet_module, atom_update_module])
    try:
        import mattergen.common.gemnet.gemnet_ctrl as gemnet_ctrl_module

        scatter_modules.append(gemnet_ctrl_module)
    except ImportError:
        pass
    for target in scatter_modules:
        if not hasattr(target, "scatter"):
            continue
        original_scatter = target.scatter

        def scatter_wrapper(
            *args, __original=original_scatter, **kwargs
        ):
            with record_function("spg::scatter_segment"):
                return __original(*args, **kwargs)

        target.scatter = scatter_wrapper
        restorers.append(
            lambda target=target, original=original_scatter: setattr(
                target, "scatter", original
            )
        )
    try:
        yield gemnet
    finally:
        for restore in reversed(restorers):
            restore()


@contextmanager
def scoped_sampler(sampler):
    """Attach phase-level scopes to CFG, Predictor, and Corrector calls."""

    restorers = []

    def wrap_method(owner, name: str, scope: str) -> None:
        original = getattr(owner, name)

        def wrapped(*args, __original=original, __scope=scope, **kwargs):
            with record_function(__scope):
                return __original(*args, **kwargs)

        setattr(owner, name, wrapped)
        restorers.append(lambda: setattr(owner, name, original))

    wrap_method(sampler, "_score_fn", "spg::cfg_score_total")
    for predictor in sampler._predictors.values():
        wrap_method(predictor, "update_given_score", "spg::predictor_update")
    for corrector in sampler._correctors.values():
        wrap_method(corrector, "step_given_score", "spg::corrector_update")
    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def run_sample(
    method: str,
    sampling_steps: int,
    *,
    use_profiler: bool,
) -> tuple[object, object]:
    configure_determinism()
    generator = build_generator(method, batch_size=1, sampling_steps=sampling_steps)
    sampler = build_sampler(generator, method, [24000])
    condition = collate([make_condition(24000)])
    with scoped_model(generator.model), scoped_sampler(sampler):
        if use_profiler:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as profiler:
                with record_function("spg::sampling_total"):
                    sampler.sample(condition, None)
            return profiler, generator
        with record_function("spg::sampling_total"):
            sampler.sample(condition, None)
    return None, generator


def event_row(event) -> dict:
    return {
        "key": event.key,
        "count": event.count,
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cpu_time_total_us": float(event.self_cpu_time_total),
        "cuda_time_total_us": float(
            getattr(
                event,
                "cuda_time_total",
                getattr(event, "device_time_total", 0.0),
            )
        ),
        "self_cuda_time_total_us": float(
            getattr(
                event,
                "self_cuda_time_total",
                getattr(event, "self_device_time_total", 0.0),
            )
        ),
        "cpu_memory_usage": int(event.cpu_memory_usage),
        "cuda_memory_usage": int(
            getattr(
                event,
                "cuda_memory_usage",
                getattr(event, "device_memory_usage", 0),
            )
        ),
        "input_shapes": str(event.input_shapes),
    }


def pytorch_profile(method: str) -> dict:
    profiler, _ = run_sample(method, 20, use_profiler=True)
    assert profiler is not None
    trace = TRACE_DIR / f"pytorch_{method.lower()}_20step_trace.json"
    profiler.export_chrome_trace(str(trace))
    events = [event_row(event) for event in profiler.key_averages(group_by_input_shape=False)]
    top_cuda = sorted(events, key=lambda row: row["self_cuda_time_total_us"], reverse=True)
    scopes = {
        row["key"]: row
        for row in events
        if row["key"].startswith("spg::")
    }
    total = scopes.get("spg::sampling_total", {}).get("cuda_time_total_us", 0.0)
    shares = {}
    for key, label in (
        ("spg::periodic_neighbor_construction", "periodic_graph"),
        ("spg::triplet_construction", "triplet"),
        ("spg::gemnet_interaction_block", "gemnet_blocks"),
        ("spg::scatter_segment", "scatter"),
        ("spg::gemnet_forward", "gemnet_total"),
        ("spg::cfg_score_total", "cfg_total"),
        ("spg::predictor_update", "predictor"),
        ("spg::corrector_update", "corrector"),
    ):
        value = scopes.get(key, {}).get("cuda_time_total_us", 0.0)
        shares[label] = value / total if total > 0 else None
    return {
        "created_at": now(),
        "method": method,
        "sampling_steps": 20,
        "trace": str(trace),
        "trace_sha256": sha256_file(trace),
        "trace_bytes": trace.stat().st_size,
        "scope_totals": scopes,
        "shares_of_sampling_cuda_time": shares,
        "top_cuda_operators": top_cuda[:100],
    }


def run_nsys() -> dict:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    nsys_tmp = TRACE_DIR / "tmp"
    nsys_tmp.mkdir(parents=True, exist_ok=True)
    nsys_environment = base_environment(0)
    nsys_environment["TMPDIR"] = str(nsys_tmp)
    prefix = TRACE_DIR / "nsys_c0_10step"
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt,cublas,cudnn",
        "--sample=none",
        "--gpu-metrics-devices=cuda-visible",
        "--gpu-metrics-frequency=1000",
        "--force-overwrite=true",
        f"--output={prefix}",
        str(PYTHON),
        "-m",
        "research.spg_fastgate.profiler_probe",
        "--nsys-target",
    ]
    run_log = TRACE_DIR / "nsys_profile.log"
    with run_log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=nsys_environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    gpu_metrics_captured = result.returncode == 0
    if result.returncode != 0:
        fallback_command = [
            value for value in command if not value.startswith("--gpu-metrics-")
        ]
        with run_log.open("a", encoding="utf-8") as stream:
            stream.write(
                "\nGPU metrics capture failed; retrying without hardware metrics.\n"
            )
            result = subprocess.run(
                fallback_command,
                cwd=PROJECT,
                env=nsys_environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    report = prefix.with_suffix(".nsys-rep")
    if result.returncode != 0 or not report.is_file():
        raise RuntimeError(f"Nsight Systems capture failed; see {run_log}")
    stats_path = TRACE_DIR / "nsys_stats.csv"
    stats = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum,osrt_sum",
            "--format",
            "csv",
            str(report),
        ],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    stats_path.write_text(stats.stdout, encoding="utf-8")
    trace_tables = {}
    for report_name in ("cuda_gpu_trace", "cuda_kern_exec_trace"):
        table = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                report_name,
                "--format",
                "csv",
                str(report),
            ],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        )
        table_path = TRACE_DIR / f"nsys_{report_name}.csv"
        table_path.write_text(table.stdout, encoding="utf-8")
        trace_tables[report_name] = {
            "path": str(table_path),
            "sha256": sha256_file(table_path),
        }
    return {
        "report": str(report),
        "report_sha256": sha256_file(report),
        "report_bytes": report.stat().st_size,
        "stats": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "trace_tables": trace_tables,
        "gpu_metrics_requested": True,
        "gpu_metrics_captured": gpu_metrics_captured,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.nsys_target:
        run_sample("C0", 10, use_profiler=False)
        return 0
    set_stage(
        "pytorch_profiler",
        "running",
        "Profiling representative C0 GemNet forwards with explicit scopes.",
    )
    pytorch = {
        method: pytorch_profile(method)
        for method in ("C0", "A0")
    }
    set_stage(
        "pytorch_profiler",
        "success",
        "PyTorch Profiler operator and scoped breakdown completed.",
        pytorch,
    )
    set_stage(
        "nsight_profile",
        "running",
        "Capturing representative C0 Nsight Systems timeline.",
    )
    nsys = run_nsys()
    set_stage(
        "nsight_profile",
        "success",
        "Nsight Systems trace and summary completed.",
        nsys,
    )
    result = {"created_at": now(), "pytorch": pytorch, "nsight": nsys}
    atomic_json(RESULT, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
