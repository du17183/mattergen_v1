"""Paired global stage-schedule sampler for bounded CFG calibration and validation.

Every policy uses the same initial state and random stream.  Bounded and
permanent policies share the exact CFG=2 prefix state at their registered start;
bounded policies return to CFG=2 after their frozen pulse duration.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from ase import Atoms

from mattergen.diffusion.sampling.risk_calibrated_cfg import (
    FullOutcomeCounterfactualSampler,
    _tensor_digest,
)


def schedule_id(spec: Mapping[str, Any]) -> str:
    kind = str(spec["kind"])
    if kind == "c0":
        return "C0"
    if kind == "constant":
        return f"C_G{int(round(float(spec['cfg']) * 100)):03d}"
    sign = "P" if float(spec["delta_g"]) > 0 else "M"
    magnitude = int(round(abs(float(spec["delta_g"])) * 100))
    if kind == "permanent":
        return f"P_T{int(spec['start']):03d}_D{sign}{magnitude:02d}"
    if kind == "bounded":
        return (
            f"S_T{int(spec['start']):03d}_D{sign}{magnitude:02d}_"
            f"K{int(spec['duration']):03d}"
        )
    raise ValueError(f"unknown schedule kind: {kind}")


def normalized_specs(specs: Sequence[Mapping[str, Any]], n_steps: int = 1000) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in specs:
        spec = dict(raw)
        kind = str(spec["kind"])
        if kind == "c0":
            spec.update(cfg=2.0, start=None, duration=None, delta_g=0.0)
        elif kind == "constant":
            cfg = float(spec["cfg"])
            if cfg == 2.0 or not 1.5 <= cfg <= 2.5:
                raise ValueError(f"invalid constant CFG: {cfg}")
            spec.update(start=None, duration=None, delta_g=cfg - 2.0)
        elif kind in ("permanent", "bounded"):
            start = int(spec["start"])
            delta = float(spec["delta_g"])
            if start < 0 or start >= n_steps or delta == 0 or not 1.5 <= 2.0 + delta <= 2.5:
                raise ValueError(f"invalid stage schedule: {spec}")
            spec["cfg"] = 2.0 + delta
            if kind == "permanent":
                spec["duration"] = n_steps - start
            else:
                duration = int(spec["duration"])
                if duration <= 0 or start + duration > n_steps:
                    raise ValueError(f"invalid bounded duration: {spec}")
        else:
            raise ValueError(f"unknown schedule kind: {kind}")
        identifier = schedule_id(spec)
        if identifier in ids:
            raise ValueError(f"duplicate schedule: {identifier}")
        spec["schedule_id"] = identifier
        ids.add(identifier)
        result.append(spec)
    if "C0" not in ids:
        raise ValueError("schedule set must contain C0")
    return tuple(result)


class StageScheduleSampler(FullOutcomeCounterfactualSampler):
    """Generate complete paired final outcomes for a frozen global schedule set."""

    def __init__(
        self,
        *,
        sample_seed: int,
        schedule_specs: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        self._schedule_specs = normalized_specs(schedule_specs)
        starts = tuple(sorted({int(spec["start"]) for spec in self._schedule_specs if spec["kind"] in ("bounded", "permanent")}))
        if not starts:
            starts = (100,)
        super().__init__(
            sample_seed=sample_seed,
            decision_indices=starts,
            candidate_scales=(1.75, 2.0, 2.25),
            base_scale=2.0,
            temporal_ema_beta=0.9,
            recent_window=5,
            **kwargs,
        )
        self.schedule_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.schedule_rows: list[dict[str, Any]] = []

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.schedule_structures = []
        self.schedule_rows = []

    def _record(self, spec: Mapping[str, Any], atoms: Atoms, prefix_digest: str) -> None:
        metadata = {
            "seed": self._risk_seed,
            "schedule_id": str(spec["schedule_id"]),
            "schedule_kind": str(spec["kind"]),
            "base_cfg": 2.0,
            "active_cfg": float(spec["cfg"]),
            "delta_g": float(spec["delta_g"]),
            "start": -1 if spec["start"] is None else int(spec["start"]),
            "duration": self.N if spec["duration"] is None else int(spec["duration"]),
            "returns_to_cfg2": bool(spec["kind"] == "bounded"),
            "prefix_state_sha256": prefix_digest,
        }
        self.schedule_structures.append((metadata, atoms))
        self.schedule_rows.append(dict(metadata))

    @torch.no_grad()
    def _denoise(
        self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False
    ) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("stage schedule sampler writes explicit policy outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        initial_rng = self._rng_snapshot()
        initial_digest = _tensor_digest(batch)

        for spec in self._schedule_specs:
            if spec["kind"] != "constant":
                continue
            self._restore_rng(initial_rng)
            _, mean = self._advance(
                batch.clone(), mask=mask, timesteps=timesteps, dt=dt,
                start=0, stop=self.N, scale=float(spec["cfg"]),
            )
            self._record(spec, self._final_atoms(mean), initial_digest)

        self._restore_rng(initial_rng)
        base = batch.clone()
        base_mean = batch.clone()
        by_start: dict[int, list[Mapping[str, Any]]] = {}
        for spec in self._schedule_specs:
            if spec["kind"] in ("bounded", "permanent"):
                by_start.setdefault(int(spec["start"]), []).append(spec)

        for index in range(self.N):
            if index in by_start:
                prefix_rng = self._rng_snapshot()
                prefix_digest = _tensor_digest(base)
                for spec in by_start[index]:
                    self._restore_rng(prefix_rng)
                    branch = base.clone()
                    pulse_stop = self.N if spec["kind"] == "permanent" else index + int(spec["duration"])
                    branch, branch_mean = self._advance(
                        branch, mask=mask, timesteps=timesteps, dt=dt,
                        start=index, stop=pulse_stop, scale=float(spec["cfg"]),
                    )
                    if pulse_stop < self.N:
                        branch, branch_mean = self._advance(
                            branch, mask=mask, timesteps=timesteps, dt=dt,
                            start=pulse_stop, stop=self.N, scale=2.0,
                        )
                    self._record(spec, self._final_atoms(branch_mean), prefix_digest)
                self._restore_rng(prefix_rng)
            base, base_mean = self._advance(
                base, mask=mask, timesteps=timesteps, dt=dt,
                start=index, stop=index + 1, scale=2.0,
            )

        c0 = next(spec for spec in self._schedule_specs if spec["kind"] == "c0")
        self._record(c0, self._final_atoms(base_mean), initial_digest)
        if len(self.schedule_structures) != len(self._schedule_specs):
            raise RuntimeError("not every frozen schedule produced an outcome")
        return base_mean, base_mean, None
