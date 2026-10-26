"""Frozen deployment-mode sampler for Linear-K2 confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from ase import Atoms

from mattergen.diffusion.sampling.field_decoupled_cfg import (
    FIELDS,
    FieldDecoupledPolicySampler,
    _tensor_digest,
)
from experiments.reference_preserved_budgeted_cfg.phase_b_sampler import PhaseBFeatureSampler


POLICIES = ("GPulse", "APulse", "PPulse", "CPulse")
METHODS = ("Fixed_K2", "Random_K2", "Linear_K2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenLinearK2ConfirmSampler(PhaseBFeatureSampler):
    """Run exact C0 and only two explicitly allocated branches per method."""

    def __init__(
        self,
        *,
        sample_seed: int,
        policy_specs: Sequence[Mapping[str, Any]],
        artifact_path: str,
        parameters_path: str,
        expected_artifact_sha256: str,
        random_policies: Sequence[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_seed=sample_seed, policy_specs=policy_specs, **kwargs)
        artifact = Path(artifact_path)
        if sha256(artifact) != expected_artifact_sha256:
            raise RuntimeError("frozen allocator artifact hash mismatch")
        self._allocator = joblib.load(artifact)
        self._allocator_parameters = json.loads(Path(parameters_path).read_text())
        self._random_policies = tuple(map(str, random_policies))
        if len(self._random_policies) != 2 or not set(self._random_policies).issubset(POLICIES):
            raise ValueError("invalid frozen Random-K2 allocation")
        self.method_allocations: dict[str, tuple[str, str]] = {}
        self.allocator_predictions: list[dict[str, Any]] = []
        self.deployment_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.deployment_rows: list[dict[str, Any]] = []
        self.standalone_c0_atoms: Atoms | None = None
        self.reference_comparison: dict[str, Any] | None = None
        self._capture_prebranch_features = True

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.method_allocations = {}
        self.allocator_predictions = []
        self.deployment_structures = []
        self.deployment_rows = []
        self.standalone_c0_atoms = None
        self.reference_comparison = None
        self._capture_prebranch_features = True

    def _score_at_field_scales(
        self,
        x: Any,
        t: torch.Tensor,
        scales: Mapping[str, float],
    ) -> Any:
        """Keep the standalone replay out of the frozen prefix feature trace."""
        if self._capture_prebranch_features:
            return super()._score_at_field_scales(x, t, scales)
        return FieldDecoupledPolicySampler._score_at_field_scales(self, x, t, scales)

    def _linear_allocation(self) -> tuple[str, str]:
        feature_columns = list(self._allocator_parameters["feature_columns"])
        model_columns = list(self._allocator_parameters["model_columns"])
        rows = []
        for policy in POLICIES:
            row = {name: float(self.branch_features[name]) for name in feature_columns}
            row["policy_id"] = policy
            for candidate in POLICIES:
                indicator = 1.0 if policy == candidate else 0.0
                for name in feature_columns:
                    row[f"ix_{candidate}_{name}"] = indicator * float(row[name])
            rows.append(row)
        frame = pd.DataFrame(rows)
        classes = list(map(int, self._allocator.named_steps["estimator"].classes_))
        scores = self._allocator.predict_proba(frame[model_columns])[:, classes.index(1)]
        scored = sorted(zip(POLICIES, map(float, scores)), key=lambda item: (-item[1], item[0]))
        selected = tuple(policy for policy, _ in scored[:2])
        for rank, (policy, score) in enumerate(scored, start=1):
            self.allocator_predictions.append({
                "seed": self._risk_seed,
                "policy_id": policy,
                "safe_beneficial_probability": score,
                "rank": rank,
                "selected_top2": policy in selected,
            })
        return selected  # type: ignore[return-value]

    def _record_deployment(self, *, method: str, rank: int, policy: Mapping[str, Any], atoms: Atoms, prefix_digest: str) -> None:
        metadata = {
            "seed": self._risk_seed,
            "method": method,
            "allocation_rank": rank,
            "policy_id": str(policy["policy_id"]),
            "policy_kind": str(policy["kind"]),
            "g_atomic": float(policy["scales"]["atomic_numbers"]),
            "g_pos": float(policy["scales"]["pos"]),
            "g_cell": float(policy["scales"]["cell"]),
            "start": -1 if policy["start"] is None else int(policy["start"]),
            "duration": self.N if policy["duration"] is None else int(policy["duration"]),
            "prefix_state_sha256": prefix_digest,
            "branch_id": f"{method}:{self._risk_seed}:{policy['policy_id']}:{rank}",
        }
        self.deployment_structures.append((dict(metadata), atoms))
        self.deployment_rows.append(dict(metadata))

    @torch.no_grad()
    def _denoise(self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("confirmatory sampler writes explicit deployment outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        base_scales = {field: 2.0 for field in FIELDS}
        initial_rng = self._rng_snapshot()
        initial_batch = batch.clone()

        self._restore_rng(initial_rng)
        self._capture_prebranch_features = False
        try:
            _, standalone_mean = self._advance_field(
                initial_batch.clone(), mask=mask, timesteps=timesteps, dt=dt,
                start=0, stop=self.N, scales=base_scales,
            )
        finally:
            self._capture_prebranch_features = True
        self.standalone_c0_atoms = self._final_atoms(standalone_mean)

        self._restore_rng(initial_rng)
        base = initial_batch.clone()
        base_mean = initial_batch.clone()
        for index in range(self.branch_point):
            base, base_mean = self._advance_field(
                base, mask=mask, timesteps=timesteps, dt=dt,
                start=index, stop=index + 1, scales=base_scales,
            )
        prefix_rng = self._rng_snapshot()
        prefix_digest = _tensor_digest(base)
        self.branch_features = self._summarize_prefix(base, prefix_digest)
        self.prefix_checkpoint = {
            "schema_version": 1,
            "seed": self._risk_seed,
            "branch_point": self.branch_point,
            "prefix_state_sha256": prefix_digest,
            "batch": base.clone(),
            "rng_state": prefix_rng,
        }
        self.method_allocations = {
            "Fixed_K2": ("GPulse", "PPulse"),
            "Random_K2": self._random_policies,
            "Linear_K2": self._linear_allocation(),
        }
        lookup = {str(policy["policy_id"]): policy for policy in self._policy_specs}

        self._restore_rng(prefix_rng)
        c0_batch, c0_mean = self._advance_field(
            base.clone(), mask=mask, timesteps=timesteps, dt=dt,
            start=self.branch_point, stop=self.N, scales=base_scales,
        )
        c0_atoms = self._final_atoms(c0_mean)
        self._record_deployment(method="C0", rank=0, policy=lookup["C0"], atoms=c0_atoms, prefix_digest=prefix_digest)

        for method in METHODS:
            for rank, identifier in enumerate(self.method_allocations[method], start=1):
                policy = lookup[identifier]
                self._restore_rng(prefix_rng)
                branch = base.clone()
                pulse_stop = self.branch_point + int(policy["duration"])
                branch, branch_mean = self._advance_field(
                    branch, mask=mask, timesteps=timesteps, dt=dt,
                    start=self.branch_point, stop=pulse_stop, scales=policy["scales"],
                )
                branch, branch_mean = self._advance_field(
                    branch, mask=mask, timesteps=timesteps, dt=dt,
                    start=pulse_stop, stop=self.N, scales=base_scales,
                )
                self._record_deployment(method=method, rank=rank, policy=policy, atoms=self._final_atoms(branch_mean), prefix_digest=prefix_digest)

        if self.standalone_c0_atoms is None:
            raise RuntimeError("standalone C0 was not produced")
        self.reference_comparison = self._exact_atoms_comparison(self.standalone_c0_atoms, c0_atoms)
        self.reference_comparison["success"] = bool(
            self.reference_comparison["atomic_numbers_identical"]
            and self.reference_comparison["positions_max_abs_error"] == 0.0
            and self.reference_comparison["cell_max_abs_error"] == 0.0
        )
        if not self.reference_comparison["success"]:
            raise RuntimeError(f"REFERENCE_REPRODUCTION_FAIL: {self.reference_comparison}")
        if len(self.deployment_structures) != 7:
            raise RuntimeError("deployment outcome count must be exactly seven")
        return c0_batch, c0_mean, None
