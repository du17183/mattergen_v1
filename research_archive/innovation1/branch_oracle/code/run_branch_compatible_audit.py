"""Phase A: zero-generation Branch-Compatible Oracle Audit."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymatgen.entries.computed_entries import ComputedStructureEntry

from run_offline_analysis import (
    ROOT,
    SOURCE,
    add_headroom,
    c0_metrics,
    json_write,
    load_quality,
    metrics,
    quality_gate,
    select_terminal,
    sha256,
)


OUT = ROOT / "offline_branchable"
SOURCE_CONFIG = SOURCE / "policies.yaml"
SOURCE_IMPLEMENTATION = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg/mattergen/diffusion/sampling/field_decoupled_cfg.py"
)
SOURCE_GENERATION_MANIFEST = SOURCE / "generation_manifest.csv"
MATRIX_PATH = ROOT / "offline/candidate_matrix.csv"
SPLIT_PATH = ROOT / "offline/split_manifest.json"
PROTOCOL_PATH = ROOT / "branch_compatible_protocol.yaml"
EXPECTED_BRANCH_POINT = 400
TOTAL_STEPS = 1000
BRANCHABLE_ADAPTIVE = ("GPulse", "APulse", "PPulse", "CPulse")
RANDOM_RESAMPLES = 20_000
RANDOM_SEED_BASE = 2026091900


def implementation_audit() -> dict:
    source = SOURCE_IMPLEMENTATION.read_text()
    required_fragments = (
        'if policy["kind"] == "pulse"',
        "by_start.setdefault(int(policy[\"start\"]), []).append(policy)",
        "prefix_rng = self._rng_snapshot()",
        "prefix_digest = _tensor_digest(base)",
        "self._restore_rng(prefix_rng)",
        "branch = base.clone()",
        "pulse_stop = index + int(policy[\"duration\"])",
        "start=index, stop=pulse_stop, scales=policy[\"scales\"]",
        "start=pulse_stop, stop=self.N, scales=base_scales",
        "base, base_mean = self._advance_field",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    return {
        "implementation_path": str(SOURCE_IMPLEMENTATION),
        "implementation_sha256": sha256(SOURCE_IMPLEMENTATION),
        "required_static_fragments": len(required_fragments),
        "missing_static_fragments": missing,
        "static_shared_prefix_logic_pass": not missing,
    }


def compatibility_manifest() -> tuple[pd.DataFrame, dict]:
    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    implementation = implementation_audit()
    if int(config["pulse_start"]) != EXPECTED_BRANCH_POINT:
        raise RuntimeError("frozen pulse branch point changed")
    if int(config["pulse_duration"]) != 100:
        raise RuntimeError("frozen pulse duration changed")
    rows = []
    for policy in config["policies"]:
        identifier = str(policy["policy_id"])
        kind = str(policy["kind"])
        scales = {field: float(policy["scales"][field]) for field in ("atomic", "pos", "cell")}
        if identifier == "C0":
            compatible = kind == "constant" and set(scales.values()) == {2.0}
            branch_point = EXPECTED_BRANCH_POINT
            reason = "exact CFG2 reference; unchanged before and after the common branch point"
            post = "constant CFG2 reference continuation"
        elif kind == "pulse":
            compatible = (
                int(policy["start"]) == EXPECTED_BRANCH_POINT
                and implementation["static_shared_prefix_logic_pass"]
            )
            branch_point = int(policy["start"])
            reason = (
                "sampler advances the shared CFG2 base to step 400, snapshots full state and RNG, then clones the branch"
                if compatible
                else "pulse configuration or static shared-prefix implementation audit failed"
            )
            post = (
                f"steps {policy['start']}–{int(policy['start']) + int(policy['duration']) - 1}: "
                f"atomic={scales['atomic']}, pos={scales['pos']}, cell={scales['cell']}; "
                f"steps {int(policy['start']) + int(policy['duration'])}–999: CFG2 continuation"
            )
        else:
            compatible = False
            branch_point = 0
            changed = [field for field, value in scales.items() if value != 2.0]
            reason = f"constant policy changes {','.join(changed)} from sampling step 0"
            post = f"full trajectory constant atomic={scales['atomic']}, pos={scales['pos']}, cell={scales['cell']}"
        rows.append(
            {
                "policy_id": identifier,
                "is_branch_compatible": bool(compatible),
                "branch_point": int(branch_point),
                "reason": reason,
                "prebranch_cfg_atomic": 2.0 if compatible else scales["atomic"],
                "prebranch_cfg_pos": 2.0 if compatible else scales["pos"],
                "prebranch_cfg_cell": 2.0 if compatible else scales["cell"],
                "postbranch_behavior": post,
                "policy_kind": kind,
                "configured_g_atomic": scales["atomic"],
                "configured_g_pos": scales["pos"],
                "configured_g_cell": scales["cell"],
            }
        )
    frame = pd.DataFrame(rows)
    actual = tuple(frame.loc[frame.is_branch_compatible & (frame.policy_id != "C0"), "policy_id"])
    if set(actual) != set(BRANCHABLE_ADAPTIVE):
        raise RuntimeError(f"unexpected branch-compatible bank: {actual}")
    return frame, implementation


def prefix_digest_audit() -> dict:
    manifest = pd.read_csv(SOURCE_GENERATION_MANIFEST)
    pulse = manifest.loc[manifest.policy_id.isin(BRANCHABLE_ADAPTIVE)]
    per_seed = pulse.groupby("seed").prefix_state_sha256.nunique()
    return {
        "seed_count": int(pulse.seed.nunique()),
        "pulse_policy_count": int(pulse.policy_id.nunique()),
        "every_seed_has_four_pulses": bool((pulse.groupby("seed").size() == 4).all()),
        "one_common_prefix_digest_per_seed": bool((per_seed == 1).all()),
        "common_prefix_digest_pass": bool((per_seed == 1).all() and (pulse.groupby("seed").size() == 4).all()),
        "note": "The step-400 state tensor itself was not serialized; equality is supported by the common digest plus static clone/RNG implementation audit.",
    }


def subset_cost(k: int) -> float:
    b = EXPECTED_BRANCH_POINT / TOTAL_STEPS
    return b + (k + 1) * (1.0 - b)


def compute_budget() -> pd.DataFrame:
    rows = []
    for k in (0, 1, 2, 3, 4):
        rows.append(
            {
                "k_adaptive_branches": k,
                "branch_point": EXPECTED_BRANCH_POINT,
                "total_steps": TOTAL_STEPS,
                "shared_prefix_fraction": EXPECTED_BRANCH_POINT / TOTAL_STEPS,
                "suffix_fraction": 1 - EXPECTED_BRANCH_POINT / TOTAL_STEPS,
                "compute_multiplier": subset_cost(k),
                "formula": "b + (K+1)*(1-b)",
            }
        )
    return pd.DataFrame(rows)


def oracle_analysis(matrix: pd.DataFrame, entries: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    output = []
    selected_output = []
    cache = {}
    for selector, safe_column in (("SAFE-A", "safe_a"), ("SAFE-B", "safe_b")):
        for split_name, frame in [("all", matrix), *list(matrix.groupby("offline_split", sort=False))]:
            selected = select_terminal(frame, BRANCHABLE_ADAPTIVE, safe_column)
            result, enriched = metrics(
                selected,
                entries,
                candidate_count=len(BRANCHABLE_ADAPTIVE),
                compute_multiplier=subset_cost(len(BRANCHABLE_ADAPTIVE)),
            )
            c0, _ = c0_metrics(frame, entries)
            result = add_headroom(result, c0, result)
            frequencies = enriched.policy_id.value_counts().sort_index().to_dict()
            result.update(
                selector=selector,
                split=split_name,
                method="Branchable-Oracle-All",
                selected_policy_frequencies=json.dumps(frequencies, sort_keys=True),
            )
            output.append(result)
            cache[(selector, split_name)] = result
            enriched.insert(0, "selector", selector)
            enriched.insert(1, "split", split_name)
            selected_output.append(enriched)
    return pd.DataFrame(output), pd.concat(selected_output, ignore_index=True), cache


def oracle_k_analysis(matrix: pd.DataFrame, entries: dict, oracle_cache: dict) -> pd.DataFrame:
    rows = []
    for selector, safe_column in (("SAFE-A", "safe_a"), ("SAFE-B", "safe_b")):
        for split_name, frame in matrix.groupby("offline_split", sort=False):
            selected = select_terminal(frame, BRANCHABLE_ADAPTIVE, safe_column)
            c0, _ = c0_metrics(frame, entries)
            for k in (1, 2, 3):
                result, _ = metrics(selected, entries, candidate_count=k, compute_multiplier=subset_cost(k))
                result = add_headroom(result, c0, oracle_cache[(selector, split_name)])
                result.update(
                    selector=selector,
                    split=split_name,
                    method=f"Branchable-Oracle-K{k}",
                    k=k,
                    hindsight_policy_identity_required=True,
                    interpretation="per-seed hindsight chooses a set containing the best safe branch-compatible policy",
                )
                rows.append(result)
    return pd.DataFrame(rows)


def fixed_k_analysis(matrix: pd.DataFrame, entries: dict, oracle_cache: dict) -> pd.DataFrame:
    rows = []
    train = matrix.loc[matrix.offline_split == "offline_train"]
    split_frames = {
        "offline_train": train,
        "offline_validation": matrix.loc[matrix.offline_split == "offline_validation"],
        "offline_test": matrix.loc[matrix.offline_split == "offline_test"],
    }
    for selector, safe_column in (("SAFE-A", "safe_a"), ("SAFE-B", "safe_b")):
        for k in (1, 2, 3):
            candidates = []
            for subset in itertools.combinations(BRANCHABLE_ADAPTIVE, k):
                selected = select_terminal(train, subset, safe_column)
                candidates.append((float(selected.property_error.mean()), tuple(sorted(subset))))
            _, best = min(candidates, key=lambda item: (item[0], item[1]))
            retained = False
            current = []
            for split_name, frame in split_frames.items():
                selected = select_terminal(frame, best, safe_column)
                result, _ = metrics(selected, entries, candidate_count=k, compute_multiplier=subset_cost(k))
                c0, _ = c0_metrics(frame, entries)
                result = add_headroom(result, c0, oracle_cache[(selector, split_name)])
                if split_name == "offline_validation":
                    retained = quality_gate(result, c0)
                result.update(
                    selector=selector,
                    split=split_name,
                    method=f"Fixed-Branchable-K{k}",
                    k=k,
                    policies=";".join(best),
                    subset_discovery="offline_train_exhaustive",
                )
                current.append(result)
            for result in current:
                result["validation_retained"] = retained
                rows.append(result)
    return pd.DataFrame(rows)


def precompute_cross_seed_matches(test: pd.DataFrame, entries: dict) -> tuple[dict, int]:
    keys = [(int(row.seed), str(row.policy_id)) for row in test.itertuples()]
    parsed = {key: ComputedStructureEntry.from_dict(entries[key]).structure for key in keys}
    matcher = __import__(
        "mattergen.evaluation.utils.structure_matcher", fromlist=["DefaultDisorderedStructureMatcher"]
    ).DefaultDisorderedStructureMatcher()
    matches = {}
    positive = 0
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            if left[0] == right[0]:
                continue
            if parsed[left].composition.chemical_system != parsed[right].composition.chemical_system:
                value = False
            else:
                value = bool(matcher.fit(parsed[left], parsed[right]))
            matches[(left, right)] = value
            matches[(right, left)] = value
            positive += int(value)
    return matches, positive


def selected_unique(keys: list[tuple[int, str]], matches: dict) -> np.ndarray:
    unique_keys = []
    flags = []
    for key in keys:
        is_unique = not any(matches.get((key, previous), False) for previous in unique_keys)
        flags.append(is_unique)
        if is_unique:
            unique_keys.append(key)
    return np.asarray(flags, dtype=bool)


def random_analysis(matrix: pd.DataFrame, entries: dict, oracle_cache: dict) -> tuple[pd.DataFrame, dict]:
    test = matrix.loc[matrix.offline_split == "offline_test"].copy()
    seeds = sorted(map(int, test.seed.unique()))
    rows_by_key = {(int(row.seed), str(row.policy_id)): row for row in test.itertuples()}
    matches, positive_cross_seed_matches = precompute_cross_seed_matches(test, entries)
    output = []
    metric_names = (
        "property_mae", "recovered_oracle_headroom", "e_hull_mean", "stable_rate",
        "novel_rate", "unique_rate", "nus_rate", "validity_rate", "fallback_rate",
    )
    for selector_index, (selector, safe_column) in enumerate((("SAFE-A", "safe_a"), ("SAFE-B", "safe_b"))):
        c0 = float(test.loc[test.policy_id == "C0", "property_error"].mean())
        headroom = c0 - float(oracle_cache[(selector, "offline_test")]["property_mae"])
        for k in (1, 2, 3):
            rng = np.random.default_rng(RANDOM_SEED_BASE + selector_index * 10 + k)
            samples = {name: np.empty(RANDOM_RESAMPLES, dtype=float) for name in metric_names}
            for replicate in range(RANDOM_RESAMPLES):
                selected_rows = []
                keys = []
                for seed in seeds:
                    subset = rng.choice(np.asarray(BRANCHABLE_ADAPTIVE, dtype=object), size=k, replace=False)
                    eligible = [rows_by_key[(seed, str(policy))] for policy in subset if bool(getattr(rows_by_key[(seed, str(policy))], safe_column))]
                    row = min(eligible, key=lambda item: (float(item.property_error), str(item.policy_id))) if eligible else rows_by_key[(seed, "C0")]
                    selected_rows.append(row)
                    keys.append((seed, str(row.policy_id)))
                unique = selected_unique(keys, matches)
                prop = np.asarray([float(row.property_error) for row in selected_rows])
                stable = np.asarray([bool(row.stable) for row in selected_rows])
                novel = np.asarray([bool(row.novel) for row in selected_rows])
                samples["property_mae"][replicate] = prop.mean()
                samples["recovered_oracle_headroom"][replicate] = (c0 - prop.mean()) / headroom
                samples["e_hull_mean"][replicate] = np.mean([float(row.E_hull) for row in selected_rows])
                samples["stable_rate"][replicate] = stable.mean()
                samples["novel_rate"][replicate] = novel.mean()
                samples["unique_rate"][replicate] = unique.mean()
                samples["nus_rate"][replicate] = (stable & novel & unique).mean()
                samples["validity_rate"][replicate] = np.mean([bool(row.valid) for row in selected_rows])
                samples["fallback_rate"][replicate] = np.mean([row.policy_id == "C0" for row in selected_rows])
            record = {
                "selector": selector,
                "split": "offline_test",
                "method": f"Random-Branchable-K{k}",
                "k": k,
                "resamples": RANDOM_RESAMPLES,
                "random_seed": RANDOM_SEED_BASE + selector_index * 10 + k,
                "compute_multiplier": subset_cost(k),
            }
            for name, values in samples.items():
                record[f"{name}_mean"] = float(values.mean())
                record[f"{name}_ci95_low"] = float(np.quantile(values, 0.025))
                record[f"{name}_ci95_high"] = float(np.quantile(values, 0.975))
            output.append(record)
    return pd.DataFrame(output), {
        "cross_seed_structure_matches_in_branchable_test_pool": positive_cross_seed_matches,
        "mixed_uniqueness_method": "precomputed exact official matcher pair graph; first selected representative per seed order retained",
    }


def write_reports(
    compatibility: pd.DataFrame,
    implementation: dict,
    digest_audit: dict,
    matrix: pd.DataFrame,
    oracle: pd.DataFrame,
    oracle_k: pd.DataFrame,
    fixed: pd.DataFrame,
    random: pd.DataFrame,
    random_audit: dict,
) -> dict:
    oa = oracle.loc[(oracle.selector == "SAFE-A") & (oracle.split == "offline_test")].iloc[0]
    ob = oracle.loc[(oracle.selector == "SAFE-B") & (oracle.split == "offline_test")].iloc[0]
    ok2 = oracle_k.loc[(oracle_k.selector == "SAFE-A") & (oracle_k.split == "offline_test") & (oracle_k.k == 2)].iloc[0]
    fk2 = fixed.loc[(fixed.selector == "SAFE-A") & (fixed.split == "offline_test") & (fixed.k == 2)].iloc[0]
    rk2 = random.loc[(random.selector == "SAFE-A") & (random.k == 2)].iloc[0]
    a1 = bool(oa.relative_property_improvement_vs_c0 >= 0.10)
    a2 = bool(ok2.recovered_oracle_headroom >= 0.60) or bool(
        oracle_k.loc[(oracle_k.selector == "SAFE-A") & (oracle_k.split == "offline_test") & (oracle_k.k == 3), "recovered_oracle_headroom"].iloc[0] >= 0.75
    )
    adaptive_needed = bool(fk2.recovered_oracle_headroom < 0.90)
    k2_compute = subset_cost(2)
    a4 = bool(k2_compute <= 2.20 + 1e-12)
    phase_a_go = a1 and a2 and a4
    status = {
        "BRANCH_COMPATIBILITY_AUDIT": "COMPLETE" if implementation["static_shared_prefix_logic_pass"] and digest_audit["common_prefix_digest_pass"] else "FAIL",
        "BRANCHABLE_POLICY_BANK": list(BRANCHABLE_ADAPTIVE),
        "BRANCH_POINT": EXPECTED_BRANCH_POINT,
        "BRANCHABLE_ORACLE_HEADROOM": float(oa.oracle_headroom),
        "BRANCHABLE_ORACLE_IMPROVEMENT": float(oa.relative_property_improvement_vs_c0),
        "BRANCHABLE_ORACLE_K2_RECOVERY": float(ok2.recovered_oracle_headroom),
        "BEST_FIXED_BRANCHABLE_K2": fk2.policies,
        "BEST_FIXED_BRANCHABLE_K2_RECOVERY": float(fk2.recovered_oracle_headroom),
        "RANDOM_BRANCHABLE_K2_RECOVERY": float(rk2.recovered_oracle_headroom_mean),
        "RANDOM_BRANCHABLE_K2_RECOVERY_CI95": [float(rk2.recovered_oracle_headroom_ci95_low), float(rk2.recovered_oracle_headroom_ci95_high)],
        "K2_SHARED_PREFIX_COMPUTE": k2_compute,
        "COMPUTE_BUDGET": "PASS" if a4 else ("BORDERLINE" if k2_compute <= 2.5 else "FAIL"),
        "BRANCHABLE_BUDGETED_HEADROOM": "GO" if phase_a_go else "FAIL",
        "ADAPTIVE_SELECTION_NEEDED": "YES" if adaptive_needed else "NO",
        "PHASE_A_GATES": {"A1_headroom": a1, "A2_small_k": a2, "A3_adaptive_needed": adaptive_needed, "A4_compute": a4},
        "NEXT": "START_PHASE_B" if phase_a_go and adaptive_needed else "STOP",
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "ZERO_NEW_GENERATION": True,
        "ZERO_NEW_GPU_SAMPLING": True,
        "DFT_VERIFIED": False,
    }
    json_write(OUT / "phase_a_status.json", status)
    root_status_path = ROOT / "pipeline_status.json"
    root_status = json.loads(root_status_path.read_text())
    root_status.update(status)
    root_status["ADAPTIVE_ALLOCATION_VALUE"] = "NOT_EVALUABLE"
    root_status["BEST_ADAPTIVE_K2"] = "NOT_EVALUABLE_NO_PREBRANCH_FEATURES"
    json_write(root_status_path, root_status)

    branch_report = f"""# Branch-point and compatibility audit

The frozen policy configuration sets `pulse_start=400`, `pulse_duration=100`, and `N=1000`. Sampling indices run from 0 at the high-noise `max_t` end to 999 at `eps_t`; the sampler uses `torch.linspace(max_t, eps_t, N)`. Thus branch point 400 is after 400 reverse predictor/corrector updates (40% of the update sequence), leaving a 60% suffix.

For pulse policies, the implementation first advances the shared base with field scales `(2.0, 2.0, 2.0)`. At index 400 it snapshots the RNG and hashes the base tensor, restores that RNG for every branch, clones the complete base state, applies the policy for steps 400–499, and resumes CFG2 for steps 500–999. The C0 base is advanced from the same pre-branch state and RNG snapshot.

- Static implementation audit: `{'PASS' if implementation['static_shared_prefix_logic_pass'] else 'FAIL'}`
- Pulse prefix digest equality: `{'PASS' if digest_audit['common_prefix_digest_pass'] else 'FAIL'}` across {digest_audit['seed_count']} seeds and {digest_audit['pulse_policy_count']} policies.
- State tensors at every prefix step were not serialized. The strongest available audit is the frozen schedule/config, code path, and common step-400 tensor digest.
- Constant adaptive policies start changing at step 0 and are not branch-compatible with a step-400 C0 prefix.

Config SHA-256: `{sha256(SOURCE_CONFIG)}`
Implementation SHA-256: `{implementation['implementation_sha256']}`
"""
    (OUT / "branch_point_audit.md").write_text(branch_report)

    c0 = float(matrix.loc[(matrix.offline_split == "offline_test") & (matrix.policy_id == "C0"), "property_error"].mean())
    report = f"""# Phase A — Branch-Compatible Oracle Audit

This audit generated no new structures and launched no GPU sampling. It retained the existing 24/12/12 split and restricted candidates to `GPulse`, `APulse`, `PPulse`, and `CPulse` plus exact C0.

## Offline-test oracle

| Selector | C0 MAE | Branchable Oracle-All MAE | Relative improvement | Stable | Novel | Unique | NUS | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAFE-A | {c0:.8f} | {oa.property_mae:.8f} | {oa.relative_property_improvement_vs_c0:.2%} | {oa.stable_rate:.2%} | {oa.novel_rate:.2%} | {oa.unique_rate:.2%} | {oa.nus_rate:.2%} | {oa.fallback_rate:.2%} |
| SAFE-B | {c0:.8f} | {ob.property_mae:.8f} | {ob.relative_property_improvement_vs_c0:.2%} | {ob.stable_rate:.2%} | {ob.novel_rate:.2%} | {ob.unique_rate:.2%} | {ob.nus_rate:.2%} | {ob.fallback_rate:.2%} |

## Budget comparison

| Method | Policies / definition | Property MAE | Recovered headroom | Compute ×C0 |
|---|---|---:|---:|---:|
| Oracle-K2 | per-seed hindsight | {ok2.property_mae:.8f} | {ok2.recovered_oracle_headroom:.2%} | {ok2.estimated_compute_multiplier:.2f}× |
| Fixed-K2 | `{fk2.policies}` selected on train | {fk2.property_mae:.8f} | {fk2.recovered_oracle_headroom:.2%} | {fk2.estimated_compute_multiplier:.2f}× |
| Random-K2 | 20k deterministic allocations | {rk2.property_mae_mean:.8f} mean | {rk2.recovered_oracle_headroom_mean:.2%} mean | {rk2.compute_multiplier:.2f}× |

The Oracle-K result is an existence upper bound because the registered definition gives hindsight access to the best policy identity for each seed. Random-K confidence intervals are randomization intervals over candidate allocations on the fixed 12-seed test cohort. Mixed Unique/NUS were recomputed; cross-seed matcher edges in the complete test pool: {random_audit['cross_seed_structure_matches_in_branchable_test_pool']}.

```text
BRANCHABLE_ORACLE_IMPROVEMENT = {status['BRANCHABLE_ORACLE_IMPROVEMENT']:.6f}
BRANCHABLE_ORACLE_K2_RECOVERY = {status['BRANCHABLE_ORACLE_K2_RECOVERY']:.6f}
BEST_FIXED_BRANCHABLE_K2 = {status['BEST_FIXED_BRANCHABLE_K2']}
BEST_FIXED_BRANCHABLE_K2_RECOVERY = {status['BEST_FIXED_BRANCHABLE_K2_RECOVERY']:.6f}
RANDOM_BRANCHABLE_K2_RECOVERY = {status['RANDOM_BRANCHABLE_K2_RECOVERY']:.6f}
K2_SHARED_PREFIX_COMPUTE = {status['K2_SHARED_PREFIX_COMPUTE']:.2f}
BRANCHABLE_BUDGETED_HEADROOM = {status['BRANCHABLE_BUDGETED_HEADROOM']}
ADAPTIVE_SELECTION_NEEDED = {status['ADAPTIVE_SELECTION_NEEDED']}
```

Phase A next action: `{status['NEXT']}`. All physical-quality and property values remain surrogate evaluations; `DFT_VERIFIED=False`.
"""
    (OUT / "branchable_oracle_all_report.md").write_text(report)
    return status


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(MATRIX_PATH)
    original_split = json.loads(SPLIT_PATH.read_text())
    compatibility, implementation = compatibility_manifest()
    digest_audit = prefix_digest_audit()
    compatibility.to_csv(OUT / "branch_compatibility_manifest.csv", index=False)
    branchable = matrix.loc[matrix.policy_id.isin(("C0", *BRANCHABLE_ADAPTIVE))].copy()
    if len(branchable) != 240 or branchable.seed.nunique() != 48:
        raise RuntimeError("branchable candidate matrix shape mismatch")
    if {
        name: sorted(map(int, branchable.loc[branchable.offline_split == name, "seed"].unique()))
        for name in ("offline_train", "offline_validation", "offline_test")
    } != {
        name: sorted(map(int, original_split[name]))
        for name in ("offline_train", "offline_validation", "offline_test")
    }:
        raise RuntimeError("offline split changed")
    branchable.to_csv(OUT / "candidate_matrix_branchable.csv", index=False)
    _, entries = load_quality()
    oracle, selected, oracle_cache = oracle_analysis(branchable, entries)
    selected.to_csv(OUT / "branchable_oracle_all_selected.csv", index=False)
    oracle.to_csv(OUT / "branchable_oracle_all_metrics.csv", index=False)
    oracle_k = oracle_k_analysis(branchable, entries, oracle_cache)
    oracle_k.to_csv(OUT / "oracle_branchable_k_results.csv", index=False)
    fixed = fixed_k_analysis(branchable, entries, oracle_cache)
    fixed.to_csv(OUT / "fixed_branchable_k_results.csv", index=False)
    random, random_audit = random_analysis(branchable, entries, oracle_cache)
    random.to_csv(OUT / "random_branchable_k_results.csv", index=False)
    json_write(OUT / "random_branchable_audit.json", random_audit)
    compute_budget().to_csv(OUT / "branchable_compute_budget.csv", index=False)
    json_write(
        OUT / "phase_a_source_manifest.json",
        {
            "branch_compatible_protocol.yaml": sha256(PROTOCOL_PATH),
            "candidate_matrix.csv": sha256(MATRIX_PATH),
            "split_manifest.json": sha256(SPLIT_PATH),
            "policies.yaml": sha256(SOURCE_CONFIG),
            "generation_manifest.csv": sha256(SOURCE_GENERATION_MANIFEST),
            "field_decoupled_cfg.py": sha256(SOURCE_IMPLEMENTATION),
        },
    )
    status = write_reports(
        compatibility, implementation, digest_audit, branchable, oracle, oracle_k, fixed, random, random_audit
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
