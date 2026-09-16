"""Seal every outcome-sensitive field-CFG protocol and implementation file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT=Path(__file__).resolve().parents[2]; ROOT=PROJECT/"experiments/field_decoupled_adaptive_cfg"
FILES=(
"mattergen/diffusion/sampling/risk_calibrated_cfg.py","mattergen/diffusion/sampling/field_decoupled_cfg.py",
"experiments/field_decoupled_adaptive_cfg/preregistered_protocol.yaml","experiments/field_decoupled_adaptive_cfg/calibration/policies.yaml","experiments/field_decoupled_adaptive_cfg/calibration/seeds.json","experiments/field_decoupled_adaptive_cfg/seed_registry.json","experiments/field_decoupled_adaptive_cfg/seed_overlap_audit.json",
"experiments/field_decoupled_adaptive_cfg/stage0/field_guidance_norm.csv","experiments/field_decoupled_adaptive_cfg/stage0/field_guidance_by_timestep.csv","experiments/field_decoupled_adaptive_cfg/stage0/field_attribution_report.md","experiments/field_decoupled_adaptive_cfg/implementation/implementation_check.json",
"experiments/field_decoupled_adaptive_cfg/run_stage0.py","experiments/field_decoupled_adaptive_cfg/register_seeds.py","experiments/field_decoupled_adaptive_cfg/run_generation.py","experiments/field_decoupled_adaptive_cfg/prepare_phase.py","experiments/field_decoupled_adaptive_cfg/run_quality.py","experiments/field_decoupled_adaptive_cfg/analyze_phase.py","experiments/field_decoupled_adaptive_cfg/run_pipeline.py","experiments/field_decoupled_adaptive_cfg/initial_audit.md")


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def main()->None:
    destination=ROOT/"frozen_manifest.json"
    if destination.exists(): raise FileExistsError(destination)
    hashes={}
    for relative in FILES:
        path=PROJECT/relative
        if not path.is_file(): raise FileNotFoundError(path)
        hashes[relative]=sha256(path)
    manifest={"schema_version":1,"frozen_before_generation":True,"git_base_commit":"f281462d8f690493f3bc3b78faffaea6abdeb2ae","branch":"experiment/field-decoupled-adaptive-cfg","sha256":hashes,"calibration_seeds":48,"p0_seeds_reserved":32,"formal256_seeds_reserved":256,"policy_count":12,"bootstrap_resamples":20000,"future_cohorts_reserved_before_calibration_outcomes":True,"implementation_exact_replay_pass":True,"allowed_after_freeze":"execution and technical bug fixes that do not change scientific decisions","forbidden_after_freeze":"new CFG amplitude field combination timestep duration threshold metric seed controller or rescue tuning","surrogate_property_eval":True,"dft_verified":False}; destination.write_text(json.dumps(manifest,indent=2)+"\n")
    status={"FIELD_ATTRIBUTION_DIAGNOSTIC":"COMPLETE","FIELD_DECOUPLING_CALIBRATION":"RUNNING","PROPERTY_FIELD":"NONE","QUALITY_HARM_FIELD":"NONE","FIELD_CFG_P0":"NOT_RUN","FIELD_DECOUPLED_CFG_FORMAL256":"NOT_RUN","FIELD_DECOUPLING_EFFECT":"NOT_SUPPORTED","FIELD_STAGE_EFFECT":"NOT_RUN","INNOVATION1_FINAL_STATUS":"MIXED","stage":"FROZEN_READY","active_phase":"calibration","phase_complete":0,"phase_total":48}; (ROOT/"pipeline_status.json").write_text(json.dumps(status,indent=2)+"\n"); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
