"""Run frozen MatterSim quality evaluation for every field policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from experiments.field_decoupled_adaptive_cfg.run_generation import phase_policies


PROJECT = Path(__file__).resolve().parents[2]; ROOT = PROJECT / "experiments/field_decoupled_adaptive_cfg"
REFERENCE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/reference_MP2020correction.gz")
POTENTIAL = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"; PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def run_group(phase: str, group: str, gpu: int, memory_fd: int) -> None:
    phase_root=ROOT/phase; structure_path=phase_root/"evaluation_structures"/f"{group}.extxyz"
    if not structure_path.exists(): print(f"skip empty {phase}/{group}",flush=True); return
    output=phase_root/"quality"/group
    if (output/"official_detailed.json.gz").exists(): print(f"already complete {phase}/{group}",flush=True); return
    if output.exists(): raise FileExistsError(output)
    log_path=ROOT/"logs"/f"quality_{phase}_{group}_gpu_{gpu}.log"; env=dict(os.environ); env.update(CUDA_VISIBLE_DEVICES=str(gpu),FIELD_CFG_PHYSICAL_GPU=str(gpu),TMPDIR=str(phase_root/"runtime_tmp"),XDG_CACHE_HOME=str(PROJECT/".cache"),HF_HOME=str(PROJECT/".cache/huggingface"),TORCH_HOME=str(PROJECT/".cache/torch"),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1")
    command=[PYTHON,"-m","research.corrector_distillation.evaluate_quality","--method",group,"--structures-path",str(structure_path),"--potential-path",str(POTENTIAL),"--reference-memory-fd",str(memory_fd),"--output-dir",str(output),"--temporary-dir",str(phase_root/"runtime_tmp"),"--device","cuda"]
    with log_path.open("x",encoding="utf-8") as stream: result=subprocess.run(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(memory_fd,))
    if result.returncode: raise RuntimeError(f"quality failed {phase}/{group}: {log_path}")


def worker(phase: str,gpu:int,groups:tuple[str,...],fd:int)->None:
    for group in groups: run_group(phase,group,gpu,fd)


def main(phase:str,gpus:tuple[int,...])->None:
    if sha256(POTENTIAL)!=POTENTIAL_SHA256: raise RuntimeError("MatterSim hash mismatch")
    if not gpus or len(set(gpus))!=len(gpus): raise ValueError("invalid GPU list")
    phase_root=ROOT/phase; (phase_root/"quality").mkdir(exist_ok=True); (phase_root/"runtime_tmp").mkdir(exist_ok=True); (ROOT/"logs").mkdir(exist_ok=True)
    groups=tuple(sorted(policy["policy_id"] for policy in phase_policies(phase))); assignments={gpu:tuple(group for i,group in enumerate(groups) if i%len(gpus)==rank) for rank,gpu in enumerate(gpus)}
    fd=os.memfd_create(f"field_cfg_{phase}_reference",os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE,"rb") as source:
        with os.fdopen(os.dup(fd),"wb") as destination: shutil.copyfileobj(source,destination,length=16*1024*1024)
    fcntl.fcntl(fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE|fcntl.F_SEAL_SEAL)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,phase,gpu,groups_for_gpu,fd) for gpu,groups_for_gpu in assignments.items()]
            for future in futures: future.result()
    finally: os.close(fd)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("calibration","p0","formal256"),required=True); parser.add_argument("--gpus",required=True); args=parser.parse_args(); main(args.phase,tuple(int(v) for v in args.gpus.split(",") if v))
