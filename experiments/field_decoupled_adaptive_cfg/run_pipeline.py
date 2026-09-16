"""Gate-enforcing supervisor from field calibration through Formal256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT=Path(__file__).resolve().parents[2]; ROOT=PROJECT/"experiments/field_decoupled_adaptive_cfg"; PYTHON="/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def assert_frozen()->None:
    manifest=json.loads((ROOT/"frozen_manifest.json").read_text())
    for relative,expected in manifest["sha256"].items():
        if sha256(PROJECT/relative)!=expected: raise RuntimeError(f"frozen file changed: {relative}")


def update_status(**updates)->None:
    path=ROOT/"pipeline_status.json"; values=json.loads(path.read_text()); values.update(updates); temporary=path.with_suffix(".json.partial"); temporary.write_text(json.dumps(values,indent=2)+"\n"); os.replace(temporary,path)


def audit_selected_gpus(gpus:tuple[int,...])->None:
    completed=subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True); snapshot={}
    for line in completed.stdout.splitlines():
        fields=[field.strip() for field in line.split(",")]; index=int(fields[0]); snapshot[index]={"index":index,"name":fields[1],"memory_total_mib":int(fields[2]),"memory_used_mib":int(fields[3]),"utilization_percent":int(fields[4])}
    missing=sorted(set(gpus)-set(snapshot)); busy=[snapshot[gpu] for gpu in gpus if snapshot[gpu]["memory_used_mib"]>1024 or snapshot[gpu]["utilization_percent"]>10]; audit={"selected_gpus":list(gpus),"idle_rule":"memory_used_mib <= 1024 and utilization_percent <= 10","snapshot":[snapshot[i] for i in sorted(snapshot)],"selected_missing":missing,"selected_busy":busy,"launch_allowed":not missing and not busy}; (ROOT/"launch_gpu_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    if missing or busy: raise RuntimeError(f"selected GPU audit failed: missing={missing}, busy={busy}")


def environment(gpu:int|None=None)->dict[str,str]:
    result=dict(os.environ); result.update(PYTHONPATH=str(PROJECT),PYTHONUNBUFFERED="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MPLCONFIGDIR=str(ROOT/".matplotlib"))
    if gpu is not None: result.update(CUDA_VISIBLE_DEVICES=str(gpu),FIELD_CFG_PHYSICAL_GPU=str(gpu))
    return result


def run_logged(command:list[str],name:str,gpu:int|None=None)->None:
    path=ROOT/"logs"/name
    if path.exists(): raise FileExistsError(path)
    with path.open("x",encoding="utf-8") as stream: result=subprocess.run(command,cwd=PROJECT,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"command failed {result.returncode}: {path}")


def generation(phase:str,gpus:tuple[int,...])->None:
    phase_root=ROOT/phase; seeds=list(map(int,json.loads((phase_root/"seeds.json").read_text())["seeds"])); pending=[seed for seed in seeds if not (phase_root/"generation"/str(seed)/"run_summary.json").exists()]; partitions={gpu:pending[index::len(gpus)] for index,gpu in enumerate(gpus)}; processes=[]
    for gpu,assigned in partitions.items():
        if not assigned: continue
        log=ROOT/"logs"/f"generation_{phase}_gpu_{gpu}.log"
        if log.exists(): raise FileExistsError(log)
        stream=log.open("x",encoding="utf-8"); command=[PYTHON,str(ROOT/"run_generation.py"),"--phase",phase,"--seeds",",".join(map(str,assigned))]; process=subprocess.Popen(command,cwd=PROJECT,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT); processes.append((gpu,process,stream,log))
    while processes:
        running=[]; failure=None
        for gpu,process,stream,log in processes:
            code=process.poll()
            if code is None: running.append((gpu,process,stream,log))
            else:
                stream.close()
                if code: failure=(gpu,code,log); break
        if failure:
            for _,process,stream,_ in running:
                if process.poll() is None: process.terminate()
                stream.close()
            raise RuntimeError(f"worker failure {failure}")
        processes=running; complete=sum((phase_root/"generation"/str(seed)/"run_summary.json").exists() for seed in seeds); update_status(stage=f"{phase}_generation",active_phase=phase,phase_complete=complete,phase_total=len(seeds)); print(json.dumps({"stage":f"{phase}_generation","complete":complete,"total":len(seeds)}),flush=True)
        if processes: time.sleep(30)
    complete=sum((phase_root/"generation"/str(seed)/"run_summary.json").exists() for seed in seeds)
    if complete!=len(seeds): raise RuntimeError(f"{phase} incomplete {complete}/{len(seeds)}")


def materialize_seeds(phase:str)->None:
    path=ROOT/phase/"seeds.json"
    if path.exists(): return
    registry=json.loads((ROOT/"seed_registry.json").read_text()); key="p0_seeds" if phase=="p0" else "formal256_seeds"; seeds=list(map(int,registry[key])); path.parent.mkdir(parents=True,exist_ok=False); path.write_text(json.dumps({"schema_version":1,"purpose":phase,"materialized_only_after_required_gate":True,"count":len(seeds),"seeds":seeds,"historical_overlap":0},indent=2)+"\n")


def run_phase(phase:str,gpus:tuple[int,...])->dict:
    generation(phase,gpus); update_status(stage=f"{phase}_property_evaluation"); run_logged([PYTHON,str(ROOT/"prepare_phase.py"),"--phase",phase],f"prepare_{phase}.log",gpus[0]); update_status(stage=f"{phase}_quality_evaluation"); run_logged([PYTHON,str(ROOT/"run_quality.py"),"--phase",phase,"--gpus",",".join(map(str,gpus))],f"quality_driver_{phase}.log"); update_status(stage=f"{phase}_statistics_bootstrap_plots_report"); run_logged([PYTHON,str(ROOT/"analyze_phase.py"),"--phase",phase],f"analysis_{phase}.log"); return json.loads((ROOT/phase/"decision_summary.json").read_text())


def main(gpus:tuple[int,...])->None:
    assert_frozen()
    if not gpus or len(set(gpus))!=len(gpus): raise ValueError("invalid GPU list")
    audit_selected_gpus(gpus); (ROOT/"logs").mkdir(exist_ok=True); calibration=run_phase("calibration",gpus); update_status(**calibration)
    if calibration["FIELD_DECOUPLING_CALIBRATION"]!="GO": update_status(stage="stopped_after_calibration_fail",active_phase=None); return
    materialize_seeds("p0"); p0=run_phase("p0",gpus); update_status(**p0)
    if p0["FIELD_CFG_P0"]!="GO": update_status(stage="stopped_after_p0_fail",active_phase=None); return
    materialize_seeds("formal256"); formal=run_phase("formal256",gpus); update_status(**formal,stage="complete",active_phase=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--gpus",required=True); args=parser.parse_args(); main(tuple(int(value) for value in args.gpus.split(",") if value))
