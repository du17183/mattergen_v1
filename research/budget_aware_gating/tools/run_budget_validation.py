#!/usr/bin/env python3
"""Resume-safe 32/64-seed Budget-Aware Corrector Gating validation."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, shutil, signal, statistics, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import ase.io
import numpy as np
import pandas as pd

ROOT=Path("/data/dxl"); PROJECT=ROOT/"mattergen_v1"
RESULT=ROOT/"results/budget_aware_gating"; REPORT=ROOT/"reports/budget_aware_gating"
TOOLS=ROOT/"tools/budget_aware_gating"; LOG=ROOT/"logs/budget_aware_gating"
DEV=RESULT/"development"; DEV_PROGRESS=DEV/"progress"; GENERATION=DEV/"generation"; RELAXED=DEV/"relaxed"
DEV_REPORT=REPORT/"development"; FINAL=REPORT/"final"; FIGURES=FINAL/"figures"
PYTHON=ROOT/"envs/mattergen_py310/bin/python"; TASK_RUNNER=TOOLS/"run_budget_task.py"
SMOKE=REPORT/"eight_seed_go_no_go.json"; CONFIG_ROOT=REPORT/"frozen_candidate_configs"
OLD_G3=REPORT/"frozen_formal_baseline/innovation2_g3_config.json"; STOP=RESULT/"progress/stop_requested"
TZ=ZoneInfo("Asia/Shanghai")
LEVEL1=("rng_state_hash","initial_atomic_numbers_hash","initial_pos_hash","initial_cell_hash","initial_state_hash","final_structure_hash","extxyz_sha256")

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
core=module("budget_core",ROOT/"tools/innovation2_next/run_corrector_32.py")
met=module("budget_metrics",ROOT/"tools/innovation2_next/analyze_corrector_64.py")
prog=module("budget_progress",TOOLS/"progress.py")
def now(): return datetime.now(TZ).isoformat(timespec="seconds")
def clean(v):
    if isinstance(v,np.bool_): return bool(v)
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,(np.floating,float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
    return v
def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x",encoding="utf-8") as f: f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def jwrite(path,value): text(path,json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def cwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp.{os.getpid()}")
    pd.DataFrame([clean(x) for x in rows]).to_csv(tmp,index=False); os.replace(tmp,path)
def read(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def stage(name,status,message,fields=None): prog.update(name,status,message,fields or {})

def a0():
    return dict(config_id="A0",gating_enabled=False,budget_aware_enabled=False,warmup_frac=.15,min_progress=.15,max_progress=.95,convergence_threshold=.05,consecutive_stable_steps=3,calibration_interval=10,max_consecutive_skips=8,fallback_threshold=.2,rescue_enabled=True,max_skip_ratio=1.,atomic_veto_enabled=False,atomic_stability_threshold=.05,atomic_min_stable_steps=1,adaptive_calibration_enabled=False,calibration_interval_min=4,calibration_interval_max=16,field_aggregation="all_fields")
def g3():
    g=read(OLD_G3)["corrector_gating"]; interval=g["corrector_calibration_interval"]
    return dict(config_id="G3",gating_enabled=True,budget_aware_enabled=False,warmup_frac=g["corrector_warmup_frac"],min_progress=g["corrector_min_progress"],max_progress=g["corrector_max_progress"],convergence_threshold=g["corrector_convergence_threshold"],consecutive_stable_steps=g["corrector_consecutive_stable_steps"],calibration_interval=interval,max_consecutive_skips=g["corrector_max_consecutive_skips"],fallback_threshold=g["corrector_fallback_threshold"],rescue_enabled=g["corrector_rescue_enabled"],max_skip_ratio=1.,atomic_veto_enabled=False,atomic_stability_threshold=.05,atomic_min_stable_steps=1,adaptive_calibration_enabled=False,calibration_interval_min=interval,calibration_interval_max=interval,field_aggregation="all_fields")
def configs(names):
    allc={"A0":a0(),"G3":g3()}
    for n in ("G1","G2"):
        if (CONFIG_ROOT/f"{n}.json").exists(): allc[n]=read(CONFIG_ROOT/f"{n}.json")
    return {n:allc[n] for n in names}
def args(c):
    flag=lambda n,v:f"--{n}" if v else f"--no-{n}"
    out=[]
    if c["gating_enabled"]: out += ["--gating-enabled"]
    if c["budget_aware_enabled"]: out += ["--budget-aware-enabled"]
    out += ["--warmup",str(c["warmup_frac"]),"--min-progress",str(c["min_progress"]),"--max-progress",str(c["max_progress"]),"--threshold",str(c["convergence_threshold"]),"--stable-steps",str(c["consecutive_stable_steps"]),"--calibration-interval",str(c["calibration_interval"]),"--max-skips",str(c["max_consecutive_skips"]),"--fallback-threshold",str(c["fallback_threshold"]),"--max-skip-ratio",str(c["max_skip_ratio"]),flag("atomic-veto-enabled",c["atomic_veto_enabled"]),"--atomic-threshold",str(c["atomic_stability_threshold"]),"--atomic-stable-steps",str(c["atomic_min_stable_steps"]),flag("adaptive-calibration-enabled",c["adaptive_calibration_enabled"]),"--calibration-min",str(c["calibration_interval_min"]),"--calibration-max",str(c["calibration_interval_max"]),"--field-aggregation",c["field_aggregation"],flag("rescue-enabled",c["rescue_enabled"]),"--trace","off"]
    return out
def configure(names,seeds,timing):
    core.RESULT=DEV; core.REPORT=DEV_REPORT; core.LOG=LOG/"development"; core.PROGRESS=DEV_PROGRESS
    core.GENERATION=GENERATION; core.RELAXED=RELAXED; core.REPEATS=DEV/"determinism_repeats"; core.TASK_RUNNER=TASK_RUNNER
    core.MASTER_JSON=DEV_PROGRESS/"unused_master.json"; core.MASTER_CSV=DEV_PROGRESS/"unused_master.csv"
    core.GEN_JSON=DEV_PROGRESS/"generation_progress.json"; core.GEN_CSV=DEV_PROGRESS/"generation_progress.csv"
    core.RELAX_JSON=DEV_PROGRESS/"relax_progress.json"; core.RELAX_CSV=DEV_PROGRESS/"relax_progress.csv"
    core.EVENTS=DEV_PROGRESS/"events.jsonl"; core.LAUNCHER=DEV_PROGRESS/"launcher.json"; core.STOP_MARKER=STOP; core.WAVE_TIMING=timing
    core.CONFIGS=tuple(names); core.SEEDS=list(seeds); cmap=configs(names); core.set_stage=lambda *a,**k:None
    def command(output,seed,config,gpu,slot,workers):
        return [str(PYTHON),str(TASK_RUNNER),"--output-dir",str(output),"--seed",str(seed),"--physical-gpu",str(gpu),"--gpu-slot",str(slot),"--workers-per-gpu",str(workers),"--config-id",config,"--sampling-steps","1000",*args(cmap[config])]
    core.task_command=command

def generation_state():
    expected=core.generation_tasks()
    if core.GEN_JSON.exists():
        state=read(core.GEN_JSON); ids={x["task_id"] for x in state["tasks"]}; state["tasks"] += [x for x in expected if x["task_id"] not in ids]
    else: state={"schema_version":2,"created_at":now(),"updated_at":now(),"full_status":"pending","tasks":expected}
    core.save_generation_state(state); return core.load_generation_state()
def snapshot(name,names,seeds):
    selected=[x for x in core.load_generation_state()["tasks"] if x["config"] in names and int(x["seed"]) in set(seeds)]
    jwrite(RESULT/"progress"/name,{"updated_at":now(),"tasks":selected,"counts":{s:sum(x["status"]==s for x in selected) for s in sorted({x["status"] for x in selected})}})
def run_generation(names,seeds,timing_path):
    configure(names,seeds,timing_path); state=generation_state(); timing=read(timing_path) if timing_path.exists() else {}
    seedset=set(seeds)
    for name in names:
        pending=sum(x["config"]==name and int(x["seed"]) in seedset and not core.validate_generation_output(Path(x["output_dir"]),int(x["seed"])) for x in state["tasks"])
        result=core.run_generation_wave(name,state,4)
        if not result["success"]: raise RuntimeError(f"generation failed: {name}")
        if name not in timing and pending:
            elapsed=float(result["elapsed_seconds"]); timing[name]={"config":name,"workers_per_gpu":4,"total_concurrency":32,"structures":pending,"elapsed_seconds":elapsed,"structures_per_hour":pending*3600/max(elapsed,1e-9),"measurement":"same-concurrency pending-task wave","recorded_at":now()}; jwrite(timing_path,timing)
        elif name not in timing:
            rows=[core.generation_result(x) for x in state["tasks"] if x["config"]==name and int(x["seed"]) in seedset]
            elapsed=max(float(x["elapsed_seconds"]) for x in rows); timing[name]={"config":name,"workers_per_gpu":4,"total_concurrency":32,"structures":len(rows),"elapsed_seconds":elapsed,"structures_per_hour":len(rows)*3600/elapsed,"measurement":"resume-derived max task elapsed","recorded_at":now()}; jwrite(timing_path,timing)

def freeze_mattersim():
    path=DEV_REPORT/"mattersim_config.json"
    if path.exists(): return
    src=ROOT/"reports/guidance_stage7_eval/relax_config_frozen.json"; c=read(src)
    c.update(source_config=str(src),source_config_sha256=sha(src),STABILITY_SOURCE="MatterSim-5M surrogate",DFT_VERIFIED=False,PROPERTY_TARGET_VERIFIED=False,TRI2024_reference=str(ROOT/"reference_assets/reference_TRI2024correction.gz"),TRI2024_reference_sha256="3631b54625f2a5410fb83aab16fda78073a2a713e8457e3beec523d0682315f5",structure_matcher="DefaultDisorderedStructureMatcher",energy_correction="TRI110Compatibility2024",optimizer="FIRE",cell_filter="EXPCELLFILTER",fmax_ev_ang=.05,max_steps=500,stability_threshold_ev_atom=.1,workers_per_gpu=2,total_concurrency=16,freeze_time=now())
    jwrite(path,c); text(DEV_REPORT/"mattersim_config.sha256",f"{sha(path)}  mattersim_config.json\n")
def relax_state():
    expected=core.relax_initial()
    if core.RELAX_JSON.exists():
        state=read(core.RELAX_JSON); ids={x["task_id"] for x in state["tasks"]}; state["tasks"] += [x for x in expected["tasks"] if x["task_id"] not in ids]
    else: state=expected
    core.save_relax(state); return core.relax_initialize()
def run_relax():
    relax_state(); processes=[]; logs=LOG/"development_relax"; logs.mkdir(parents=True,exist_ok=True)
    for gpu in range(8):
        for slot in range(2):
            env=core.task_environment(gpu,f"budget_relax_gpu{gpu}_{slot}"); env["MATTERGEN_BUDGET_RELAX_WORKER"]="1"; stream=(logs/f"gpu{gpu}_slot{slot}.log").open("a")
            p=subprocess.Popen([str(PYTHON),str(TOOLS/"run_budget_validation.py"),"relax-worker","--gpu",str(gpu),"--slot",str(slot)],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            with core.children_lock: core.children[p.pid]=p
            processes.append((p,stream))
    for p,s in processes:
        p.wait(); s.close()
        with core.children_lock: core.children.pop(p.pid,None)
    state=core.relax_initialize()
    if not all(core.validate_relax(x) for x in state["tasks"]): raise RuntimeError("strict MatterSim validation failed")

def generation_rows(names,seeds):
    wanted={(n,s) for n in names for s in seeds}; tasks=[x for x in core.load_generation_state()["tasks"] if (x["config"],int(x["seed"])) in wanted]
    if len(tasks)!=len(wanted) or not all(core.validate_generation_output(Path(x["output_dir"]),int(x["seed"])) for x in tasks): raise RuntimeError("generation outputs incomplete")
    return [core.generation_result(x) for x in tasks]
def integrity(names,seeds,rows):
    by={(x["config"],int(x["seed"])):x for x in rows}; mismatches=[]
    for seed in seeds:
        for name in names[1:]:
            if any(by[("A0",seed)][k]!=by[(name,seed)][k] for k in LEVEL1[:5]): mismatches.append({"seed":seed,"config":name})
    smoke=read(SMOKE)
    out={"generated_at":now(),"configs":names,"seeds":[min(seeds),max(seeds)],"expected":len(names)*len(seeds),"total_success":len(rows),"generation_success_100_percent":len(rows)==len(names)*len(seeds),"basic_structure_validity":statistics.mean(bool(x["basic_structure_valid"]) for x in rows),"composition_validity":{n:statistics.mean(bool(x["composition_valid"]) for x in rows if x["config"]==n) for n in names},"cross_config_initial_hash_mismatch_count":len(mismatches),"cross_config_initial_hash_mismatches":mismatches,"same_config_same_seed_level1_inherited_from_8_seed":{n:smoke["aggregate"][n]["determinism_level1"] for n in names}}
    if mismatches or out["basic_structure_validity"]!=1.: raise RuntimeError("generation integrity failed")
    return out
def frames(names,seeds,grows):
    from pymatgen.io.ase import AseAtomsAdaptor
    gen={(x["config"],int(x["seed"])):x for x in grows}; relax={(x["config"],int(x["seed"])):x for x in read(core.RELAX_JSON)["tasks"]}; out={}; failures=[]
    for name in names:
        rows=[]
        for seed in seeds:
            task=relax[(name,seed)]; path=Path(task["output_path"]); summary=read(path.parent/"relax_summary.json"); original=ase.io.read(Path(gen[(name,seed)]["output_dir"])/"generated_crystals.extxyz"); relaxed=ase.io.read(path); struct=AseAtomsAdaptor.get_structure(relaxed)
            rows.append(dict(config=name,seed=seed,task_id=task["task_id"],energy_ev=float(summary["energy_ev"]),energy_per_atom_ev=float(summary["energy_per_atom_ev"]),maximum_force_ev_ang=float(summary["maximum_force_ev_ang"]),converged=bool(summary["converged"]),relax_elapsed_seconds=float(summary["elapsed_seconds"]),steps=int(summary["steps"]),formula=struct.composition.reduced_formula,chemical_system=struct.composition.chemical_system,input_hash=summary["input_hash"],output_hash=summary["output_hash"],_relaxed_atoms=relaxed,_original_atoms=original))
            if not summary["converged"]: failures.append({"stage":"relax","config":name,"seed":seed,"failure_type":"valid_max_step_nonconvergence","details":f"steps={summary['steps']}; max_force={summary['maximum_force_ev_ang']}"})
        out[name]=pd.DataFrame(rows)
    return out,failures
def stat_remap(row,comparison,candidate):
    row=dict(row); row["comparison"]=comparison
    for suffix in ("wins","losses"):
        if f"G3_{suffix}" in row: row[f"{candidate}_{suffix}"]=row.pop(f"G3_{suffix}")
    return row
def paired(names,seeds,fs,grows):
    gen={n:pd.DataFrame([x for x in grows if x["config"]==n]).sort_values("seed").reset_index(drop=True) for n in names}; records=[]; stats=[]; pairs=[("A0",n) for n in names[1:]]
    if "G1" in names and "G2" in names: pairs.append(("G1","G2"))
    for left,right in pairs:
        a=fs[left].sort_values("seed").reset_index(drop=True); b=fs[right].sort_values("seed").reset_index(drop=True); ga=gen[left]; gb=gen[right]
        for i,seed in enumerate(seeds): records.append({"comparison":f"{left}_vs_{right}","seed":seed,f"elapsed_{left}":ga.loc[i,"elapsed_seconds"],f"elapsed_{right}":gb.loc[i,"elapsed_seconds"],"speed_multiplier_left_over_right":ga.loc[i,"elapsed_seconds"]/gb.loc[i,"elapsed_seconds"],f"physical_forward_{left}":ga.loc[i,"physical_model_forward_count"],f"physical_forward_{right}":gb.loc[i,"physical_model_forward_count"],f"energy_above_hull_{left}":a.loc[i,"energy_above_hull_per_atom"],f"energy_above_hull_{right}":b.loc[i,"energy_above_hull_per_atom"],f"stable_{left}":bool(a.loc[i,"stable"]),f"stable_{right}":bool(b.loc[i,"stable"]),f"NUS_{left}":bool(a.loc[i,"novel_unique_stable"]),f"NUS_{right}":bool(b.loc[i,"novel_unique_stable"])})
        comp=f"{right}-{left}"
        for x,y,m,lower in ((ga.elapsed_seconds,gb.elapsed_seconds,"generation_elapsed_seconds",True),(ga.physical_model_forward_count,gb.physical_model_forward_count,"physical_model_forward_count",True),(a.energy_above_hull_per_atom,b.energy_above_hull_per_atom,"energy_above_hull_per_atom",True)): stats.append(stat_remap(met.continuous_stats(x.to_numpy(),y.to_numpy(),m,lower),comp,right))
        for x,y,m in ((a.stable,b.stable,"stable"),(a.comp_validity,b.comp_validity,"composition_valid"),(a.novel,b.novel,"novel"),(a.unique,b.unique,"unique"),(a.novel_unique_stable,b.novel_unique_stable,"novel_unique_stable"),(a.converged,b.converged,"relax_converged")): stats.append(stat_remap(met.binary_stats(x.to_numpy(),y.to_numpy(),m),comp,right))
    return records,stats
def summaries(names,seeds,fs,official,grows,timing_path):
    timing=read(timing_path); out={}
    for n in names:
        rows=[x for x in grows if x["config"]==n]; elapsed=[float(x["elapsed_seconds"]) for x in rows]; f=fs[n]
        out[n]={"config":n,"paired_seeds":len(seeds),"generation_success_rate":1.,"generation_composition_validity":statistics.mean(bool(x["composition_valid"]) for x in rows),"generation_basic_structure_validity":statistics.mean(bool(x["basic_structure_valid"]) for x in rows),"median_elapsed_seconds":statistics.median(elapsed),"p95_elapsed_seconds":float(np.quantile(elapsed,.95)),"physical_forward_mean":statistics.mean(float(x["physical_model_forward_count"]) for x in rows),"corrector_skip_rate":statistics.mean(float(x["corrector_skip_rate"]) for x in rows),"budget_used_mean":statistics.mean(float(read(Path(x["output_dir"])/"corrector_summary.json").get("corrector_budget_used",0)) for x in rows),"atomic_veto_rate":statistics.mean(float(read(Path(x["output_dir"])/"corrector_summary.json").get("corrector_atomic_veto_count",0))/1000 for x in rows),"calibration_rate":statistics.mean(float(x["calibration_rate"]) for x in rows),"fallback_rate":statistics.mean(float(x["fallback_rate"]) for x in rows),"rescue_rate":statistics.mean(float(x["rescue_rate"]) for x in rows),"fixed_concurrency_wave_seconds":timing[n]["elapsed_seconds"],"fixed_concurrency_structures_per_hour":timing[n]["structures_per_hour"],"force_convergence_rate":float(f.converged.astype(bool).mean()),**official[n]}
    a=out["A0"]
    for n,x in out.items():
        x.update(speed_multiplier=a["median_elapsed_seconds"]/x["median_elapsed_seconds"],throughput_multiplier=x["fixed_concurrency_structures_per_hour"]/a["fixed_concurrency_structures_per_hour"],physical_forward_reduction=1-x["physical_forward_mean"]/a["physical_forward_mean"],stable_change=x["frac_stable_structures"]-a["frac_stable_structures"],NUS_change=x["frac_novel_unique_stable_structures"]-a["frac_novel_unique_stable_structures"],Ehull_change_ev_atom=x["avg_energy_above_hull_per_atom"]-a["avg_energy_above_hull_per_atom"])
    return out
def gate(name,sums,stats):
    x=sums[name]; a=sums["A0"]; t={"G1":dict(speed=1.15,forward=.12,stable=-.02,nus=-.02,ehull=.01),"G2":dict(speed=1.25,forward=.20,stable=-.03,nus=-.03,ehull=.015)}[name]; comp=next(r for r in stats if r["comparison"]==f"{name}-A0" and r["metric"]=="composition_valid")
    gates={"median_speed_multiplier":x["speed_multiplier"]>=t["speed"],"physical_forward_reduction":x["physical_forward_reduction"]>=t["forward"],"stable_decline":x["stable_change"]>=t["stable"],"NUS_decline":x["NUS_change"]>=t["nus"],"Ehull_degradation":x["Ehull_change_ev_atom"]<=t["ehull"],"generation_success_100_percent":x["generation_success_rate"]==1.,"structure_validity_100_percent":x["generation_basic_structure_validity"]==1.,"composition_validity_no_significant_decline":x["generation_composition_validity"]>=a["generation_composition_validity"] or float(comp["exact_discordant_binomial_p_value"])>=.05,"same_seed_level1":bool(read(SMOKE)["aggregate"][name]["determinism_level1"])}
    return {"go":all(gates.values()),"gates":gates,"targets":t}
def evaluate(label,names,seeds,timing):
    out=DEV_REPORT/label; out.mkdir(parents=True,exist_ok=True); grows=generation_rows(names,seeds); jwrite(out/"generation_integrity_report.json",integrity(names,seeds,grows)); cwrite(out/"generation_manifest.csv",grows); fs,failures=frames(names,seeds,grows); met.REPORT=out; met.CONFIGS=tuple(names); met.SEEDS=list(seeds); official,errors=met.official_metrics(fs); records,stats=paired(names,seeds,fs,grows); sums=summaries(names,seeds,fs,official,grows,timing); decisions={n:gate(n,sums,stats) for n in names[1:]}; jwrite(out/"quality_speed_summary.json",{"configs":sums,"metric_errors":errors}); cwrite(out/"quality_speed_summary.csv",list(sums.values())); cwrite(out/"paired_seed_records.csv",records); cwrite(out/"paired_statistics.csv",stats); cwrite(out/"failure_cases.csv",failures); jwrite(out/"candidate_decisions.json",{"stage":label,"seeds":[min(seeds),max(seeds)],"decisions":decisions,"STABILITY_SOURCE":"MatterSim-5M surrogate","DFT_VERIFIED":False,"PROPERTY_TARGET_VERIFIED":False}); return sums,decisions,records,stats,failures,grows
def freeze(passed,decisions):
    value={"freeze_time":now(),"source_commit":subprocess.check_output(["git","-C",str(PROJECT),"rev-parse","HEAD"],text=True).strip(),"candidates":{},"screen_decisions":decisions,"parameter_retuning_after_freeze_allowed":False}
    for n in passed[:2]:
        target=DEV_REPORT/"frozen_pareto_candidates"/f"{n}.json"; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(CONFIG_ROOT/f"{n}.json",target); value["candidates"][n]={"path":str(target),"sha256":sha(target)}
    jwrite(DEV_REPORT/"frozen_pareto_candidates.json",value)
def figures(pareto,records,grows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True)
    for filename,x,y,xlabel,ylabel in (("speed_vs_ehull.png","speed_multiplier","avg_energy_above_hull_per_atom","Speed multiplier","Average E-hull"),("speed_vs_stable.png","speed_multiplier","frac_stable_structures","Speed multiplier","Stable fraction"),("speed_vs_nus.png","speed_multiplier","frac_novel_unique_stable_structures","Speed multiplier","NUS fraction"),("forward_reduction_vs_quality.png","physical_forward_reduction","avg_energy_above_hull_per_atom","Physical forward reduction","Average E-hull"),("skip_ratio_vs_quality.png","corrector_skip_rate","frac_stable_structures","Corrector skip ratio","Stable fraction")):
        fig,ax=plt.subplots(figsize=(6,4))
        for r in pareto: ax.scatter(r.get(x,0),r.get(y,np.nan),s=60); ax.annotate(r["config"],(r.get(x,0),r.get(y,np.nan)))
        ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/filename,dpi=180); plt.close(fig)
    dev=[r for r in pareto if r["experiment_stage"]=="new_development_14000_14063"]; keys=("budget_used_mean","atomic_veto_rate","calibration_rate","fallback_rate","rescue_rate"); fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(dev)); width=.16
    for i,k in enumerate(keys): ax.bar(x+(i-2)*width,[(r.get(k,0)/1000 if k=="budget_used_mean" else r.get(k,0)) for r in dev],width,label=k)
    ax.set_xticks(x,[r["config"] for r in dev]); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES/"mechanism_rates.png",dpi=180); plt.close(fig)
    names=[r["config"] for r in dev]; budget=[]; veto=[]
    for name in names:
        summaries=[read(Path(row["output_dir"])/"corrector_summary.json") for row in grows if row["config"]==name]
        budget.append([float(item.get("corrector_budget_used",0))/1000 for item in summaries]); veto.append([float(item.get("corrector_atomic_veto_count",0))/1000 for item in summaries])
    fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot(budget,labels=names); ax.set(ylabel="Budget used / 1000 steps",title="Skip-budget utilization distribution"); fig.tight_layout(); fig.savefig(FIGURES/"budget_used_distribution.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot(veto,labels=names); ax.set(ylabel="Atomic veto count / 1000 steps",title="Atomic-number veto rate"); fig.tight_layout(); fig.savefig(FIGURES/"atomic_veto_rate.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); mech=("calibration_rate","fallback_rate","rescue_rate"); width=.24
    for i,key in enumerate(mech): ax.bar(x+(i-1)*width,[r.get(key,0) for r in dev],width,label=key)
    ax.set_xticks(x,names); ax.legend(); ax.set_title("Calibration / fallback / rescue rates"); fig.tight_layout(); fig.savefig(FIGURES/"calibration_fallback_rescue.png",dpi=180); plt.close(fig)
    values=[r["speed_multiplier_left_over_right"] for r in records if r["comparison"].startswith("A0_vs_")]; fig,ax=plt.subplots(figsize=(6,4)); ax.hist(values,bins=16); ax.axvline(1,color="black",ls="--"); ax.set(xlabel="Paired speedup",ylabel="Count"); fig.tight_layout(); fig.savefig(FIGURES/"paired_speedup.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for r in pareto: ax.scatter(r["speed_multiplier"],r["avg_energy_above_hull_per_atom"],s=90); ax.annotate(f"{r['config']} ({r['experiment_stage']})",(r["speed_multiplier"],r["avg_energy_above_hull_per_atom"]),fontsize=7)
    ax.set(xlabel="Speed multiplier",ylabel="Average E-hull",title="Quality-speed Pareto frontier"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"pareto_frontier.png",dpi=180); plt.close(fig)
def final(names,sums,decisions,records,stats,failures):
    FINAL.mkdir(parents=True,exist_ok=True); old=read(OLD_G3)["decision"]; pareto=[]; allseeds=list(range(14000,14064)); grows=generation_rows(names,allseeds)
    relaxrows=[row for row in read(core.RELAX_JSON)["tasks"] if row["config"] in names and int(row["seed"]) in set(allseeds)]
    for n in names: pareto.append({**sums[n],"experiment_stage":"new_development_14000_14063","paired_with_A0":True})
    gs=old["speed"]; gq=old["quality"]["G3"]; pareto.append({"config":"G3","experiment_stage":"old_formal_independent_seeds","paired_with_A0":False,"speed_multiplier":1+gs["single_worker_normalized_median_speedup"],"throughput_multiplier":1+gs["fixed_concurrency_throughput_gain"],"physical_forward_reduction":gs["physical_forward_reduction"],"corrector_skip_rate":old["mechanism"]["corrector_skip_rate"],"avg_energy_above_hull_per_atom":gq["avg_energy_above_hull_per_atom"],"frac_stable_structures":gq["frac_stable_structures"],"frac_novel_unique_stable_structures":gq["frac_novel_unique_stable_structures"],"warning":"Different old formal seed set; excluded from new paired statistics."})
    cwrite(FINAL/"generation_manifest.csv",grows); cwrite(FINAL/"relax_manifest.csv",relaxrows)
    validated=any(x["go"] for x in decisions.values()); payload={"BUDGET_AWARE_GATING_COMPLETED":True,"BUDGET_AWARE_GATING_DEVELOPMENT_VALIDATED":validated,"development_seeds":[14000,14063],"paired_seeds":64,"configs":sums,"candidate_decisions":decisions,"G3_FORMAL_REFERENCE":pareto[-1],"PARETO_FRONTIER":[x["config"] for x in pareto],"STABILITY_SOURCE":"MatterSim-5M surrogate","DFT_VERIFIED":False,"PROPERTY_TARGET_VERIFIED":False,"FORMAL_30000_SEEDS_STARTED":False,"limitations":["Development-scale 64-seed evidence; no new independent formal seeds were run.","MatterSim-5M stability is a surrogate and is not DFT verification.","Old G3 formal results use a different seed set and are not mixed into paired tests."]}; jwrite(FINAL/"budget_aware_final_report.json",payload); cwrite(FINAL/"pareto_comparison.csv",pareto); jwrite(FINAL/"pareto_comparison.json",{"rows":pareto}); cwrite(FINAL/"paired_seed_records.csv",records); cwrite(FINAL/"paired_statistics.csv",stats); cwrite(FINAL/"failure_cases.csv",failures); jwrite(FINAL/"frozen_candidate_configs.json",read(DEV_REPORT/"frozen_pareto_candidates.json")); cwrite(FINAL/"seed_manifest.csv",[{"seed":s,"split":"development","screen_32":s<=14031,"extension_only":s>=14032,"formal":False} for s in range(14000,14064)])
    env={"generated_at":now(),"hostname":platform.node(),"platform":platform.platform(),"python":sys.version,"git_branch":subprocess.check_output(["git","-C",str(PROJECT),"branch","--show-current"],text=True).strip(),"git_commit":subprocess.check_output(["git","-C",str(PROJECT),"rev-parse","HEAD"],text=True).strip(),"git_status":subprocess.check_output(["git","-C",str(PROJECT),"status","--short"],text=True).strip(),"nvidia_smi":subprocess.check_output(["nvidia-smi","-L"],text=True).strip(),"generation_workers_per_gpu":4,"relax_workers_per_gpu":2,"checkpoint_sha256":"01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e","mattersim_checkpoint_sha256":"e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5","FORMAL_30000_SEEDS_STARTED":False}; jwrite(FINAL/"environment_manifest.json",env); text(FINAL/"reproduction_commands.md","# Reproduction commands\n\n```bash\nbash /data/dxl/reports/budget_aware_gating/status_budget_aware.sh\nbash /data/dxl/reports/budget_aware_gating/resume_budget_aware.sh\n```\n\nCompleted tasks are strictly validated and skipped.\n")
    table=pd.DataFrame([x for x in pareto if x["experiment_stage"].startswith("new_")]).to_markdown(index=False); lines=["# Budget-Aware Convergence-Guided Corrector Scheduling","",f"- Development validated: **{validated}**","- Development seeds: 14000–14063 (paired n=64)","- STABILITY_SOURCE=MatterSim-5M surrogate","- DFT_VERIFIED=False","- PROPERTY_TARGET_VERIFIED=False","- FORMAL_30000_SEEDS_STARTED=False","","## New development Pareto results","",table,"","## Candidate decisions","",*[f"- {n}: GO={x['go']}; gates={json.dumps(x['gates'],ensure_ascii=False)}" for n,x in decisions.items()],"","## Old G3 formal reference","","G3 uses independent old formal seeds and is excluded from new paired statistics."]; text(FINAL/"budget_aware_final_report.md","\n".join(lines)+"\n"); figures(pareto,records,grows)
def on_signal(signum,_frame):
    core.stop_requested.set(); text(STOP,f"signal={signum} time={now()}\n"); stage(None,None,"Validated SIGINT stop requested.",{"stop_requested":True})
    with core.children_lock: children=list(core.children.values())
    for p in children:
        try: os.killpg(p.pid,signal.SIGINT)
        except (ProcessLookupError,PermissionError): pass
def run():
    signal.signal(signal.SIGINT,on_signal); signal.signal(signal.SIGTERM,on_signal)
    if STOP.exists(): STOP.unlink()
    stage(None,None,"32/64-seed launcher started.",{"launcher_pid":os.getpid(),"launcher_pgid":os.getpgid(0),"stop_requested":False,"formal_30000_seeds_started":False}); smoke=read(SMOKE); passed8=[n for n in ("G1","G2") if smoke["candidate_decisions"][n]["go"]]
    if not passed8: stage("thirty_two_generation","not_applicable","No candidate passed 8-seed gates.",{}); return 2
    names=["A0",*passed8]; seeds32=list(range(14000,14032)); timing32=DEV_REPORT/"screen_wave_timings.json"
    try:
        freeze_mattersim(); stage("thirty_two_generation","running",f"4 workers/GPU; configs={names}; seeds=14000-14031.",{}); run_generation(names,seeds32,timing32); snapshot("thirty_two_generation_tasks.json",names,seeds32); stage("thirty_two_generation","success",f"{len(names)*32}/{len(names)*32} valid outputs.",{})
        configure(names,seeds32,timing32); stage("thirty_two_relax","running","MatterSim-5M; 2 persistent workers/GPU.",{}); run_relax(); stage("thirty_two_relax","success",f"{len(names)*32}/{len(names)*32} valid relax outputs.",{})
        stage("thirty_two_metrics","running","Official TRI2024 metrics and paired statistics.",{}); sums32,dec32,*_=evaluate("screen_32",names,seeds32,timing32); stage("thirty_two_metrics","success","32-seed official metrics complete.",{}); passed32=[n for n in passed8 if dec32[n]["go"]][:2]
        stage("pareto_candidate_freeze","running",f"Freezing candidates={passed32}; no later retuning.",{}); freeze(passed32,dec32); stage("pareto_candidate_freeze","success",f"Frozen exact candidates={passed32}.",{})
        if not passed32: stage("sixty_four_generation","not_applicable","No 32-seed candidate passed; stopped without retuning.",{"budget_aware_gating_development_validated":False}); return 2
        names64=["A0",*passed32]; seeds64=list(range(14000,14064)); timing64=DEV_REPORT/"extension_wave_timings.json"; stage("sixty_four_generation","running",f"Only supplementing 14032-14063; configs={names64}.",{}); run_generation(names64,seeds64,timing64); snapshot("sixty_four_generation_tasks.json",names64,seeds64); stage("sixty_four_generation","success","64-seed set complete; 14000-14031 reused.",{})
        configure(names64,seeds64,timing64); stage("sixty_four_relax","running","Only missing 14032-14063 relaxations are claimed.",{}); run_relax(); stage("sixty_four_relax","success","64-seed MatterSim outputs complete.",{}); stage("sixty_four_metrics","running","Official 64-seed development metrics.",{}); sums,decisions,records,stats,failures,_=evaluate("validation_64",names64,seeds64,timing64); stage("sixty_four_metrics","success","64-seed official metrics complete.",{}); stage("paired_analysis","running","New same-seed paired statistics; old G3 separate.",{}); stage("paired_analysis","success","Paired n=64 comparisons complete.",{}); stage("pareto_report","running","Writing final tables and figures.",{}); final(names64,sums,decisions,records,stats,failures); validated=any(x["go"] for x in decisions.values()); stage("pareto_report","success",f"BUDGET_AWARE_GATING_DEVELOPMENT_VALIDATED={validated}",{"budget_aware_gating_completed":True,"budget_aware_gating_development_validated":validated,"formal_30000_seeds_started":False}); return 0 if validated else 2
    except KeyboardInterrupt: stage(None,None,"Interrupted safely; complete tasks will be skipped on resume.",{"overall_status":"interrupted","stop_requested":True}); return 130
    except BaseException:
        text(DEV_REPORT/"runner_error.log",traceback.format_exc()); current=read(RESULT/"progress/master_progress.json")["current_stage"]; stage(current,"failed","See development/runner_error.log.",{"overall_status":"failed"}); raise
def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",nargs="?",default="main",choices=("main","relax-worker")); p.add_argument("--gpu",type=int); p.add_argument("--slot",type=int); a=p.parse_args()
    if a.mode=="relax-worker":
        smoke=read(SMOKE); passed=[n for n in ("G1","G2") if smoke["candidate_decisions"][n]["go"]]; frozen=DEV_REPORT/"frozen_pareto_candidates.json"; names=["A0",*(read(frozen)["candidates"].keys() if frozen.exists() else passed)]; seeds=list(range(14000,14064 if frozen.exists() else 14032)); configure(names,seeds,DEV_REPORT/"worker_unused_timing.json"); return core.relax_worker(int(a.gpu),int(a.slot))
    return run()
if __name__=="__main__": raise SystemExit(main())
