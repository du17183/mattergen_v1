#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,platform,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
ROOT=Path('/data/dxl'); RESULT=ROOT/'results/budget_aware_gating'; REPORT=ROOT/'reports/budget_aware_gating'; FINAL=REPORT/'final'; FIG=FINAL/'figures'; SCREEN=REPORT/'development/screen_32'; PROJECT=ROOT/'mattergen_v1'; TZ=ZoneInfo('Asia/Shanghai')
def now(): return datetime.now(TZ).isoformat(timespec='seconds')
def read(p): return json.loads(p.read_text())
def clean(v):
 if isinstance(v,np.bool_): return bool(v)
 if isinstance(v,np.integer): return int(v)
 if isinstance(v,(np.floating,float)): return None if not np.isfinite(v) else float(v)
 if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
 return v
def text(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(f'.{p.name}.tmp.{os.getpid()}'); t.write_text(v); os.replace(t,p)
def js(p,v): text(p,json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(f'.{p.name}.tmp.{os.getpid()}'); pd.DataFrame(rows).to_csv(t,index=False); os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
summary=read(SCREEN/'quality_speed_summary.json')['configs']; decisions=read(SCREEN/'candidate_decisions.json')['decisions']; formal=read(REPORT/'frozen_formal_baseline/formal_final_report.json'); formal_i2=formal['innovation2']; formal_quality={row['method']:row for row in pd.read_csv(REPORT/'frozen_formal_baseline/formal_comparison.csv').to_dict('records')}; smoke=read(REPORT/'eight_seed_go_no_go.json')
rows=[]
for name in ('A0','G1','G2'): rows.append({**summary[name],'experiment_stage':'new_development_screen_14000_14031','paired_with_A0':True,'screen_decision':'baseline' if name=='A0' else ('GO' if decisions[name]['go'] else 'NO_GO')})
gq=formal_quality['G3']; rows.append({'config':'G3','experiment_stage':'old_formal_256_seeds_20000_20255','paired_with_A0':False,'screen_decision':'formal_reference_only','formal_seed_count':formal['FORMAL_SEED_COUNT'],'formal_innovation2_confirmed':formal_i2['FORMAL_INNOVATION2_CONFIRMED'],'speed_multiplier':formal_i2['G3_SPEED_MULTIPLIER'],'throughput_multiplier':1+formal_i2['G3_THROUGHPUT_GAIN'],'physical_forward_reduction':formal_i2['G3_PHYSICAL_FORWARD_REDUCTION'],'corrector_skip_rate':formal_i2['G3_CORRECTOR_SKIP_RATE'],'avg_energy_above_hull_per_atom':gq['avg_energy_above_hull_per_atom'],'frac_stable_structures':gq['frac_stable_structures'],'frac_novel_unique_stable_structures':gq['frac_novel_unique_stable_structures'],'generation_composition_validity':gq['generation_composition_validity'],'avg_structure_validity':gq['avg_structure_validity'],'precision':gq['precision'],'recall':gq['recall'],'Ehull_change_ev_atom':formal_i2['EHULL_CHANGE'],'stable_change':formal_i2['STABLE_RATE_CHANGE'],'NUS_change':formal_i2['NUS_RATE_CHANGE'],'warning':'Frozen 256-seed formal reference uses seeds 20000-20255; excluded from new 32-seed paired statistics.'})
fail=pd.read_csv(SCREEN/'failure_cases.csv').to_dict('records'); strict_gen=sum(1 for x in read(RESULT/'development/progress/generation_progress.json')['tasks'] if x['status']=='failed'); strict_relax=sum(1 for x in read(RESULT/'development/progress/relax_progress.json')['tasks'] if x['status']=='failed')
payload={'BUDGET_AWARE_GATING_COMPLETED':True,'BUDGET_AWARE_GATING_DEVELOPMENT_VALIDATED':False,'reason':'No candidate passed every frozen 32-seed Pareto gate; stopped before 64-seed without retuning.','DEVELOPMENT_SEEDS':[14000,14031],'PAIRED_SEEDS':32,'A0_RESULTS':summary['A0'],'G1_RESULTS':summary['G1'],'G2_RESULTS':summary['G2'],'G3_FORMAL_REFERENCE':rows[-1],'G1_PHYSICAL_FORWARD_REDUCTION':summary['G1']['physical_forward_reduction'],'G1_SPEED_MULTIPLIER':summary['G1']['speed_multiplier'],'G1_EHULL_CHANGE':summary['G1']['Ehull_change_ev_atom'],'G1_STABLE_CHANGE':summary['G1']['stable_change'],'G1_NUS_CHANGE':summary['G1']['NUS_change'],'G2_PHYSICAL_FORWARD_REDUCTION':summary['G2']['physical_forward_reduction'],'G2_SPEED_MULTIPLIER':summary['G2']['speed_multiplier'],'G2_EHULL_CHANGE':summary['G2']['Ehull_change_ev_atom'],'G2_STABLE_CHANGE':summary['G2']['stable_change'],'G2_NUS_CHANGE':summary['G2']['NUS_change'],'BEST_CONSERVATIVE_CONFIG':None,'BEST_MODERATE_CONFIG':None,'PARETO_FRONTIER':['A0','G1_rejected','G2_rejected','G3_old_formal_reference'],'CANDIDATE_DECISIONS':decisions,'DETERMINISM_PASSED':all(smoke['aggregate'][x]['determinism_level1'] for x in ('A0','G1','G2')),'GENERATION_FAILURES':strict_gen,'RELAX_FAILURES':strict_relax,'VALID_MAX_STEP_NONCONVERGENCES':len(fail),'FROZEN_CONFIGS':read(REPORT/'development/frozen_pareto_candidates.json'),'FORMAL_30000_SEEDS_STARTED':False,'FORMAL_14032_14063_STARTED':False,'STABILITY_SOURCE':'MatterSim-5M surrogate','DFT_VERIFIED':False,'PROPERTY_TARGET_VERIFIED':False,'limitations':['32-seed development screen only; hard-gate failure prevented 64-seed extension.','MatterSim-5M stability is a surrogate and is not DFT verification.','Old G3 formal results use a different seed set and are not included in new paired tests.','Observed percentage-point differences at n=32 are quantized in 3.125 pp increments.']}
FINAL.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); js(FINAL/'budget_aware_final_report.json',payload); csv(FINAL/'pareto_comparison.csv',rows); js(FINAL/'pareto_comparison.json',{'rows':rows,'warning':'G3 uses different formal seeds.'}); shutil.copy2(SCREEN/'paired_seed_records.csv',FINAL/'paired_seed_records.csv'); shutil.copy2(SCREEN/'paired_statistics.csv',FINAL/'paired_statistics.csv'); shutil.copy2(SCREEN/'failure_cases.csv',FINAL/'failure_cases.csv'); shutil.copy2(REPORT/'development/frozen_pareto_candidates.json',FINAL/'frozen_candidate_configs.json'); shutil.copy2(SCREEN/'generation_manifest.csv',FINAL/'generation_manifest.csv'); csv(FINAL/'relax_manifest.csv',read(RESULT/'development/progress/relax_progress.json')['tasks']); shutil.copy2(SCREEN/'candidate_decisions.json',FINAL/'candidate_decisions.json')
seedrows=[{'seed':s,'role':'development_screen' if s<=14031 else 'planned_extension_not_started_due_to_32_seed_no_go','generated':s<=14031,'formal':False} for s in range(14000,14064)]; csv(FINAL/'seed_manifest.csv',seedrows)
env={'generated_at':now(),'hostname':platform.node(),'platform':platform.platform(),'python':sys.version,'git_branch':subprocess.check_output(['git','-C',str(PROJECT),'branch','--show-current'],text=True).strip(),'git_commit':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),'git_status':subprocess.check_output(['git','-C',str(PROJECT),'status','--short'],text=True).strip(),'nvidia_smi':subprocess.check_output(['nvidia-smi','-L'],text=True).strip(),'generation_workers_per_gpu':4,'relax_workers_per_gpu':2,'checkpoint_sha256':'01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e','mattersim_checkpoint_sha256':'e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5','FORMAL_30000_SEEDS_STARTED':False}; js(FINAL/'environment_manifest.json',env)
text(FINAL/'reproduction_commands.md','# Reproduction commands\n\n```bash\nbash /data/dxl/reports/budget_aware_gating/status_budget_aware.sh\nbash /data/dxl/reports/budget_aware_gating/resume_budget_aware.sh\n```\n\nThe frozen 32-seed No-Go is terminal for this development round. Resume validates existing outputs and does not start 64-seed candidates because none were frozen.\n')
lines=['# Budget-Aware Convergence-Guided Corrector Scheduling — final development report','', '- BUDGET_AWARE_GATING_COMPLETED=True','- BUDGET_AWARE_GATING_DEVELOPMENT_VALIDATED=False','- Development seeds: 14000–14031; paired n=32','- 14032–14063: not started after frozen 32-seed No-Go','- FORMAL_30000_SEEDS_STARTED=False','- STABILITY_SOURCE=MatterSim-5M surrogate','- DFT_VERIFIED=False','- PROPERTY_TARGET_VERIFIED=False','','## Outcome','',payload['reason'],'','## Quality-speed table','',pd.DataFrame(rows[:3]).to_markdown(index=False),'','## Hard gates','',*[f"- {n}: GO={d['go']}; {json.dumps(d['gates'],ensure_ascii=False)}" for n,d in decisions.items()],'','## Interpretation','','G1 delivered its intended conservative speed/compute reduction and improved mean E-hull, but missed the frozen stable and NUS tolerances. G2 delivered the intended compute reduction but missed median speed, stable, and E-hull thresholds. No post-hoc retuning was performed.','', 'The old G3 formal reference is displayed separately and is never mixed into the new paired statistics.']; text(FINAL/'budget_aware_final_report.md','\n'.join(lines)+'\n')
# Figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
for filename,x,y,xlabel,ylabel in [('speed_vs_ehull.png','speed_multiplier','avg_energy_above_hull_per_atom','Speed multiplier','Average E-hull (eV/atom)'),('speed_vs_stable.png','speed_multiplier','frac_stable_structures','Speed multiplier','Stable fraction'),('speed_vs_nus.png','speed_multiplier','frac_novel_unique_stable_structures','Speed multiplier','NUS fraction'),('forward_reduction_vs_quality.png','physical_forward_reduction','avg_energy_above_hull_per_atom','Physical forward reduction','Average E-hull'),('skip_ratio_vs_quality.png','corrector_skip_rate','frac_stable_structures','Corrector skip ratio','Stable fraction')]:
 fig,ax=plt.subplots(figsize=(6,4))
 for r in rows: ax.scatter(r.get(x,0),r.get(y,np.nan),s=70); ax.annotate(r['config'],(r.get(x,0),r.get(y,np.nan)))
 ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/filename,dpi=180); plt.close(fig)
gen=read(RESULT/'development/progress/generation_progress.json')['tasks']; by={n:[] for n in ('A0','G1','G2')}; veto={n:[] for n in by}
for task in gen:
 if task['config'] not in by: continue
 c=read(Path(task['output_dir'])/'corrector_summary.json'); by[task['config']].append(c.get('corrector_budget_used',0)/1000); veto[task['config']].append(c.get('corrector_atomic_veto_count',0)/1000)
for filename,data,title,ylabel in [('budget_used_distribution.png',by,'Skip-budget utilization distribution','Budget used / 1000 steps'),('atomic_veto_rate.png',veto,'Atomic-number veto rate','Veto count / 1000 steps')]:
 fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot([data[n] for n in data],labels=list(data)); ax.set(title=title,ylabel=ylabel); fig.tight_layout(); fig.savefig(FIG/filename,dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4)); names=['A0','G1','G2']; x=np.arange(3); width=.24
for i,key in enumerate(('calibration_rate','fallback_rate','rescue_rate')): ax.bar(x+(i-1)*width,[summary[n][key] for n in names],width,label=key)
ax.set_xticks(x,names); ax.legend(); ax.set_title('Calibration / fallback / rescue rates'); fig.tight_layout(); fig.savefig(FIG/'calibration_fallback_rescue.png',dpi=180); plt.close(fig)
paired=pd.read_csv(FINAL/'paired_seed_records.csv'); values=[]
for col in paired.columns:
 if col=='speed_multiplier_left_over_right': values.extend(pd.to_numeric(paired[col],errors='coerce').dropna().tolist())
fig,ax=plt.subplots(figsize=(6,4)); ax.hist(values,bins=16); ax.axvline(1,color='black',ls='--'); ax.set(xlabel='Paired speed multiplier',ylabel='Count'); fig.tight_layout(); fig.savefig(FIG/'paired_speedup.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
for r in rows: ax.scatter(r['speed_multiplier'],r['avg_energy_above_hull_per_atom'],s=90); ax.annotate(f"{r['config']} ({r['experiment_stage']})",(r['speed_multiplier'],r['avg_energy_above_hull_per_atom']),fontsize=7)
ax.set(xlabel='Speed multiplier',ylabel='Average E-hull',title='Quality-speed Pareto view'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/'pareto_frontier.png',dpi=180); plt.close(fig)
# Integrity manifest
files=[p for p in sorted(FINAL.rglob('*')) if p.is_file() and p.name!='sha256_manifest.txt']; text(FINAL/'sha256_manifest.txt',''.join(f'{sha(p)}  {p.relative_to(FINAL)}\n' for p in files))
print(json.dumps({'final':str(FINAL/'budget_aware_final_report.json'),'figures':len(list(FIG.glob('*.png'))),'validated':False},indent=2))
