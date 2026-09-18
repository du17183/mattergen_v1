"""Frozen field-attribution, selection gates, paired inference, and plots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from itertools import combinations

from ase.io import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.field_decoupled_adaptive_cfg.run_generation import calibration_policies, phase_policies


PROJECT=Path(__file__).resolve().parents[2]; ROOT=PROJECT/"experiments/field_decoupled_adaptive_cfg"; PLOTS=ROOT/"plots"
EPS=1e-12; N_BOOT=20_000; BOOT_SEEDS={"calibration":2026091605,"p0":2026091606,"formal256":2026091607}
FIELD_CANDIDATES=("A19","P19","C19","AP19","AC19","PC19","APulse","PPulse","CPulse")
SUBSETS={frozenset():"C0",frozenset("A"):"A19",frozenset("P"):"P19",frozenset("C"):"C19",frozenset("AP"):"AP19",frozenset("AC"):"AC19",frozenset("PC"):"PC19",frozenset("APC"):"G19"}


def json_write(path:Path,payload:object)->None: path.write_text(json.dumps(payload,indent=2,default=float)+"\n")


def quality_enriched(phase:str)->pd.DataFrame:
    phase_root=ROOT/phase; values=pd.read_csv(phase_root/"property_metrics_raw.csv")
    for column,default in (("e_hull",np.nan),("stable",False),("nus",False),("novel",False),("unique",False),("quality_success",False)): values[column]=default
    for policy in phase_policies(phase):
        group=policy["policy_id"]; structure_path=phase_root/"evaluation_structures"/f"{group}.extxyz"; quality_root=phase_root/"quality"/group
        if not structure_path.exists(): continue
        if not (quality_root/"official_detailed.json.gz").exists(): raise FileNotFoundError(f"missing quality {phase}/{group}")
        atoms=read(structure_path,index=":"); per=pd.read_csv(quality_root/"per_structure.csv")
        with gzip.open(quality_root/"official_detailed.json.gz","rt") as stream: detailed=json.load(stream)
        if len(per)!=len(detailed["energy_above_hull_per_atom"]): raise RuntimeError(f"quality length mismatch {group}")
        for output_index,row in per.reset_index(drop=True).iterrows():
            identifier=str(atoms[int(row["index"])].info["branch_id"]); match=values.index[values["branch_id"]==identifier]
            if len(match)!=1: raise RuntimeError(f"branch lookup failed {identifier}")
            target=match[0]; values.loc[target,"e_hull"]=float(detailed["energy_above_hull_per_atom"][output_index]); values.loc[target,"stable"]=bool(detailed["stable"][output_index]); values.loc[target,"nus"]=bool(detailed["novel_unique_stable"][output_index]); values.loc[target,"novel"]=bool(detailed["novel"][output_index]); values.loc[target,"unique"]=bool(detailed["unique"][output_index]); values.loc[target,"quality_success"]=True
    for column in ("structure_valid","stable","nus","novel","unique","quality_success"): values[column]=values[column].fillna(False).astype(bool)
    values["evaluation_valid"]=values["structure_valid"]&values["quality_success"]; values.to_csv(phase_root/"evaluated_results.csv",index=False); return values


def build_paired(values:pd.DataFrame)->pd.DataFrame:
    base=values[values.policy_id=="C0"].set_index("seed")
    if len(base)!=values.seed.nunique(): raise RuntimeError("C0 incomplete")
    rows=[]
    for _,row in values.iterrows():
        c0=base.loc[int(row.seed)]; delta=float(row.property_absolute_error-c0.property_absolute_error); e_delta=float(row.e_hull-c0.e_hull) if np.isfinite(row.e_hull) and np.isfinite(c0.e_hull) else float("nan")
        rows.append({**row.to_dict(),"c0_property_absolute_error":float(c0.property_absolute_error),"property_delta_vs_c0":delta,"property_gain_vs_c0":-delta,"property_harm_vs_c0":bool(row.property_absolute_error>1.05*c0.property_absolute_error),"positive_property_degradation":max(delta,0.0),"e_hull_delta_vs_c0":e_delta,"e_hull_harm_vs_c0":bool(np.isfinite(e_delta) and e_delta>0.02),"stable_loss_vs_c0":bool(c0.stable and not row.stable),"nus_loss_vs_c0":bool(c0.nus and not row.nus),"validity_loss_vs_c0":bool(c0.evaluation_valid and not row.evaluation_valid)})
    return pd.DataFrame(rows)


def summarize(paired:pd.DataFrame)->pd.DataFrame:
    c0=paired[paired.policy_id=="C0"]; c0_mean=float(c0.property_absolute_error.mean()); c0_median=float(c0.property_absolute_error.median()); c0_e=float(c0.e_hull.mean()); c0_rates={m:float(c0[m].mean()) for m in ("stable","nus","evaluation_valid","novel","unique")}; rows=[]
    for identifier,frame in paired.groupby("policy_id",sort=True):
        gain=frame.property_gain_vs_c0.to_numpy(float); degradation=frame.positive_property_degradation.to_numpy(float); n=len(frame); worst=max(1,int(math.ceil(.25*n))); first=frame.iloc[0]
        stable_drop=c0_rates["stable"]-float(frame.stable.mean()); nus_drop=c0_rates["nus"]-float(frame.nus.mean()); validity_drop=c0_rates["evaluation_valid"]-float(frame.evaluation_valid.mean()); e_worse=float(frame.e_hull.mean())-c0_e
        margins=[(.05-stable_drop)/.05,(.05-nus_drop)/.05,(.05-validity_drop)/.05,(.01-e_worse)/.01]; guard=bool(stable_drop<=.05+EPS and nus_drop<=.05+EPS and validity_drop<=.05+EPS and np.isfinite(e_worse) and e_worse<=.01+EPS)
        modified=sum(abs(float(first[f"g_{field}"])-2.0)>EPS for field in ("atomic","pos","cell")); complexity=modified+(1 if first.policy_kind=="pulse" else 0)
        rows.append({"policy_id":identifier,"policy_kind":first.policy_kind,"g_atomic":float(first.g_atomic),"g_pos":float(first.g_pos),"g_cell":float(first.g_cell),"start":int(first.start),"duration":int(first.duration),"policy_complexity":complexity,"n":n,"property_mae_mean":float(frame.property_absolute_error.mean()),"property_mae_median":float(frame.property_absolute_error.median()),"mean_property_gain":float(gain.mean()),"median_property_gain":float(np.median(gain)),"relative_mean_gain":float(gain.mean())/max(c0_mean,EPS),"relative_median_gain":float(np.median(gain))/max(c0_median,EPS),"wins":int((gain>EPS).sum()),"ties":int((np.abs(gain)<=EPS).sum()),"losses":int((gain<-EPS).sum()),"property_harm_rate":float(frame.property_harm_vs_c0.mean()),"p75_positive_degradation":float(np.quantile(degradation,.75)),"p90_positive_degradation":float(np.quantile(degradation,.90)),"worst_quartile_mean_degradation":float(np.sort(degradation)[-worst:].mean()),"e_hull_mean":float(frame.e_hull.mean()),"stable_rate":float(frame.stable.mean()),"nus_rate":float(frame.nus.mean()),"validity_rate":float(frame.evaluation_valid.mean()),"novel_rate":float(frame.novel.mean()),"unique_rate":float(frame.unique.mean()),"stable_drop":stable_drop,"nus_drop":nus_drop,"validity_drop":validity_drop,"e_hull_worsening":e_worse,"guardrail_margin":float(min(margins)),"guardrail_pass":guard,"e_hull_harm_rate":float(frame.e_hull_harm_vs_c0.mean()),"stable_loss_rate":float(frame.stable_loss_vs_c0.mean()),"nus_loss_rate":float(frame.nus_loss_vs_c0.mean()),"validity_loss_rate":float(frame.validity_loss_vs_c0.mean())})
    result=pd.DataFrame(rows); global_gain=float(result.loc[result.policy_id=="G19","mean_property_gain"].iloc[0]); result["global_gain"]=global_gain; result["benefit_retention"]=result.mean_property_gain/global_gain if global_gain>0 else np.nan; return result


def bootstrap(values:np.ndarray,seed:int)->dict:
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values): return {"n":0,"mean":float("nan"),"ci95_low":float("nan"),"ci95_high":float("nan")}
    rng=np.random.default_rng(seed); boot=values[rng.integers(0,len(values),size=(N_BOOT,len(values)))].mean(axis=1); return {"n":len(values),"mean":float(values.mean()),"ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),"resamples":N_BOOT}


def pair(paired:pd.DataFrame,left:str,right:str,column:str)->np.ndarray:
    pivot=paired.pivot(index="seed",columns="policy_id",values=column); return pivot[left].to_numpy(float)-pivot[right].to_numpy(float)


def row(summary:pd.DataFrame,identifier:str)->pd.Series:
    match=summary[summary.policy_id==identifier]
    if len(match)!=1: raise RuntimeError(identifier)
    return match.iloc[0]


def physical_point_improvements(left:pd.Series,right:pd.Series)->list[str]:
    improved=[]
    if float(left.e_hull_mean)<float(right.e_hull_mean)-EPS: improved.append("e_hull")
    for metric in ("stable","nus","validity"):
        if float(left[f"{metric}_rate"])>float(right[f"{metric}_rate"])+EPS: improved.append(metric)
    return improved


def physical_bootstrap(paired:pd.DataFrame,left:str,right:str,seed:int)->dict:
    results={"e_hull_reduction":bootstrap(-pair(paired,left,right,"e_hull"),seed)}
    for offset,metric in enumerate(("stable","nus","evaluation_valid"),1): results[f"{metric}_gain"]=bootstrap(pair(paired,left,right,metric),seed+offset)
    results["favorable_significant_count"]=sum(float(value["ci95_low"])>0 for key,value in results.items() if key!="favorable_significant_count")
    return results


def shapley_arrays(paired:pd.DataFrame,value_column:str,mode:str)->dict[str,np.ndarray]:
    pivot=paired.pivot(index="seed",columns="policy_id",values=value_column); base=pivot["C0"].to_numpy(float); values={}
    for subset,identifier in SUBSETS.items():
        current=pivot[identifier].to_numpy(float)
        if mode=="lower_benefit": values[subset]=base-current
        elif mode=="higher_harm": values[subset]=base-current
        elif mode=="lower_harm": values[subset]=current-base
        else: raise ValueError(mode)
    output={}
    universe=set("APC")
    for field in "APC":
        contribution=np.zeros(len(base),float)
        for subset in (set(x) for size in range(3) for x in combinations(universe-{field},size)):
            weight=math.factorial(len(subset))*math.factorial(2-len(subset))/math.factorial(3); frozen=frozenset(subset); contribution+=weight*(values[frozen|{field}]-values[frozen])
        output[field]=contribution
    return output


def label_from_contributions(values:dict[str,float])->tuple[str,dict[str,float]]:
    positive={field:max(0.0,float(value)) for field,value in values.items()}; total=sum(positive.values()); shares={field:(value/total if total>0 else 0.0) for field,value in positive.items()}
    if total<=0: return "NONE",shares
    leader=max(shares,key=shares.get)
    if shares[leader]>=.60: return {"A":"ATOMIC","P":"POS","C":"CELL"}[leader],shares
    if sum(value>0 for value in positive.values())>=2: return "MULTI",shares
    return "NONE",shares


def mechanism(paired:pd.DataFrame,summary:pd.DataFrame)->tuple[str,str,pd.DataFrame]:
    records=[]; prop_arrays=shapley_arrays(paired,"property_absolute_error","lower_benefit"); prop_means={field:float(values.mean()) for field,values in prop_arrays.items()}; property_label,prop_shares=label_from_contributions(prop_means)
    for field in "APC": records.append({"target":"property_benefit","field":field,"shapley_mean":prop_means[field],"positive_share":prop_shares[field]})
    quality_shares=[]
    for metric,mode in (("stable","higher_harm"),("nus","higher_harm"),("evaluation_valid","higher_harm"),("e_hull","lower_harm")):
        arrays=shapley_arrays(paired,metric,mode); means={field:float(np.nanmean(values)) for field,values in arrays.items()}; global_harm={"stable":float(row(summary,"G19").stable_drop),"nus":float(row(summary,"G19").nus_drop),"evaluation_valid":float(row(summary,"G19").validity_drop),"e_hull":float(row(summary,"G19").e_hull_worsening)}[metric]
        _,shares=label_from_contributions(means)
        for field in "APC": records.append({"target":f"{metric}_harm","field":field,"shapley_mean":means[field],"positive_share":shares[field]})
        if global_harm>0: quality_shares.append(shares)
    average={field:float(np.mean([shares[field] for shares in quality_shares])) if quality_shares else 0.0 for field in "APC"}; quality_label,_=label_from_contributions(average)
    for field in "APC": records.append({"target":"quality_harm_aggregate","field":field,"shapley_mean":float("nan"),"positive_share":average[field]})
    return property_label,quality_label,pd.DataFrame(records)


def pareto(summary:pd.DataFrame)->pd.DataFrame:
    candidates=summary[summary.policy_id.isin(FIELD_CANDIDATES)].copy(); objectives=[("mean_property_gain",1),("stable_rate",1),("nus_rate",1),("validity_rate",1),("e_hull_mean",-1),("property_harm_rate",-1)]; keep=[]
    for i,left in candidates.iterrows():
        dominated=False
        for j,right in candidates.iterrows():
            if i==j: continue
            weak=all(direction*float(right[name])>=direction*float(left[name])-EPS for name,direction in objectives); strict=any(direction*float(right[name])>direction*float(left[name])+EPS for name,direction in objectives)
            if weak and strict: dominated=True; break
        keep.append(not dominated)
    candidates["pareto_front"]=keep; return candidates[candidates.pareto_front].copy()


def write_metrics(phase:str,paired:pd.DataFrame,summary:pd.DataFrame)->None:
    root=ROOT/phase; paired.to_csv(root/"paired_results.csv",index=False)
    summary[["policy_id","n","property_mae_mean","property_mae_median","mean_property_gain","median_property_gain","relative_mean_gain","relative_median_gain","global_gain","benefit_retention","wins","ties","losses"]].to_csv(root/("property_metrics.csv" if phase=="calibration" else "metrics.csv"),index=False)
    summary[["policy_id","n","e_hull_mean","stable_rate","nus_rate","validity_rate","novel_rate","unique_rate","stable_drop","nus_drop","validity_drop","e_hull_worsening","guardrail_margin","guardrail_pass"]].to_csv(root/"quality_metrics.csv",index=False)
    summary[["policy_id","n","property_harm_rate","e_hull_harm_rate","stable_loss_rate","nus_loss_rate","validity_loss_rate","p75_positive_degradation","p90_positive_degradation","worst_quartile_mean_degradation"]].to_csv(root/"harm_metrics.csv",index=False)


def calibration_gate(paired:pd.DataFrame,summary:pd.DataFrame)->dict:
    property_field,quality_field,mechanism_frame=mechanism(paired,summary); mechanism_frame.to_csv(ROOT/"calibration/field_mechanism.csv",index=False); global_gain=float(row(summary,"G19").mean_property_gain)
    if global_gain <= 0:
        property_field = "NONE"
    ranking=[]
    for identifier in FIELD_CANDIDATES:
        candidate=row(summary,identifier); better_g19=physical_point_improvements(candidate,row(summary,"G19")); better_pulse=physical_point_improvements(candidate,row(summary,"GPulse")); base_signal=global_gain>0
        eligible=bool(base_signal and candidate.guardrail_pass and candidate.relative_mean_gain>=.05-EPS and int(candidate.wins)>int(candidate.losses) and candidate.benefit_retention>=.70-EPS and (len(better_g19)>=2 or len(better_pulse)>=2))
        ranking.append({**candidate.to_dict(),"quality_improvements_vs_G19":";".join(better_g19),"quality_improvements_vs_GPulse":";".join(better_pulse),"tradeoff_baseline_pass":len(better_g19)>=2 or len(better_pulse)>=2,"calibration_eligible":eligible})
    ranked=pd.DataFrame(ranking).sort_values(["calibration_eligible","guardrail_margin","benefit_retention","property_harm_rate","policy_complexity","policy_id"],ascending=[False,False,False,True,True,True]); ranked.to_csv(ROOT/"calibration/field_policy_ranking.csv",index=False); pareto(summary).to_csv(ROOT/"calibration/pareto_front.csv",index=False)
    eligible=ranked[ranked.calibration_eligible]; selected=eligible.iloc[0] if len(eligible) else None
    selection={"FIELD_DECOUPLING_CALIBRATION":"GO" if selected is not None else "FAIL","global_gain":global_gain,"calibration_base_signal":"PASS" if global_gain>0 else "FAIL","eligible_count":len(eligible),"selected_policy_id":None if selected is None else str(selected.policy_id),"PROPERTY_FIELD":property_field,"QUALITY_HARM_FIELD":quality_field,"selection_order":["guardrail_margin","benefit_retention","property_harm_rate","policy_complexity"]}; json_write(ROOT/"calibration/selected_policy.json",selection)
    field_stage="NOT_RUN"
    if selected is not None:
        policy=next(policy for policy in calibration_policies() if policy["policy_id"]==selected.policy_id); constant_map={"APulse":"A19","PPulse":"P19","CPulse":"C19"}
        if selected.policy_id in constant_map:
            constant=row(summary,constant_map[str(selected.policy_id)]); field_stage="SUPPORTED" if (not constant.guardrail_pass or len(physical_point_improvements(selected,constant))>=2) else "NOT_SUPPORTED"
        frozen={"schema_version":1,"frozen_after_calibration_before_p0":True,**policy,"property_field":property_field,"quality_harm_field":quality_field,"selection_rule":"guardrails, property, retention, then lexicographic priorities"}; path=ROOT/"implementation/frozen_field_cfg.yaml"; path.write_text(yaml.safe_dump(frozen,sort_keys=False)); digest=hashlib.sha256(path.read_bytes()).hexdigest(); json_write(ROOT/"implementation/frozen_field_cfg_manifest.json",{"schema_version":1,"frozen_before_p0":True,"frozen_field_cfg_sha256":digest,"selected_policy_id":str(selected.policy_id)})
    decision={"FIELD_ATTRIBUTION_DIAGNOSTIC":"COMPLETE","FIELD_DECOUPLING_CALIBRATION":"GO" if selected is not None else "FAIL","PROPERTY_FIELD":property_field,"QUALITY_HARM_FIELD":quality_field,"FIELD_CFG_P0":"NOT_RUN","FIELD_DECOUPLED_CFG_FORMAL256":"NOT_RUN","FIELD_DECOUPLING_EFFECT":"SUPPORTED" if selected is not None else "NOT_SUPPORTED","FIELD_STAGE_EFFECT":field_stage,"INNOVATION1_FINAL_STATUS":"MIXED","calibration_n":int(paired.seed.nunique()),"eligible_policy_count":len(eligible),"selected_policy_id":None if selected is None else str(selected.policy_id),"stop_reason":None if selected is not None else ("CALIBRATION_BASE_SIGNAL_FAIL" if global_gain<=0 else "No field policy passed every frozen property, retention, physical-quality, and baseline gate.")}
    plot_calibration(summary, None if selected is None else str(selected.policy_id)); return decision


def fresh_gate(phase:str,paired:pd.DataFrame,summary:pd.DataFrame)->dict:
    frozen=yaml.safe_load((ROOT/"implementation/frozen_field_cfg.yaml").read_text()); s0_id=str(frozen["policy_id"]); s0=row(summary,s0_id); g19=row(summary,"G19"); gpulse=row(summary,"GPulse"); global_gain=float(g19.mean_property_gain); retention=float(s0.mean_property_gain/global_gain) if global_gain>0 else float("nan")
    property_pass=bool(s0.relative_mean_gain>=.05-EPS and int(s0.wins)>int(s0.losses)); retention_pass=bool(global_gain>0 and retention>=.70-EPS); quality_pass=bool(s0.guardrail_pass)
    g19_boot=physical_bootstrap(paired,s0_id,"G19",BOOT_SEEDS[phase]+10); gpulse_boot=physical_bootstrap(paired,s0_id,"GPulse",BOOT_SEEDS[phase]+20); quality_g19=int(g19_boot["favorable_significant_count"])>=2
    gpulse_gain=float(gpulse.mean_property_gain); gpulse_retention=float(s0.mean_property_gain/gpulse_gain) if gpulse_gain>0 else (float("inf") if s0.mean_property_gain>0 else float("nan")); gpulse_point=physical_point_improvements(s0,gpulse); tradeoff_gpulse=bool((gpulse_gain<=0 or gpulse_retention>=.70-EPS) and len(gpulse_point)>=2)
    property_boot={}
    for index,other in enumerate(("C0","G19","GPulse")): property_boot[f"{s0_id}_vs_{other}_property_gain"]=bootstrap(-pair(paired,s0_id,other,"property_absolute_error"),BOOT_SEEDS[phase]+index)
    if phase=="p0":
        go=property_pass and retention_pass and quality_pass and quality_g19 and tradeoff_gpulse
        decision={"FIELD_ATTRIBUTION_DIAGNOSTIC":"COMPLETE","FIELD_DECOUPLING_CALIBRATION":"GO","PROPERTY_FIELD":frozen["property_field"],"QUALITY_HARM_FIELD":frozen["quality_harm_field"],"FIELD_CFG_P0":"GO" if go else "FAIL","FIELD_DECOUPLED_CFG_FORMAL256":"NOT_RUN","FIELD_DECOUPLING_EFFECT":"SUPPORTED" if go else "NOT_SUPPORTED","FIELD_STAGE_EFFECT":"SUPPORTED" if go and frozen["kind"]=="pulse" else "NOT_RUN" if frozen["kind"]=="constant" else "NOT_SUPPORTED","INNOVATION1_FINAL_STATUS":"MIXED","p0_n":int(paired.seed.nunique()),"selected_policy_id":s0_id,"property_pass":property_pass,"benefit_retention":retention,"benefit_retention_pass":retention_pass,"quality_guardrails_pass":quality_pass,"significant_quality_improvements_vs_G19":g19_boot["favorable_significant_count"],"quality_vs_G19_pass":quality_g19,"gpulse_gain_retention":gpulse_retention,"quality_point_improvements_vs_GPulse":gpulse_point,"tradeoff_vs_GPulse_pass":tradeoff_gpulse,"stop_reason":None if go else "Fresh P0 failed at least one frozen field-CFG gate; Formal256 is prohibited."}
    else:
        primary=float(property_boot[f"{s0_id}_vs_C0_property_gain"]["ci95_low"])>0
        c0_boot=physical_bootstrap(paired,s0_id,"C0",BOOT_SEEDS[phase]+30); no_adverse=bool(float(c0_boot["e_hull_reduction"]["ci95_high"])>=-.01 and all(float(c0_boot[f"{m}_gain"]["ci95_high"])>=0 for m in ("stable","nus","evaluation_valid")))
        quality_g19_formal=int(g19_boot["favorable_significant_count"])>=2; tradeoff_gpulse_formal=bool((gpulse_gain<=0 or gpulse_retention>=.70-EPS) and int(gpulse_boot["favorable_significant_count"])>=2); confirmed=primary and retention_pass and quality_pass and no_adverse and quality_g19_formal and tradeoff_gpulse_formal
        decision={"FIELD_ATTRIBUTION_DIAGNOSTIC":"COMPLETE","FIELD_DECOUPLING_CALIBRATION":"GO","PROPERTY_FIELD":frozen["property_field"],"QUALITY_HARM_FIELD":frozen["quality_harm_field"],"FIELD_CFG_P0":"GO","FIELD_DECOUPLED_CFG_FORMAL256":"CONFIRMED" if confirmed else "NOT_CONFIRMED","FIELD_DECOUPLING_EFFECT":"SUPPORTED" if confirmed else "NOT_SUPPORTED","FIELD_STAGE_EFFECT":"SUPPORTED" if confirmed and frozen["kind"]=="pulse" else "NOT_RUN" if frozen["kind"]=="constant" else "NOT_SUPPORTED","INNOVATION1_FINAL_STATUS":"CONFIRMED" if confirmed else "MIXED","formal_n":int(paired.seed.nunique()),"selected_policy_id":s0_id,"primary_ci_pass":primary,"benefit_retention":retention,"benefit_retention_pass":retention_pass,"quality_guardrails_pass":quality_pass,"no_adverse_quality_ci":no_adverse,"significant_quality_improvements_vs_G19":g19_boot["favorable_significant_count"],"significant_quality_improvements_vs_GPulse":gpulse_boot["favorable_significant_count"],"tradeoff_vs_GPulse_pass":tradeoff_gpulse_formal,"stop_reason":None if confirmed else "Formal256 failed at least one frozen confirmation gate."}
    json_write(ROOT/phase/"bootstrap_results.json",{"property":property_boot,"quality_vs_G19":g19_boot,"quality_vs_GPulse":gpulse_boot,**({"quality_vs_C0":c0_boot} if phase=="formal256" else {})}); plot_fresh(phase,paired,summary,s0_id,property_boot); return decision


def plot_calibration(summary:pd.DataFrame,selected:str|None)->None:
    view=summary.set_index("policy_id").loc[[p["policy_id"] for p in calibration_policies()]]
    specs=[("field_property_gain.png","mean_property_gain","Mean paired property gain"),("field_ehull_change.png","e_hull_worsening","Mean E-hull change"),("field_stable_change.png","stable_drop","Stable drop"),("field_nus_change.png","nus_drop","NUS drop")]
    for filename,column,title in specs:
        fig,ax=plt.subplots(figsize=(10,4.5)); ax.bar(range(len(view)),view[column]); ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(range(len(view)),view.index,rotation=35,ha="right"); ax.set_title(title); fig.tight_layout(); fig.savefig(PLOTS/filename,dpi=180); plt.close(fig)
    for filename,ylabel,column in (("property_vs_ehull_pareto.png","E-hull mean","e_hull_mean"),("property_vs_stable_pareto.png","Stable rate","stable_rate")):
        fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(view.mean_property_gain,view[column]);
        for identifier,item in view.iterrows(): ax.annotate(identifier,(item.mean_property_gain,item[column]),fontsize=8)
        ax.set(xlabel="Mean property gain",ylabel=ylabel); fig.tight_layout(); fig.savefig(PLOTS/filename,dpi=180); plt.close(fig)
    ids=[value for value in ("C0","G19","GPulse",selected) if value is not None]; selected_view=view.loc[ids]; fig,axes=plt.subplots(1,2,figsize=(9,4)); axes[0].bar(range(len(ids)),selected_view.mean_property_gain); axes[1].bar(range(len(ids)),selected_view.guardrail_margin)
    for ax in axes: ax.set_xticks(range(len(ids)),ids,rotation=25,ha="right")
    axes[0].set_title("Property gain"); axes[1].set_title("Guardrail margin"); fig.tight_layout(); fig.savefig(PLOTS/"global_vs_field_decoupled.png",dpi=180); plt.close(fig)
    pairs=[("A19","APulse"),("P19","PPulse"),("C19","CPulse")]; fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3); ax.bar(x-.18,[view.loc[a].mean_property_gain for a,_ in pairs],.36,label="constant"); ax.bar(x+.18,[view.loc[b].mean_property_gain for _,b in pairs],.36,label="pulse"); ax.set_xticks(x,["atomic","pos","cell"]); ax.legend(); ax.set_ylabel("Mean property gain"); fig.tight_layout(); fig.savefig(PLOTS/"constant_vs_field_pulse.png",dpi=180); plt.close(fig)


def plot_fresh(phase:str,paired:pd.DataFrame,summary:pd.DataFrame,s0_id:str,boot:dict)->None:
    prefix="p0" if phase=="p0" else "formal"; ids=["C0","G19","GPulse",s0_id]; pivot=paired.pivot(index="seed",columns="policy_id",values="property_absolute_error")
    fig,ax=plt.subplots(figsize=(6,5));
    for _,values in pivot[["C0",s0_id]].iterrows(): ax.plot([0,1],values,color=".75",linewidth=.5)
    ax.set_xticks([0,1],["C0","S0"]); ax.set_ylabel("Property absolute error"); fig.tight_layout(); fig.savefig(PLOTS/f"{prefix}_property_paired.png",dpi=180); plt.close(fig)
    if phase=="p0":
        e=paired.pivot(index="seed",columns="policy_id",values="e_hull"); fig,ax=plt.subplots(figsize=(6,5));
        for _,values in e[["C0",s0_id]].dropna().iterrows(): ax.plot([0,1],values,color=".75",linewidth=.5)
        ax.set_xticks([0,1],["C0","S0"]); ax.set_ylabel("E-hull"); fig.tight_layout(); fig.savefig(PLOTS/"p0_ehull_paired.png",dpi=180); plt.close(fig)
        for metric,filename in (("stable_rate","p0_stable.png"),("nus_rate","p0_nus.png")):
            view=summary.set_index("policy_id").loc[ids]; fig,ax=plt.subplots(figsize=(7,4)); ax.bar(range(4),view[metric]); ax.set_xticks(range(4),["C0","G19","GPulse","S0"]); ax.set_ylabel(metric); fig.tight_layout(); fig.savefig(PLOTS/filename,dpi=180); plt.close(fig)
    view=summary.set_index("policy_id").loc[ids]; fig,ax=plt.subplots(figsize=(7,4.5)); ax.boxplot([pivot[i].to_numpy(float) for i in ids],labels=["C0","G19","GPulse","S0"],showmeans=True); ax.set_ylabel("Property absolute error"); fig.tight_layout(); fig.savefig(PLOTS/f"{prefix}_c0_g19_gpulse_s0.png",dpi=180); plt.close(fig)
    if phase=="formal256":
        names=[];means=[];low=[];high=[]
        for name,result in boot.items(): names.append(name.replace(f"{s0_id}_vs_","").replace("_property_gain","")); means.append(result["mean"]); low.append(result["mean"]-result["ci95_low"]); high.append(result["ci95_high"]-result["mean"])
        fig,ax=plt.subplots(figsize=(7,4)); ax.errorbar(range(len(names)),means,yerr=[low,high],fmt="o",capsize=5); ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(range(len(names)),names); fig.tight_layout(); fig.savefig(PLOTS/"formal_bootstrap.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(view.mean_property_gain,view.e_hull_mean); [ax.annotate(i,(r.mean_property_gain,r.e_hull_mean)) for i,r in view.iterrows()]; ax.set(xlabel="Property gain",ylabel="E-hull"); fig.tight_layout(); fig.savefig(PLOTS/"formal_pareto.png",dpi=180); plt.close(fig)


def report(phase:str,decision:dict,summary:pd.DataFrame)->None:
    key={"calibration":"FIELD_DECOUPLING_CALIBRATION","p0":"FIELD_CFG_P0","formal256":"FIELD_DECOUPLED_CFG_FORMAL256"}[phase]; lines=[f"# {phase} final report","","Surrogate property and MatterSim quality evaluation; no DFT verification is claimed.","",f"{key} = {decision[key]}",f"Selected policy = {decision.get('selected_policy_id')}",f"Stop reason = {decision.get('stop_reason') or 'none'}","","## Final status",""]
    for status in ("FIELD_ATTRIBUTION_DIAGNOSTIC","FIELD_DECOUPLING_CALIBRATION","PROPERTY_FIELD","QUALITY_HARM_FIELD","FIELD_CFG_P0","FIELD_DECOUPLED_CFG_FORMAL256","FIELD_DECOUPLING_EFFECT","FIELD_STAGE_EFFECT","INNOVATION1_FINAL_STATUS"): lines.append(f"{status} = {decision[status]}")
    (ROOT/phase/"final_report.md").write_text("\n".join(lines)+"\n")


def main(phase:str)->None:
    PLOTS.mkdir(exist_ok=True); paired=build_paired(quality_enriched(phase)); summary=summarize(paired); write_metrics(phase,paired,summary); decision=calibration_gate(paired,summary) if phase=="calibration" else fresh_gate(phase,paired,summary); json_write(ROOT/phase/"decision_summary.json",decision); report(phase,decision,summary); print(json.dumps(decision,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("calibration","p0","formal256"),required=True); args=parser.parse_args(); main(args.phase)
