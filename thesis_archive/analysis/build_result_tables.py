import pandas as pd
from common.archive_utils import ROOT,load,pair_summary

def emit(name,df):
 for d in [ROOT/"tables/csv",ROOT/"tables/markdown",ROOT/"tables/latex"]:d.mkdir(parents=True,exist_ok=True)
 df.to_csv(ROOT/f"tables/csv/{name}.csv",index=False)
 (ROOT/f"tables/markdown/{name}.md").write_text(df.to_markdown(index=False)+"\n",encoding="utf-8")
 (ROOT/f"tables/latex/{name}.tex").write_text(df.to_latex(index=False,float_format=lambda x:f"{x:.6g}"),encoding="utf-8")
i1=load("innovation1");emit("table_innovation1",pd.DataFrame([{"evidence_class":"Independent formal","metric":"E-hull (eV/atom)","C0":i1.c0_ehull.mean(),"A0":i1.a0_ehull.mean(),"change":(i1.a0_ehull-i1.c0_ehull).mean()},{"evidence_class":"Independent formal","metric":"Stable rate","C0":i1.c0_stable.mean(),"A0":i1.a0_stable.mean(),"change":i1.a0_stable.mean()-i1.c0_stable.mean()},{"evidence_class":"Independent formal","metric":"NUS rate","C0":i1.c0_nus.mean(),"A0":i1.a0_nus.mean(),"change":i1.a0_nus.mean()-i1.c0_nus.mean()}]))
i2=load("innovation2");emit("table_innovation2",pd.DataFrame([{"evidence_class":"Independent formal","method":"C0","mean_max_force":i2.c0_max_force.mean(),"relative_change":0.0,"harm_rate":0.0},{"evidence_class":"Independent formal","method":"E3-A","mean_max_force":i2.e3a_max_force.mean(),"relative_change":(i2.e3a_max_force.mean()-i2.c0_max_force.mean())/i2.c0_max_force.mean(),"harm_rate":(i2.e3a_force_change>1e-6).mean()},{"evidence_class":"Independent formal","method":"E3-G","mean_max_force":i2.e3g_max_force.mean(),"relative_change":(i2.e3g_max_force.mean()-i2.c0_max_force.mean())/i2.c0_max_force.mean(),"harm_rate":i2.refinement_harm.mean()}]))
for data,name,label in [("compatibility_1","table_compatibility","Independent compatibility"),("compatibility_2","table_replication","Independent replication")]:
 d=load(data);s=pair_summary(d,"a0_max_force","a0_e3g_max_force");emit(name,pd.DataFrame([{"evidence_class":label,**s}]))
l=load("leakage_diagnostic");rows=[]
for cohort,d,label in [("training_overlap",l[l.training_overlap],"Leakage diagnostic"),("held_out",l[~l.training_overlap],"Supplementary held-out"),("mixed_256",l,"Invalid mixed cohort")]:rows.append({"evidence_class":label,"cohort":cohort,"n":len(d),"mean_force_difference":d.force_difference.mean(),"harm_count":int(d.refinement_harm.sum()),"harm_rate":d.refinement_harm.mean(),"valid_for_formal_claims":False,"valid_for_supplementary_claims":cohort=="held_out"})
emit("table_leakage_diagnostic",pd.DataFrame(rows));print("result tables generated")
