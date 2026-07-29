from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact
from common.archive_utils import ROOT,load,pair_summary,dump
out={}
i1=load("innovation1")
out["innovation1"]={"n":len(i1),"c0_ehull":float(i1.c0_ehull.mean()),"a0_ehull":float(i1.a0_ehull.mean()),"ehull_change":float((i1.a0_ehull-i1.c0_ehull).mean()),"stable_change":float(i1.a0_stable.mean()-i1.c0_stable.mean()),"nus_change":float(i1.a0_nus.mean()-i1.c0_nus.mean())}
i2=load("innovation2");out["innovation2_e3a"]=pair_summary(i2,"c0_max_force","e3a_max_force");out["innovation2_e3g"]=pair_summary(i2,"c0_max_force","e3g_max_force")
for key in ["compatibility_1","compatibility_2"]:out[key]=pair_summary(load(key),"a0_max_force","a0_e3g_max_force")
leak=load("leakage_diagnostic");tr=leak[leak.training_overlap];ho=leak[~leak.training_overlap];table=[[int(tr.refinement_harm.sum()),int((~tr.refinement_harm).sum())],[int(ho.refinement_harm.sum()),int((~ho.refinement_harm).sum())]];out["leakage"]={"training_overlap_n":len(tr),"heldout_n":len(ho),"training_harm_count":table[0][0],"heldout_harm_count":table[1][0],"training_harm_rate":float(tr.refinement_harm.mean()),"heldout_harm_rate":float(ho.refinement_harm.mean()),"fisher_one_sided_p":float(fisher_exact(table,alternative="less").pvalue)}
dump(ROOT/"analysis_outputs/recomputed_statistics.json",out)
(ROOT/"tables/csv").mkdir(parents=True,exist_ok=True)
rows=[]
for exp,v in out.items():
 for metric,value in v.items():rows.append({"experiment":exp,"metric":metric,"value":value})
pd.DataFrame(rows).to_csv(ROOT/"tables/csv/recomputed_statistics.csv",index=False)
print(ROOT/"analysis_outputs/recomputed_statistics.json")
