import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.archive_utils import ROOT,load,bootstrap_relative
OUT=ROOT/"figures/generated";OUT.mkdir(parents=True,exist_ok=True)
def save(fig,name):
 fig.tight_layout()
 for ext in ["png","svg","pdf"]:
  p=OUT/f"{name}.{ext}";fig.savefig(p,dpi=220 if ext=="png" else None,bbox_inches="tight")
  if ext=="svg":p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n",encoding="utf-8")
 plt.close(fig)
# 1 Adaptive metrics
i1=load("innovation1");fig,ax=plt.subplots(figsize=(6.2,4));x=np.arange(3);w=.34;ax.bar(x-w/2,[i1.c0_ehull.mean(),i1.c0_stable.mean(),i1.c0_nus.mean()],w,label="C0");ax.bar(x+w/2,[i1.a0_ehull.mean(),i1.a0_stable.mean(),i1.a0_nus.mean()],w,label="A0");ax.set_xticks(x,["E-hull","Stable","NUS"]);ax.set_title("Adaptive CFG proxy metrics");ax.legend();save(fig,"fig1_adaptive_cfg_metrics")
# 2 three-arm force
i2=load("innovation2");fig,ax=plt.subplots(figsize=(5.5,4));ax.bar(["C0","E3-A","E3-G"],[i2.c0_max_force.mean(),i2.e3a_max_force.mean(),i2.e3g_max_force.mean()]);ax.set_ylabel("Pre-relax max force (eV/A)");ax.set_title("E3-PCR formal 256");save(fig,"fig2_e3_pcr_force")
# 3 harm
fig,ax=plt.subplots(figsize=(5.5,4));ax.bar(["E3-A","E3-G"],[(i2.e3a_force_change>1e-6).mean(),i2.refinement_harm.mean()]);ax.set_ylabel("Refinement harm rate");ax.set_ylim(0,.32);ax.set_title("Always-on vs learned gate safety");save(fig,"fig3_gate_safety")
# 4/5 paired plots
for data,name,title in [("compatibility_1","fig4_compatibility_paired","Compatibility 41000-41063"),("compatibility_2","fig5_replication_paired","Replication 50000-50063")]:
 d=load(data);fig,ax=plt.subplots(figsize=(5.5,4.5));
 for _,r in d.iterrows():ax.plot([0,1],[r.a0_max_force,r.a0_e3g_max_force],color="0.75",alpha=.45,lw=.7)
 ax.scatter(np.zeros(len(d)),d.a0_max_force,s=12,label="A0");ax.scatter(np.ones(len(d)),d.a0_e3g_max_force,s=12,label="A0+E3-G");ax.set_xticks([0,1],["A0","A0+E3-G"]);ax.set_ylabel("Pre-relax max force (eV/A)");ax.set_title(title);save(fig,name)
# 6 forest
d1=load("compatibility_1");d2=load("compatibility_2");effects=[];cis=[]
for d in [d1,d2]:effects.append((d.a0_e3g_max_force.mean()-d.a0_max_force.mean())/d.a0_max_force.mean());cis.append(bootstrap_relative(d,"a0_max_force","a0_e3g_max_force"))
fig,ax=plt.subplots(figsize=(6,3.7));y=np.arange(2);err=np.array([[effects[i]-cis[i][0] for i in range(2)],[cis[i][1]-effects[i] for i in range(2)]]);ax.errorbar(effects,y,xerr=err,fmt="o",capsize=4);ax.axvline(0,color="k",lw=1);ax.set_yticks(y,["Compatibility 1","Replication 2"]);ax.set_xlabel("Relative max-force change");ax.set_title("Independent validation forest plot");save(fig,"fig6_independent_forest")
# 7 leakage
l=load("leakage_diagnostic");fig,ax=plt.subplots(figsize=(5.7,4));ax.bar(["Training overlap\n(n=64)","Held-out\n(n=192)"],[l[l.training_overlap].refinement_harm.mean(),l[~l.training_overlap].refinement_harm.mean()]);ax.set_ylabel("Refinement harm rate");ax.set_title("Training-overlap leakage diagnostic");save(fig,"fig7_leakage_harm")
print(f"generated {len(list(OUT.glob('*')))} figure files in {OUT}")
