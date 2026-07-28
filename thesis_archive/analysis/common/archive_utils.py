from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
DATASETS={
 "innovation1":ROOT/"data/innovation1/per_seed_metrics.csv",
 "innovation2":ROOT/"data/innovation2/per_seed_metrics.csv",
 "compatibility_1":ROOT/"data/compatibility_1/per_seed_metrics.csv",
 "compatibility_2":ROOT/"data/compatibility_2/per_seed_metrics.csv",
 "leakage_diagnostic":ROOT/"data/leakage_diagnostic/per_seed_metrics.csv",
}
def load(name): return pd.read_csv(DATASETS[name])
def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def pair_summary(df,base,selected,tolerance=1e-6):
 d=df[selected]-df[base];b=float(df[base].mean());s=float(df[selected].mean())
 try:
  from scipy.stats import wilcoxon
  p=float(wilcoxon(d).pvalue) if np.any(d!=0) else 1.0
 except ValueError:p=1.0
 return {"n":int(len(df)),"baseline_mean":b,"selected_mean":s,"mean_difference":float(d.mean()),"relative_change":float((s-b)/b),"wins":int((d < -tolerance).sum()),"ties":int((d.abs()<=tolerance).sum()),"losses":int((d>tolerance).sum()),"harm_rate":float((d>tolerance).mean()),"wilcoxon_p":p}
def bootstrap_relative(df,base,selected,n=10000,seed=20260728):
 rng=np.random.default_rng(seed);x=df[[base,selected]].to_numpy();vals=[]
 for _ in range(n):
  z=x[rng.integers(0,len(x),len(x))];vals.append((z[:,1].mean()-z[:,0].mean())/z[:,0].mean())
 return [float(v) for v in np.quantile(vals,[0.025,0.975])]
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True)+"\n",encoding="utf-8")
def md_table(df): return df.to_markdown(index=False)+"\n"
