from pathlib import Path
import json,re
import pandas as pd
from common.archive_utils import ROOT,load,sha256,dump
errors=[];warnings=[];checks={}
def require(rel):
 p=ROOT/rel
 if not p.exists():errors.append(f"missing: {rel}")
 return p
required=["README.md","README_FOR_LAPTOP.md","FINAL_EXPERIMENT_MANIFEST.md","FINAL_EXPERIMENT_MANIFEST.json","CLAIM_EVIDENCE_MATRIX.md","DATA_DICTIONARY.md","EXPERIMENT_LINEAGE.md","LIMITATIONS.md","ARTIFACTS_NOT_IN_GITHUB.md","FILE_INVENTORY.csv","CHECKSUMS.sha256","requirements-analysis.txt","configs/adaptive_cfg_final.yaml","configs/e3_pcr_final.yaml","configs/evaluation_final.yaml"]
for x in required:require(x)
expdirs=["innovation1_adaptive_cfg","innovation2_e3_pcr_formal256","compatibility_41000_41063","replication_50000_50063","leakage_diagnostic","source_data_incomplete_audit"]
for e in expdirs:
 for f in ["README.md","experiment_manifest.json","seeds.txt","data_schema.md","source_files.md","reproduction.md","limitations.md","final_summary.json"]:require(Path("experiments")/e/f)
for p in ROOT.rglob("*.json"):
 try:json.loads(p.read_text(encoding="utf-8"))
 except Exception as exc:errors.append(f"invalid JSON {p.relative_to(ROOT)}: {exc}")
expected={"innovation1":(256,20000,20255),"innovation2":(256,40000,40255),"compatibility_1":(64,41000,41063),"compatibility_2":(64,50000,50063),"leakage_diagnostic":(256,20000,20255)}
D={}
for name,(n,lo,hi) in expected.items():
 try:d=load(name);D[name]=d
 except Exception as exc:errors.append(f"CSV unreadable {name}: {exc}");continue
 if len(d)!=n:errors.append(f"{name} row count {len(d)} != {n}")
 if d.seed.duplicated().any():errors.append(f"{name} duplicate seeds")
 if int(d.seed.min())!=lo or int(d.seed.max())!=hi:errors.append(f"{name} seed range mismatch")
training=set(range(20000,20064))
for name in ["innovation2","compatibility_1","compatibility_2"]:
 if name in D and set(D[name].seed)&training:errors.append(f"formal E3-G dataset {name} overlaps Q3 training")
if "leakage_diagnostic" in D:
 l=D["leakage_diagnostic"];tr=l[l.training_overlap];ho=l[~l.training_overlap]
 if len(tr)!=64 or len(ho)!=192:errors.append("leakage cohort counts mismatch")
 if set(tr.seed)!=training or set(ho.seed)!=set(range(20064,20256)):errors.append("leakage cohort labels mismatch")
 if l.valid_for_formal_claims.astype(bool).any():errors.append("diagnostic rows marked formal")
 if not ho.valid_for_supplementary_claims.astype(bool).all() or tr.valid_for_supplementary_claims.astype(bool).any():errors.append("supplementary flags mismatch")
# source manifests: archived relative path must resolve and match recorded source hash
for mp in (ROOT/"data").glob("*/source_manifest.json"):
 try:m=json.loads(mp.read_text(encoding="utf-8"))
 except Exception:continue
 seen=set()
 for entry in m.get("sources",[]):
  rel=entry.get("archived_source","");target=ROOT/rel
  if rel in seen:errors.append(f"duplicate archived source target in {mp.relative_to(ROOT)}: {rel}")
  seen.add(rel)
  if not target.is_file():errors.append(f"source manifest target missing: {rel}")
  elif sha256(target)!=entry.get("sha256"):errors.append(f"source manifest hash mismatch: {rel}")
# cross-check frozen values
def close(label,x,y,tol=5e-7):
 if abs(float(x)-float(y))>tol:errors.append(f"DATA_MISMATCH {label}: {x} vs {y}")
if "innovation1" in D:
 d=D["innovation1"];close("i1 ehull",(d.a0_ehull-d.c0_ehull).mean(),-0.003434944937153267);close("i1 stable",d.a0_stable.mean()-d.c0_stable.mean(),0.05859375);close("i1 nus",d.a0_nus.mean()-d.c0_nus.mean(),0.03515625)
if "innovation2" in D:
 d=D["innovation2"];close("C0 force",d.c0_max_force.mean(),0.34296426286559617);close("E3-A force",d.e3a_max_force.mean(),0.24395595983940943);close("E3-G force",d.e3g_max_force.mean(),0.26310701792559776);close("E3-A harm",(d.e3a_force_change>1e-6).mean(),0.25390625);close("E3-G harm",d.refinement_harm.mean(),0.18359375)
for name,b,s in [("compatibility_1",0.21730158680797604,0.1584162688975449),("compatibility_2",0.26527988196406616,0.21482998015936608)]:
 if name in D:close(name+" base",D[name].a0_max_force.mean(),b);close(name+" selected",D[name].a0_e3g_max_force.mean(),s)
if "leakage_diagnostic" in D:
 l=D["leakage_diagnostic"];close("train harm",l[l.training_overlap].refinement_harm.sum(),0);close("heldout harm",l[~l.training_overlap].refinement_harm.sum(),31)
# portability: executable analysis/config files only; provenance docs may contain original server paths
forbidden_path="/"+"data"+"/"+"dxl"
for p in list((ROOT/"analysis").rglob("*.py"))+list((ROOT/"configs").rglob("*")):
 if p.is_file() and forbidden_path in p.read_text(encoding="utf-8",errors="ignore"):errors.append(f"absolute server path in portable file: {p.relative_to(ROOT)}")
secret_patterns=[re.compile(r"ghp_[A-Za-z0-9]{20,}"),re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),re.compile(r"AKIA[0-9A-Z]{16}"),re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")]
for p in ROOT.rglob("*"):
 if p.is_file() and p.stat().st_size<5_000_000:
  t=p.read_text(encoding="utf-8",errors="ignore")
  if any(q.search(t) for q in secret_patterns):errors.append(f"possible secret: {p.relative_to(ROOT)}")
large=[str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size>90*1024*1024]
if large:errors.extend(f"file over 90MB: {x}" for x in large)
# checksums deliberately exclude self-referential metadata and generated validation reports
checksum=ROOT/"CHECKSUMS.sha256"
if checksum.exists():
 for line in checksum.read_text().splitlines():
  if not line.strip():continue
  expected_hash,rel=line.split("  ",1);p=ROOT/rel
  if not p.exists():errors.append(f"checksum target missing: {rel}")
  elif sha256(p)!=expected_hash:errors.append(f"checksum mismatch: {rel}")
formal_leakage=any("overlaps Q3 training" in e for e in errors)
valid=not errors
result={"ARCHIVE_VALID":valid,"LAPTOP_ANALYSIS_READY":valid,"FORMAL_DATA_LEAKAGE_FOUND":formal_leakage,"DATA_MISMATCH_DETECTED":any("DATA_MISMATCH" in e for e in errors),"SECRETS_FOUND":any("secret" in e for e in errors),"FILES_OVER_90MB":large,"errors":errors,"warnings":warnings,"checks":{"required_files":len(required),"experiment_directories":len(expdirs),"dataset_counts":{k:len(v) for k,v in D.items()}}}
dump(ROOT/"ARCHIVE_VALIDATION.json",result)
(ROOT/"ARCHIVE_VALIDATION.md").write_text("# 归档验证\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in result.items() if k not in ["errors","warnings","checks"])+"\n\n## Errors\n\n"+("无。" if not errors else "\n".join(f"- {x}" for x in errors))+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if valid else 1)
