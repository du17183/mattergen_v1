from pathlib import Path
import csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={"FILE_INVENTORY.csv","CHECKSUMS.sha256","ARCHIVE_VALIDATION.json","ARCHIVE_VALIDATION.md"}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def classify(rel):
 top=rel.parts[0] if len(rel.parts)>1 else "root_document"
 return {"data":"data","experiments":"experiment","analysis":"analysis_code","tables":"result_table","figures":"figure","reports":"report","configs":"config"}.get(top,"documentation")
def experiment(rel):
 s=str(rel)
 for k in ["innovation1","innovation2","compatibility_1","compatibility_2","leakage_diagnostic","source_data_incomplete"]:
  if k in s:return k
 return "archive_wide"
files=sorted(p for p in ROOT.rglob("*") if p.is_file())
rows=[];checks=[]
for p in files:
 rel=p.relative_to(ROOT);r=str(rel).replace("\\","/");excluded=r in EXCLUDED
 rows.append({"relative_path":r,"size_bytes":p.stat().st_size,"sha256":"" if excluded else sha(p),"category":classify(rel),"experiment":experiment(rel),"required_for_laptop_analysis":str(rel.parts[0] in {"data","analysis","configs","tables"} or r in {"README.md","README_FOR_LAPTOP.md","requirements-analysis.txt"}),"source_path":"see data/*/source_manifest.json" if rel.parts[0]=="data" else "","generated_or_original":"original_copy" if "/source/" in "/"+r else "generated","notes":"excluded from recursive checksum metadata" if excluded else ""})
 if not excluded:checks.append(f"{sha(p)}  {r}")
with open(ROOT/"FILE_INVENTORY.csv","w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
(ROOT/"CHECKSUMS.sha256").write_text("\n".join(checks)+"\n",encoding="utf-8")
print(f"inventory={len(rows)} checksum_entries={len(checks)}")
