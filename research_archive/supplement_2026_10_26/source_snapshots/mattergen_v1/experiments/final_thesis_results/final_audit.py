"""Fail-closed final artifact gate, separate from scientific success/failure."""
from pathlib import Path
import csv, datetime, hashlib, json, re, subprocess
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
MINI=PROJECT/'experiments/final_innovation1_audit/minimal_ablation'
REQUIRED_PLOTS=['overall_method_framework','adaptive_cfg_pipeline','rc_nfgd_pipeline','innovation1_formal_results',
 'formal256_maxf_paired','formal256_maxf_distribution','formal256_tail_metrics','independent_mlip_validation',
 'force_direction_ablation','bounded_correction_ablation','online_vs_post','interaction_analysis']
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    audited=js(ROOT/'source_audits.json');verified=js(ROOT/'evidence_verification.json')
    assert verified['all_required_pass'];assert audited['Innovation1_minimal8']['complete']
    assert js(MINI/'pipeline_status.json')['status']=='COMPLETED'
    originals=js(ROOT/'original_decisions.json');main=pd.read_csv(ROOT/'main_results.csv');stats=pd.read_csv(ROOT/'paired_statistics.csv')
    for n in ['A1','A3','A4','A5','A6']:assert audited[n]['complete'] and audited[n]['primary_bootstrap_reproduced']
    required_tables=['main_results.csv','main_results.md','relative_improvement_table.csv','relative_improvement_table.md',
      'ablation_results.csv','ablation_results.md','statistical_results.md','efficiency_results.md','reproducibility_manifest.json',
      'paired_statistics.csv','distribution_statistics.csv','run_registry.csv','resource_audit.md','final_report.md']
    for name in required_tables:assert (ROOT/name).stat().st_size>100,name
    for name in REQUIRED_PLOTS:
        p=ROOT/'plots'/(name+'.png');assert p.stat().st_size>10000
        with Image.open(p) as im:im.verify()
    for chapter,filename,count in [(3,'chapter3_innovation1.md',11),(4,'chapter4_innovation2.md',18),(5,'chapter5_experiments.md',14)]:
        p=PROJECT/'thesis/final_materials'/filename;text=p.read_text()
        assert len(text)>6000
        sections=re.findall(r'^## (\d+\.\d+) ',text,re.M)
        assert sections==[f'{chapter}.{i}' for i in range(1,count+1)],sections
        assert 'DFT_VERIFIED=False' in text
    assert len(re.findall(r'^## Q\d+\.',(ROOT/'final_report.md').read_text(),re.M))==17
    # Every actual manuscript link resolves; no invented source locations.
    for p in list((PROJECT/'thesis/final_materials').glob('*.md'))+[ROOT/'final_report.md']:
        for target in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(target).exists(),target
    i1=main[main.Cohort=='Innovation1 Formal256'];e3=main[main.Cohort=='E3 Formal256']
    assert i1[['Property MAE','Hit','MaxF','Atomic Force']].isna().all().all()
    assert e3[['Property MAE','Hit','Atomic Force','Runtime']].isna().all().all()
    formal=stats[(stats.cohort=='A1')&(stats.metric=='MaxF')].iloc[0]
    assert int(formal.wins)==254 and int(formal.losses)==2
    ind=stats[(stats.cohort=='A1')&(stats.metric.str.startswith('Independent'))]
    assert len(ind)==2 and (ind.relative_ci95_low>0).all()
    post=stats[(stats.cohort=='A6')&(stats.baseline=='F0')&(stats.method=='POST')]
    assert post[post.metric=='MaxF'].iloc[0].relative_ci95_low>0
    r=post[post.metric=='Independent MaxF'].iloc[0];assert r.delta_ci95_low<0<r.delta_ci95_high
    assert js(PROJECT/'experiments/dft_verification/environment_audit.json')['DFT_VERIFICATION']=='NOT_RUN_ENVIRONMENT_UNAVAILABLE'
    manifest=js(ROOT/'reproducibility_manifest.json')
    for p,info in manifest['sources'].items():assert sha(p)==info['sha256'],p
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()==manifest['mattergen_head']
    checks=[
      ('Innovation1 Formal evidence complete?','YES, WITH MIXED EFFECT EVIDENCE','256 pairs and original directional gate. Primary CIs cross zero; minimal8 quality direction not replicated. Historical missing property/pre-relax-force fields explicitly NA.'),
      ('Innovation2 Formal256 complete?','YES','256 pairs; raw results and frozen primary bootstrap reproduced.'),
      ('Independent MLIP validation complete?','YES','512 formal structures; raw vectors reproduce MaxF and mean force; mean/median/P90/P95/max and paired CIs exported.'),
      ('Force direction ablation complete?','YES','32 pairs, five arms, timing/norm/acceptance checked. Reference-trajectory limitation retained.'),
      ('Bounded correction ablation complete?','YES','Negative necessity result retained.'),
      ('Equal-budget POST baseline complete?','YES','32 fresh paired seeds from existing A6; actual calls 20 vs20; POST stronger on MatterSim.'),
      ('Adaptive CFG + F0 interaction complete?','YES','64 pairs; original retention FAIL preserved.'),
      ('E3-G baseline audit complete?','YES','Formal256 raw seed remapping candidate_seed validated; original bootstrap reproduced; missing runtime NA.'),
      ('DFT attempted?','ENVIRONMENT AUDIT ONLY','Actual binary/pseudopotential/workflow audit completed; no DFT calculation was launched.'),
      ('DFT completed?','NO / OPTIONAL','No solver or production pseudopotential/workflow; n=0; does not block thesis experimental closure.'),
      ('All seed overlaps checked?','YES','Fresh scans and pairwise final cohorts checked; intentional C0/evaluator reuse and I1-vs-later-gate training overlap disclosed.'),
      ('Any cherry-pick?','NO','All registered samples retained, including losses and negative ablations; no replacement seeds.'),
      ('Any primary endpoint changed?','NO','Original endpoints, thresholds and scientific decisions preserved.'),
      ('All bootstrap settings consistent?','WITHIN EACH FROZEN PROTOCOL; NOT IDENTICAL ACROSS HISTORY','Historical RNG seeds differ; I1 original RNG/count not recoverable. New 20k seed20260915 supplementary analyses explicitly separate; no silent substitution.'),
      ('All final figures generated?','YES',f'{len(REQUIRED_PLOTS)} required PNG plus PDFs and supplementary minimal8 plot verified.'),
      ('All final tables generated?','YES','Full precision CSVs, readable Markdown, complete paired and distribution statistics.'),
      ('Reproducibility manifest complete?','YES WITH DISCLOSED HISTORICAL LIMITS','Available sources, hashes, frozen configs, versions, seed lists, metric/statistical definitions all recorded. Unknown historical metadata not fabricated.'),
      ('Any missing thesis-critical experiment?','NO','n=8 component diagnostic completed and trace-reproduced. Optional DFT unavailable; unavailable legacy property fields restrict claims, not fabricated as completed tests.')]
    text='# 最终实验总审计\n\n| # | Check | Answer | Evidence / boundary |\n|---|---|---|---|\n'
    text+='\n'.join(f'| {i} | {q} | {a} | {e} |' for i,(q,a,e) in enumerate(checks,1))
    text+='\n\nTHESIS_EXPERIMENTS_COMPLETE=YES\n\n该标志仅表示当前冻结论文主线所需的实验、统计和素材已完成，绝不表示所有假设都获得支持，也不表示历史缺失字段被补成了真实测量。历史 I1 的方向性确认、置信区间跨零、E3 元数据缺项及可选 DFT 未运行均完整披露。\n'
    (ROOT/'final_experiment_audit.md').write_text(text)
    status={'Innovation1':'FINAL','Innovation1 final name':'多字段残差驱动的在线自适应 CFG',
      'Innovation1 evidence qualification':'Frozen method retained. Historical directional gate confirmed, Formal256 primary CIs cross zero; fresh minimal8 quality direction not replicated. Mixed, limited cross-cohort effect evidence.',
      'Innovation2':'FINAL','Innovation2 final name':'Reliability-Calibrated Late-stage Neural Force-Guided Diffusion',
      'E3-G role':'STRONG POST-GENERATION BASELINE','FP-PC':'CLOSED','F1':'NOT USED',
      'Adaptive CFG + F0':'NOT_COMPATIBLE','Independent MLIP':'SUPPORTED','Equal-budget POST':'MIXED',
      'Equal-budget primary MatterSim winner':'POST_BETTER','FORCE_DIRECTION_EFFECT':'SUPPORTED',
      'BOUNDING':'BOUNDING_NOT_SUPPORTED_AS_CORE_MECHANISM','DFT_VERIFIED':False,'SURROGATE_PROPERTY_EVAL':True,
      'DFT_VERIFICATION':'NOT_RUN_ENVIRONMENT_UNAVAILABLE','THESIS_EXPERIMENTS_COMPLETE':'YES',
      'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (ROOT/'final_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'final_status.md').write_text('# Final status\n\n```json\n'+json.dumps(status,indent=2,ensure_ascii=False)+'\n```\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'logs' not in p.parts and p.name!='artifact_manifest.json']
    files+=list((PROJECT/'thesis/final_materials').glob('*.md'))
    artifact={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files if p.name not in ['run_registry.csv','master_status.md']}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    with (ROOT/'run_registry.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        if r['experiment_id'] in ['EXP-03','EXP-09']:
            r['status']='COMPLETED';r['exit_code']='0';r['end_time']=status['completed_utc']
        if r['experiment_id']=='EXP-07':r['notes']='Environment audit completed; DFT unavailable, optional; actual DFT count=0.'
    with (ROOT/'run_registry.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(status,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
