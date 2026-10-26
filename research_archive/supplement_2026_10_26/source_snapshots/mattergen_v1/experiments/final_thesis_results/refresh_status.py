"""Small, non-GPU status refresher for the detached finalization supervisor."""
from pathlib import Path
import csv, datetime, json

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
MINI=PROJECT/'experiments/final_innovation1_audit/minimal_ablation'

def main():
    status=json.loads((MINI/'pipeline_status.json').read_text())
    count=sum(1 for _ in (MINI/'generation').glob('*/*/run_summary.json'))
    qcount=sum(1 for _ in (MINI/'p0_quality').glob('*/official_detailed.json.gz'))
    complete=(ROOT/'final_status.json').exists() and json.loads((ROOT/'final_status.json').read_text()).get('THESIS_EXPERIMENTS_COMPLETE')=='YES'
    if (ROOT/'run_registry.csv').exists():
        with (ROOT/'run_registry.csv').open() as f:rows=list(csv.DictReader(f))
        for r in rows:
            if r['experiment_id']=='EXP-03':
                r['status']=status['status'];r['notes']=f"generation {count}/48; quality {qcount}/6; stage={status.get('stage','completed')}"
                if r['status']=='COMPLETED':r['exit_code']='0';r['end_time']=status.get('end_utc','')
            if r['experiment_id']=='EXP-09' and complete:r['status']='COMPLETED';r['exit_code']='0'
        with (ROOT/'run_registry.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    items=[('Innovation1 Formal','COMPLETED_EXISTING','256 pairs; directional gate confirmed, primary CIs cross zero','minimal8 quality direction not replicated; preserve mixed evidence' if complete else 'Supplementary component diagnostic '+status['status']),
      ('F0 Formal256','COMPLETED_EXISTING','256 pairs; original formal STRONG_CONFIRMED','None'),
      ('Independent MLIP','COMPLETED_EXISTING','512 structures; MaxF and mean-force raw vectors checked','None'),
      ('Force Direction Ablation','COMPLETED_EXISTING','32 pairs x 5 arms; matched event/time/norm audit','None'),
      ('Bounded Correction','COMPLETED_EXISTING','32 pairs x 3 arms; necessity not supported','Retain negative result'),
      ('Equal-budget POST','COMPLETED_EXISTING','32 pairs x 3 arms; 20 actual calls each','MatterSim POST better; independent direct CI crosses zero'),
      ('Adaptive CFG + F0','COMPLETED_EXISTING','64 pairs x 4 arms; retention FAIL','No rescue'),
      ('E3-G Baseline','COMPLETED_EXISTING','256 pairs; separate target/cohort','Do not rank absolute cross-cohort means'),
      ('DFT','SKIPPED_NOT_APPLICABLE','NOT_RUN_ENVIRONMENT_UNAVAILABLE; n=0','Optional; DFT_VERIFIED=False'),
      ('Innovation1 minimal ablation',status['status'],f'{count}/48 generation; {qcount}/6 quality',status.get('stage','None'))]
    for title,paths in [('Final Tables',['main_results.csv','relative_improvement_table.csv','ablation_results.csv','statistical_results.md','efficiency_results.md']),('Final Figures',['plots/overall_method_framework.png','plots/interaction_analysis.png'])]:
        done=all((ROOT/p).exists() for p in paths)
        items.append((title,'COMPLETED' if complete and done else 'RUNNING','Existing results exported; final refresh after minimal8' if not complete else 'All required artifacts verified','None' if complete else 'Await final refresh'))
    for n,f in [(3,'chapter3_innovation1.md'),(4,'chapter4_innovation2.md'),(5,'chapter5_experiments.md')]:
        exists=(PROJECT/'thesis/final_materials'/f).exists();items.append((f'Chapter {n} Materials','COMPLETED' if exists else 'PENDING','Generated from final data' if exists else 'Waiting for completed evaluation','None' if exists else 'Automatic report generation'))
    text='# Final thesis progress\n\nUpdated UTC: '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n\nTHESIS_EXPERIMENTS_COMPLETE='+('YES' if complete else 'NO')+'\n\n| Experiment | Status | Key Result | Next |\n|---|---|---|---|\n'
    text+='\n'.join('| '+' | '.join(r)+' |' for r in items)+'\n\n状态依据实际文件；负效果不等于作业执行失败。未修改原始实验或 stash。\n'
    (ROOT/'master_status.md').write_text(text)
    print(json.dumps({'generation':count,'quality_methods':qcount,'pipeline':status,'THESIS_EXPERIMENTS_COMPLETE':complete}))

if __name__=='__main__':main()
