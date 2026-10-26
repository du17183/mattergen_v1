"""Detached supervisor: wait for actual evaluation, then fail-closed final exports."""
from pathlib import Path
import datetime, json, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parent
MINI=ROOT.parents[1]/'experiments/final_innovation1_audit/minimal_ablation'
PY='/mnt/datasets-livsyn/dxl/alm/.venv/bin/python'

def run(name):
    env=dict(os.environ);env.update(MPLCONFIGDIR='/tmp/mattergen-thesis-matplotlib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (ROOT/'logs'/('final_'+name+'.log')).open('a') as f:
        p=subprocess.run([PY,str(ROOT/name)],cwd=ROOT.parents[1],env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{name}: exit {p.returncode}; see logs/final_{name}.log')

def main():
    try:
        while True:
            run('refresh_status.py')
            status=json.loads((MINI/'pipeline_status.json').read_text())
            if status['status']=='FAILED':raise RuntimeError(status)
            if status['status']=='COMPLETED':break
            time.sleep(20)
        for name in ['collect_results.py','verify_evidence.py','make_plots.py','make_reports.py','final_audit.py','refresh_status.py']:
            print('Running',name,flush=True);run(name)
        result={'complete':True,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except BaseException as e:
        result={'complete':False,'error':str(e)}
        (ROOT/'supervisor_completion.json').write_text(json.dumps(result,indent=2)+'\n');raise
    (ROOT/'supervisor_completion.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':main()
