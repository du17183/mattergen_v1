"""Resource-separated automatic continuation of the six registered cohorts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiments.innovation2_finalization.protocol import ROOT,PROJECT,SPECS,config,output,write_json

def env(gpu):
    e=dict(os.environ);e.update(PYTHONPATH=str(PROJECT),CUDA_VISIBLE_DEVICES=str(gpu),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TMPDIR=str(PROJECT/'.tmp'),
        MPLCONFIGDIR=str(PROJECT/'.cache/matplotlib'),XDG_CACHE_HOME=str(PROJECT/'.cache'),
        HF_HOME=str(PROJECT/'.cache/huggingface'),TORCH_HOME=str(PROJECT/'.cache/torch'))
    return e

def run(code,stage,args,gpu=7):
    out=output(code);log=out/'logs'/f'pipeline_{stage}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as f:
        process=subprocess.Popen([sys.executable,'-m','experiments.innovation2_finalization.'+args[0],*args[1:]],
            cwd=PROJECT,env=env(gpu),stdout=f,stderr=subprocess.STDOUT)
        record={'track':code,'stage':stage,'pid':process.pid,'start_epoch':time.time(),'log':str(log)}
        write_json(out/'active_stage.json',record);print(json.dumps(record),flush=True)
        rc=process.wait();record.update(return_code=rc,end_epoch=time.time())
        write_json(out/(f'{stage}_stage_result.json'),record)
        if rc:raise RuntimeError(json.dumps(record))
    print(json.dumps({'track':code,'stage_complete':stage}),flush=True)

def wait_generated(code,limit=None):
    c=config(code);out=output(code);seeds=c['seeds'] if limit is None else c['seeds'][:limit]
    while not all((out/'generation'/m/str(s)/'run_summary.json').exists() for m in c['methods'] for s in seeds):
        if (out/'generation_failures.json').exists():raise RuntimeError(f'{code} generation failures need audit')
        time.sleep(10)
    # A1 writes actual-call accounting immediately after its final summary.
    if code=='A1':
        while not all((out/'generation/F0'/str(s)/'force_call_accounting.json').exists() for s in seeds):time.sleep(2)

def gpu_evaluation(code,gpu):
    if (output(code)/'decision_summary.json').exists():
        print(json.dumps({'track':code,'evaluation':'already_complete'}),flush=True)
        return False
    for stage in ['prepare','properties']:
        run(code,stage,['evaluate_cohort','--track',code,'--stage',stage,'--gpu',str(gpu)],gpu)
    if not (ROOT/'independent_mlip'/code/'decision_summary.json').exists():
        run(code,'independent',['independent_mlip','--cohort',code,'--device','cuda'],gpu)
    run(code,'relax',['evaluate_cohort','--track',code,'--stage','relax','--gpu',str(gpu)],gpu)
    return True

def cpu_evaluation(code):
    if (output(code)/'decision_summary.json').exists():
        print(json.dumps({'track':code,'analysis':'already_complete'}),flush=True)
        return
    run(code,'quality',['evaluate_cohort','--track',code,'--stage','quality'],gpu='')
    run(code,'analysis',['analyze_cohort','--track',code],gpu='')
    subprocess.run([sys.executable,'-m','experiments.innovation2_finalization.status'],cwd=PROJECT,check=True)

def final_report():
    states={}
    for code in SPECS:
        path=output(code)/'decision_summary.json'
        states[code]=json.loads(path.read_text()) if path.exists() else {'status':'INCOMPLETE'}
    ip=ROOT/'independent_mlip/A1/decision_summary.json'
    independent=json.loads(ip.read_text()) if ip.exists() else {'status':'PENDING'}
    supported=(states['A1']['status'] in ['CONFIRMED','STRONG_CONFIRMED'] and independent['status']=='PASS'
      and states['A3']['status']=='SUPPORTED' and states['A4']['status']=='SUPPORTED' and states['A5']['status']=='PASS')
    result={'F0_Formal32':'STRONG_CONFIRMED','tracks':states,'independent_Formal256':independent,
      'complete':all(s['status'] not in ['INCOMPLETE','AUDIT_FAIL','INCOMPLETE_INDEPENDENT_EVALUATION'] for s in states.values()),
      'full_F0_evidence_chain_supported':supported,'DFT_VERIFIED':False,'thesis_body_modified':False,
      'F1_replaces_F0':False,'budget_allocation_implemented':False,
      'final_method':(states['A5'].get('FINAL_MODEL','F0 standalone') if states['A1']['status'] in ['CONFIRMED','STRONG_CONFIRMED']
                      else 'No new formal model decision; retain frozen Formal32 and prior fallback evidence'),
      'F1_next':'Request next-round Formal32 confirmation; P0 alone does not establish final model' if states['B']['status']=='GO' else 'Retain F0; no rescue tuning'}
    write_json(ROOT/'final_decision.json',result)
    lines=['# Innovation 2 unified final report','',
      f"All requested cohorts evaluated and audited: {result['complete']}",'',
      '| Track | Status | Report |','|---|---|---|']
    for code,s in states.items():lines.append(f"| {code} | {s['status']} | [{SPECS[code][0]}]({output(code)/'final_report.md'}) |")
    lines+=['',f"Independent Formal256: {independent['status']}",
      f"Complete F0 evidence chain supported: {supported}",f"Final model recommendation: {result['final_method']}",
      result['F1_next'],'',
      'F0 Formal32 remains frozen. Formal256 governs generalization claims; failed ablations must be reported. '
      'CHGNet force agreement is cross-potential evidence, not DFT verification or proof of nonoverlapping training data. '
      'The same CHGNet model supplies magnetic guardrails. '
      'G4 reverses the paired F0 reference trajectory force directions; it is not closed-loop antigradient descent on its own trajectory. '
      'Trust-region necessity is conditional on the selected uncapped comparator. '
      'F1 is only 16-pair P0; a GO does not replace F0 without a fresh next-round confirmation. '
      'No new neural network, no tuning, no budget-allocation method, and no thesis-body changes. DFT_VERIFIED=False.','']
    (ROOT/'final_report.md').write_text('\n'.join(lines));return result

def main():
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(ROOT/'pipeline_process.json',{'pid':os.getpid(),'start_epoch':time.time(),
      'big_lane':'A1 existing GPUs 5,6 -> A5 GPUs 5,6',
      'small_lane':'A3 existing smoke then full -> A4 -> B -> A6, GPU 7',
      'quality':'CPU jobs overlap next cohort generation; maximum two cohorts'})
    errors=[];futures=[]
    with ThreadPoolExecutor(max_workers=2) as cpu:
        def safely(code,fn):
            try:fn()
            except Exception as e:
                error={'track':code,'error':repr(e),'traceback':traceback.format_exc()}
                write_json(output(code)/'pipeline_failure.json',error);errors.append(error)
                print(json.dumps(error),flush=True)
        def finish_gpu(code,gpu):
            if gpu_evaluation(code,gpu):
                futures.append((code,cpu.submit(cpu_evaluation,code)))
        def big():
            ready=[]
            safely('A1',lambda:(wait_generated('A1'),ready.append(True)))
            if not ready:
                safely('A5',lambda:(_ for _ in ()).throw(RuntimeError('A1 resource release not confirmed; A5 not launched')))
                return
            safely('A1',lambda:finish_gpu('A1',5))
            safely('A5',lambda:(run('A5','generation',['run_generation','--track','A5','--gpus','5,6'],5),finish_gpu('A5',5)))
        def small():
            wait_generated('A3',1)
            for code in ['A3','A4','B','A6']:
                safely(code,lambda code=code:(run(code,'generation',['run_generation','--track',code,'--gpus','7'],7),finish_gpu(code,7)))
        with ThreadPoolExecutor(max_workers=2) as lanes:
            jobs=[lanes.submit(big),lanes.submit(small)]
            for j in jobs:j.result()
        for code,f in futures:
            try:f.result()
            except Exception as e:
                error={'track':code,'stage':'CPU evaluation','error':repr(e)}
                write_json(output(code)/'pipeline_failure.json',error);errors.append(error)
    result=final_report();write_json(ROOT/'pipeline_completion.json',{'errors':errors,'complete':result['complete'],'end_epoch':time.time()})
    subprocess.run([sys.executable,'-m','experiments.innovation2_finalization.status'],cwd=PROJECT,check=True)
    if errors or not result['complete']:raise RuntimeError('Incomplete pipeline; inspect retained failures')

if __name__=='__main__':main()
