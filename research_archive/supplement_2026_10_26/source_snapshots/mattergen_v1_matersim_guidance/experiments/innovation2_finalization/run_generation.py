"""Run registered pairs serially per GPU, with restart-safe completion checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.innovation2_finalization.protocol import ROOT, PROJECT, CLOSED, config, output, write_json, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--track',required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();conf=config(args.track);out=output(args.track)
    gpus=[int(x) for x in args.gpus.split(',')]
    if any(g not in [5,6,7] for g in gpus):raise ValueError('Only audited-free GPU pool 5,6,7')
    seeds=conf['seeds'][:args.limit] if args.limit else conf['seeds']
    (out/'logs').mkdir(exist_ok=True,parents=True)
    failures=[]
    def worker(gpu,shard):
        for seed in shard:
            lock=out/'implementation_lock.json'
            if lock.exists():
                for path,expected in json.loads(lock.read_text())['sha256'].items():
                    if sha(path)!=expected:raise RuntimeError(f'Locked implementation changed: {path}')
            if all((out/'generation'/m/str(seed)/'run_summary.json').exists() for m in conf['methods']):
                continue
            env=dict(os.environ)
            cwd=CLOSED if args.track=='B' else PROJECT
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),FORMAL32_PHYSICAL_GPU=str(gpu),
              PYTHONPATH=str(cwd)+':'+str(PROJECT),TMPDIR=str(PROJECT/'.tmp'),
              XDG_CACHE_HOME=str(PROJECT/'.cache'),HF_HOME=str(PROJECT/'.cache/huggingface'),
              TORCH_HOME=str(PROJECT/'.cache/torch'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
              PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            script='generate_frozen_pair.py' if args.track=='A1' else 'generate_variant_pair.py'
            cmd=[sys.executable,str(ROOT/script),'--seed',str(seed)]
            if args.track!='A1':cmd += ['--track',args.track]
            log=out/'logs'/f'pair_{seed}_gpu_{gpu}.log'
            with log.open('a') as f:
                proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
                print(json.dumps({'event':'start','track':args.track,'seed':seed,'gpu':gpu,'pid':proc.pid}),flush=True)
                rc=proc.wait()
            print(json.dumps({'event':'complete','track':args.track,'seed':seed,'gpu':gpu,'return_code':rc}),flush=True)
            if rc:
                failures.append({'seed':seed,'gpu':gpu,'return_code':rc,'log':str(log)})
                write_json(out/'generation_failures.json',failures)
                # Preserve failed seed and partial outputs. Continue other registered pairs.
            subprocess.run([sys.executable,'-m','experiments.innovation2_finalization.status'],
              cwd=PROJECT,env=env,stdout=subprocess.DEVNULL,check=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs=[pool.submit(worker,g,seeds[i::len(gpus)]) for i,g in enumerate(gpus)]
        for job in jobs:job.result()
    if failures:raise RuntimeError(f'{len(failures)} failures retained')
    write_json(out/'generation_completion.json',{'success':True,'seeds':seeds,'methods':conf['methods']})

if __name__=='__main__':main()
