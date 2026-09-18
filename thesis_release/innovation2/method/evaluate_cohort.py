"""Apply unchanged frozen evaluators to registered cohorts, retaining all seeds."""
import argparse
import csv
import json
from pathlib import Path
from ase.io import read,write
from experiments.innovation2_finalization.protocol import config,output,write_json,sha

def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def prepare(code):
    c=config(code);out=output(code)
    if (out/'structures/manifest.json').exists():return
    dest=out/'structures';dest.mkdir(exist_ok=True)
    rows=[];traces=[];candidates=[];hashes={}
    for method in c['methods']:
        atoms=[]
        for seed in c['seeds']:
            run=out/'generation'/method/str(seed)
            s=json.loads((run/'run_summary.json').read_text())
            assert s['success'] is True and int(s['seed'])==seed and int(s['mattergen_score_calls'])==2000
            items=read(run/'generated_crystals.extxyz',index=':');assert len(items)==1
            a=items[0];a.info.update(sample_seed=seed,sample_index_within_seed=0,method=method)
            atoms.append(a);hashes[str(run/'generated_crystals.extxyz')]=sha(run/'generated_crystals.extxyz')
            calls=int(s.get('actual_mattersim_calls',0));failed=int(s.get('failed_mattersim_calls',0))
            if code=='A1' and method=='F0':
                call=json.loads((run/'force_call_accounting.json').read_text())
                calls=call['actual_force_model_calls'];failed=call['failed_force_model_calls']
            rows.append({'method':method,'seed':seed,'success':True,'formula':a.get_chemical_formula(),
              'num_atoms':len(a),'elapsed_seconds':float(s['elapsed_seconds']),
              'potential_load_seconds':float(s['potential_load_seconds']),
              'end_to_end_seconds':float(s['end_to_end_seconds']),'physical_gpu':s['physical_gpu'],
              'peak_allocated_bytes':s['peak_allocated_bytes'],'mattergen_score_calls':s['mattergen_score_calls'],
              'guidance_eligible':s.get('mattersim_guidance_eligible',0),
              'guidance_accepted':s.get('mattersim_guidance_accepted',0),
              'guidance_fallbacks':s.get('mattersim_guidance_fallbacks',0),
              'guidance_seconds':s.get('mattersim_guidance_seconds',0.),
              'actual_mattersim_calls':calls,'failed_mattersim_calls':failed,
              'initial_mattersim_calls':s.get('initial_mattersim_calls',calls),
              'candidate_mattersim_calls':s.get('candidate_mattersim_calls',0),
              'retry_mattersim_calls':s.get('retry_mattersim_calls',0),
              'failed_candidate_calls':s.get('failed_candidate_calls',0),
              'adaptive_cfg_used':s['adaptive_cfg_used'],'dft_verified':False})
            for name,target in [('late_force_trace.csv',traces),('candidate_trace.csv',candidates)]:
                if (run/name).exists():
                    with (run/name).open() as f:rr=list(csv.DictReader(f))
                    if name=='late_force_trace.csv':assert len(rr)==20
                    target.extend([r|{'method':method} for r in rr])
        write(dest/f'{method}_generated.extxyz',atoms)
    table(out/'generation_manifest.csv',rows)
    if traces:table(out/'guidance_trace.csv',traces)
    if candidates:table(out/'candidate_trace.csv',candidates)
    for alias,target in [('p0_structures',dest)]:
        p=out/alias
        if p.is_symlink():assert p.resolve()==target.resolve()
        elif p.exists():raise FileExistsError(p)
        else:p.symlink_to(target,target_is_directory=True)
    write_json(dest/'manifest.json',{'methods':c['methods'],'paired_seeds':c['seeds'],
        'n_per_method':len(c['seeds']),'all_registered_seeds_included':True,'source_sha256':hashes})

def main():
    p=argparse.ArgumentParser();p.add_argument('--track',required=True)
    p.add_argument('--stage',choices=['prepare','properties','relax','quality'],required=True)
    p.add_argument('--gpu',type=int,default=7);a=p.parse_args();c=config(a.track);out=output(a.track)
    if a.stage=='prepare':prepare(a.track);return
    if a.stage=='properties':
        if (out/'property_summary.json').exists():return
        from experiments.mattersim_late_force_guidance_p0 import evaluate_p0_properties as e
        e.ROOT=out;e.SEEDS=tuple(c['seeds']);e.METHODS=tuple(c['methods']);e.main()
        # Keep compatibility filenames in the new cohort; old cohorts are untouched.
        for old,new in [('p0_property_metrics.csv','property_metrics.csv'),('p0_property_summary.json','property_summary.json')]:
            (out/new).symlink_to(out/old)
    elif a.stage=='relax':
        from experiments.mattersim_late_force_guidance_p0 import run_p0_relaxation as e
        e.ROOT=out
        for m in c['methods']:
            summary=out/'p0_relaxation'/m/'relaxation_summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('success') is True:continue
            e.JOBS=((m,a.gpu),);e.main()
    else:
        from experiments.mattersim_late_force_guidance_p0 import run_p0_quality as e
        e.ROOT=out
        e.METHODS=tuple(m for m in c['methods'] if not (out/'p0_quality'/m/'official_detailed.json.gz').exists())
        if e.METHODS:e.main()
    print(json.dumps({'track':a.track,'stage':a.stage,'complete':True}),flush=True)

if __name__=='__main__':main()
