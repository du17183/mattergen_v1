"""Frozen eight-pair component diagnostics, not a replacement formal experiment.

All changes are runtime-only control ablations; original source/configs are untouched.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

PROJECT=Path('/mnt/datasets-livsyn/dxl/mattergen_v1')
F0=Path('/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance')
ROOT=Path(__file__).resolve().parent/'minimal_ablation'
FINAL=PROJECT/'experiments/final_thesis_results'
PY='/mnt/datasets-livsyn/dxl/alm/.venv/bin/python'
sys.path[:0]=[str(PROJECT),str(F0)]
METHODS=['C0','A0','POS_ONLY','NO_EMA','SHARED_EMA','NO_CLIP']
SEEDS=list(range(746000,746008))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def config():
    c=json.loads((ROOT/'preregistered_config.json').read_text())
    for p,h in c['source_sha256'].items(): assert sha(p)==h, p
    assert c['seeds']==SEEDS and c['methods']==METHODS
    return c

def register():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'preregistered_config.json').exists()
    trees=[Path(s[9:]) for s in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=PROJECT,text=True).splitlines() if s.startswith('worktree ')]
    scopes=[]; hits=[]; count=0
    for wt in trees:
        for name in ['experiments','diagnostics','research','reports','thesis']:
            p=wt/name
            if not p.exists(): continue
            scopes.append(str(p))
            paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(p)],text=True).splitlines()
            count+=len(paths)
            cmd=['rg','--hidden','--no-ignore','-l',r'(^|[^0-9])74600[0-7]([^0-9]|$)']
            for suffix in ['json','jsonl','csv','yaml','yml','txt','md','py','sh']:cmd+=['-g','*.'+suffix]
            result=subprocess.run(cmd+[str(p)],text=True,capture_output=True)
            assert result.returncode in (0,1),result.stderr
            hits += [x for x in result.stdout.splitlines() if Path(__file__).resolve()!=Path(x) and ROOT not in Path(x).parents and FINAL not in Path(x).parents]
            hits += [x for x in paths if any(str(s) in Path(x).parts for s in SEEDS) and ROOT not in Path(x).parents]
    audit={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scopes':scopes,'files_enumerated':count,'prior_hits':hits,'seeds':SEEDS}
    save(ROOT/'historical_seed_audit.json',audit);assert not hits,hits
    for n in ['logs','runtime_tmp']: (ROOT/n).mkdir(exist_ok=True)
    # Central log location is also used by the unchanged evaluation wrappers.
    (ROOT/'logs').rmdir(); (ROOT/'logs').symlink_to(FINAL/'logs',target_is_directory=True)
    files=[Path(__file__),PROJECT/'mattergen/diffusion/sampling/guidance_schedule.py',PROJECT/'mattergen/diffusion/sampling/classifier_free_guidance.py']
    files += [F0/'experiments/mattersim_late_force_guidance_p0'/n for n in ['evaluate_p0_properties.py','run_p0_relaxation.py','run_p0_quality.py','analyze_p0.py']]
    cfg={'registered_utc':audit['time'],'seeds':SEEDS,'methods':METHODS,'target':0.1,'tau':0.01,
         'guidance_scale':2.0,'alpha':0.5,'beta':0.95,'eps':1e-6,'clip':[0.,5.],
         'multiplier_clip':[0.25,4.0],'steps':1000,'corrector_steps':1,'batch_size':1,
         'source_sha256':{str(p):sha(p) for p in files},
         'controls':{'POS_ONLY':'Use only position RMS; same global scale still applied to every field.',
          'NO_EMA':'beta=0, remove temporal smoothing, retain current residual and epsilon.',
          'SHARED_EMA':'One EMA for alternating predictor/corrector observations.',
          'NO_CLIP':'Remove multiplier and final-scale clipping only; retain finite fallback.'},
         'endpoint':'Same Ehull, Stable and NUS as Innovation1; diagnostic no GO threshold, no retuning.',
         'bootstrap':{'resamples':20000,'seed':20260915,'method':'paired percentile, seed unit, all eight retained'},
         'limitations':'n=8 supplementary component diagnostics; cannot establish broad component superiority or replace Formal256.'}
    save(ROOT/'preregistered_config.json',cfg);save(ROOT/'seeds.json',{'seeds':SEEDS,'historical_overlap':0})
    print(json.dumps(cfg,indent=2))

def worker(seed):
    c=config();assert seed in SEEDS
    import math, numpy as np, torch
    if not hasattr(np,'math'):np.math=math
    from hydra.utils import instantiate
    from dataclasses import replace
    from mattergen.diffusion.sampling.guidance_schedule import GuidanceController
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.generator import CrystalGenerator,draw_samples_from_sampler
    from experiments.mattersim_late_force_guidance_formal32.generate_formal32_pair import MODEL_ROOT,MODEL_SHA,ELEMENT_MASK_OVERRIDE
    assert sha(MODEL_ROOT/'checkpoints/last.ckpt')==MODEL_SHA
    class Control(GuidanceController):
        def evaluate(self,*,progress,phase,field_deltas,residual_error=None):
            original_phase=phase
            if self.mode=='POS_ONLY':field_deltas={'pos':field_deltas['pos']}
            if self.mode=='SHARED_EMA':phase='predictor'
            d=super().evaluate(progress=progress,phase=phase,field_deltas=field_deltas,residual_error=residual_error)
            if self.mode=='NO_CLIP' and d.ratio is not None:
                mult=1+self.adaptive_alpha*(d.ratio-1)
                raw=d.stage_guidance*mult
                if math.isfinite(raw):d=replace(d,adaptive_multiplier=mult,final_guidance=raw)
            return d
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    g=CrystalGenerator(checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT),config_overrides=[ELEMENT_MASK_OVERRIDE]),
        batch_size=1,num_batches=1,properties_to_condition_on={'dft_mag_density':c['target']},
        diffusion_guidance_factor=2.,guidance_schedule='constant',seed=seed,deterministic=True,record_trajectories=False)
    g._configure_deterministic_mode();g.prepare();g.model.eval()
    for p in g.model.parameters():p.requires_grad_(False)
    for method in METHODS:
        dest=ROOT/'generation'/method/str(seed);partial=dest.with_name(dest.name+'.partial')
        if (dest/'run_summary.json').exists():continue
        if dest.exists() or partial.exists():raise FileExistsError(partial)
        partial.mkdir(parents=True)
        g.guidance_schedule='constant' if method=='C0' else 'adaptive'
        sc=g.load_sampling_config(batch_size=1,num_batches=1)
        assert int(sc.sampler_partial.N)==1000 and int(sc.sampler_partial.n_steps_corrector)==1
        kw={'pl_module':g.model,'sample_seed':seed}
        if method!='C0':kw['guidance_trace_path']=str(partial/'adaptive_cfg_trace.csv')
        sampler=instantiate(sc.sampler_partial)(**kw)
        if method not in ['C0','A0']:
            ctrl=Control(schedule='adaptive',base_guidance=2.,adaptive_alpha=.5,adaptive_ema=0. if method=='NO_EMA' else .95,
                         adaptive_eps=1e-6,min_scale=0.,max_scale=5.)
            ctrl.mode=method;sampler._guidance_controller=ctrl
        g.seed=seed;g._seed_sampling_rngs();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        structures=draw_samples_from_sampler(sampler=sampler,condition_loader=g.get_condition_loader(sc),
            properties_to_condition_on={'dft_mag_density':c['target']},output_path=partial,cfg=g.cfg,record_trajectories=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t;metrics=dict(sampler.sampling_metrics)
        assert metrics['mattergen_score_calls']==2000
        row={'method':method,'seed':seed,'success':True,'formula':structures[0].composition.reduced_formula,'num_atoms':len(structures[0]),
             'elapsed_seconds':elapsed,'end_to_end_seconds':elapsed,'potential_load_seconds':0.,
             'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
             'guidance_eligible':0,'guidance_accepted':0,'guidance_fallbacks':0,'guidance_seconds':0.,
             'checkpoint_sha256':MODEL_SHA,'source_config_sha256':sha(ROOT/'preregistered_config.json'),**metrics}
        save(partial/'run_summary.json',row);os.replace(partial,dest);print(json.dumps(row),flush=True)

def stage(name):
    c=config()
    if name=='prepare':
        from ase.io import read,write
        dest=ROOT/'structures';dest.mkdir(exist_ok=True);rows=[];hashes={}
        for m in METHODS:
            atoms=[]
            for seed in SEEDS:
                p=ROOT/'generation'/m/str(seed);r=json.loads((p/'run_summary.json').read_text());assert r['success']
                a=read(p/'generated_crystals.extxyz');a.info.update(sample_seed=seed,sample_index_within_seed=0,method=m)
                atoms.append(a);rows.append(r);hashes[str(p/'generated_crystals.extxyz')]=sha(p/'generated_crystals.extxyz')
            write(dest/f'{m}_generated.extxyz',atoms)
        table(ROOT/'generation_manifest.csv',rows);save(dest/'manifest.json',{'seeds':SEEDS,'methods':METHODS,'source_sha256':hashes})
        if not (ROOT/'p0_structures').exists():(ROOT/'p0_structures').symlink_to(dest,target_is_directory=True)
    elif name=='properties':
        from experiments.mattersim_late_force_guidance_p0 import evaluate_p0_properties as e
        e.ROOT=ROOT;e.SEEDS=tuple(SEEDS);e.METHODS=tuple(METHODS);e.TARGET=c['target'];e.main()
    elif name=='relax':
        from experiments.mattersim_late_force_guidance_p0 import run_p0_relaxation as e
        e.ROOT=ROOT
        # Each wave assigns at most one method to each initially reserved GPU.
        for start in [0,3]:
            e.JOBS=tuple((m,5+i) for i,m in enumerate(METHODS[start:start+3]) if not (ROOT/'p0_relaxation'/m/'relaxation_summary.json').exists())
            e.main()
    elif name=='quality':
        from experiments.mattersim_late_force_guidance_p0 import run_p0_quality as e
        e.ROOT=ROOT;e.METHODS=tuple(m for m in METHODS if not (ROOT/'p0_quality'/m/'official_detailed.json.gz').exists());e.main()
    elif name=='analyze':
        from experiments.mattersim_late_force_guidance_p0 import analyze_p0 as e
        e.ROOT=ROOT;e.SEEDS=tuple(SEEDS);e.METHODS=tuple(METHODS)
        gen=e.read_csv(ROOT/'generation_manifest.csv');prop=e.read_csv(ROOT/'p0_property_metrics.csv')
        rows=[r for m in METHODS for r in e.load_method(m,gen,prop)]
        table(ROOT/'per_structure_metrics.csv',rows);table(ROOT/'quality_metrics.csv',[e.summarize(m,[r for r in rows if r['method']==m]) for m in METHODS])
        save(ROOT/'completion.json',{'complete':True,'n':len(rows),'all_seeds_retained':True,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

def pipeline():
    c=config();env=dict(os.environ);env.update(PYTHONPATH=str(PROJECT)+':'+str(F0),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    def execute(args,gpu,label):
        e=env|{'CUDA_VISIBLE_DEVICES':str(gpu)};log=FINAL/'logs'/('inno1_'+label+'.log')
        with log.open('a') as f:
            result=subprocess.run([PY,str(Path(__file__).resolve())]+args,cwd=F0,env=e,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{label}: exit {result.returncode}; {log}')
    save(ROOT/'pipeline_status.json',{'status':'RUNNING','stage':'generation','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    try:
        def shard(gpu,seeds):
            for seed in seeds: execute(['--worker',str(seed)],gpu,f'seed_{seed}')
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(shard,gpu,SEEDS[i::3]) for i,gpu in enumerate([5,6,7])]
            for job in jobs:job.result()
        for name in ['prepare','properties','relax','quality','analyze']:
            save(ROOT/'pipeline_status.json',{'status':'RUNNING','stage':name})
            execute(['--stage',name],5,name)
        save(ROOT/'pipeline_status.json',{'status':'COMPLETED','end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    except BaseException as e:
        save(ROOT/'pipeline_status.json',{'status':'FAILED','error':str(e)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--stage');p.add_argument('--pipeline',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.worker is not None:worker(a.worker)
    elif a.stage:stage(a.stage)
    elif a.pipeline:pipeline()
