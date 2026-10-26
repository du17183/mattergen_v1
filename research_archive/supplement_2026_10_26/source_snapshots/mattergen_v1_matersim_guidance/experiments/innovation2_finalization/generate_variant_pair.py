"""Registered multi-arm sampling; each arm resets the exact original sampling RNG."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from hydra.utils import instantiate
if not hasattr(np,'math'):np.math=math
from experiments.innovation2_finalization.protocol import config,output,sha,write_json
from experiments.mattersim_late_force_guidance_formal32.generate_formal32_pair import (
    MODEL_ROOT,MATTERSIM,MODEL_SHA,MATTERSIM_SHA,ELEMENT_MASK_OVERRIDE)
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator,draw_samples_from_sampler

FROZEN='experiments.mattersim_late_force_guidance_p0.late_force_sampler.LateMatterSimForceGuidedSampler.from_pl_module'
TARGETS={
 'A3':{'G0':None,'G1':'experiments.innovation2_finalization.recording_sampler.RecordingFrozenSampler.from_pl_module',
       'G2':'random','G3':'shuffled','G4':'anti'},
 'A4':{'T0':None,'T1':FROZEN,'T2':'unbounded'},
 'B':{'F0':FROZEN,'F1':'experiments.closed_loop_force_guidance_p0.closed_loop_sampler.ClosedLoopForceSampler.from_pl_module'},
 'A5':{'C0':None,'A0':None,'B0':FROZEN,'AB':FROZEN},
 'A6':{'C0':None,'F0':FROZEN,'POST':'post'},
}

def post_refine(out,seed,frozen):
    from ase.io import read,write
    from mattersim.forcefield.potential import MatterSimCalculator
    from experiments.mattersim_late_force_guidance_p0.late_force_sampler import (
        _load_potential,bounded_cartesian_force_correction,cartesian_to_fractional_correction,
        position_correction_is_safe)
    base=out/'generation'/'C0'/str(seed)
    baseline=json.loads((base/'run_summary.json').read_text())
    f0=json.loads((out/'generation'/'F0'/str(seed)/'run_summary.json').read_text())
    budget=int(f0['actual_mattersim_calls'])
    dest=out/'generation'/'POST'/str(seed);partial=dest.with_name(dest.name+'.partial')
    if (dest/'run_summary.json').exists():return
    if dest.exists() or partial.exists():raise FileExistsError(partial)
    partial.mkdir(parents=True)
    start=time.perf_counter();potential=_load_potential(str(MATTERSIM),'cuda')
    calc=MatterSimCalculator(potential=potential,compute_stress=False,device='cuda')
    load=time.perf_counter()-start;calls=0;failed=0
    original=calc.calculate
    def counted(*a,**kw):
        nonlocal calls,failed
        calls+=1
        try:return original(*a,**kw)
        except BaseException:failed+=1;raise
    calc.calculate=counted
    atoms=read(base/'generated_crystals.extxyz');events=[];accepted=0
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    # One force evaluation per bounded descent attempt. Final scoring is a separate evaluation budget.
    for step in range(budget):
        row={'step':step,'accepted':False,'error':'','calls_before':calls,'calls_after':calls}
        try:
            with torch.enable_grad():
                atoms.calc=calc
                energy=float(atoms.get_potential_energy());force=np.asarray(atoms.get_forces(),dtype=float)
            if not np.isfinite(force).all() or not np.isfinite(energy):raise ValueError('nonfinite physics')
            delta=bounded_cartesian_force_correction(force,
              force_reference_ev_a=frozen['force_reference_ev_a'],nominal_cart_step_a=frozen['nominal_cartesian_step_a'],
              hard_cart_cap_a=frozen['hard_cartesian_cap_a'])
            frac=cartesian_to_fractional_correction(delta,atoms.cell.array)
            safe,dist=position_correction_is_safe(atoms,frac,frozen['min_periodic_distance_floor_a'])
            row.update(maxF_before_ev_a=float(np.linalg.norm(force,axis=1).max()),
              max_correction_a=float(np.linalg.norm(delta,axis=1).max()),min_candidate_distance_a=dist)
            if safe:
                atoms.set_scaled_positions(np.remainder(atoms.get_scaled_positions()+frac,1.))
                accepted+=1;row['accepted']=True
        except Exception as e:row['error']=f'{type(e).__name__}: {e}'
        row['calls_after']=calls;events.append(row)
        if calls>=budget:break
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert calls<=budget
    atoms.calc=None;write(partial/'generated_crystals.extxyz',atoms)
    summary=baseline | {'method':'POST','elapsed_seconds':baseline['elapsed_seconds']+elapsed,
      'end_to_end_seconds':baseline['end_to_end_seconds']+elapsed+load,'potential_load_seconds':load,
      'post_seconds':elapsed,'actual_mattersim_calls':calls,'failed_mattersim_calls':failed,
      'post_budget':budget,'post_accepted':accepted,'mattersim_guidance_eligible':0,
      'mattersim_guidance_accepted':0,'mattersim_guidance_fallbacks':0,
      'mattersim_guidance_seconds':elapsed,'peak_allocated_bytes':max(int(baseline['peak_allocated_bytes']),int(torch.cuda.max_memory_allocated())),
      'post_generated_from_C0':True,'post_source_sha256':sha(base/'generated_crystals.extxyz')}
    write_json(partial/'post_trace.json',events);write_json(partial/'run_summary.json',summary)
    os.replace(partial,dest)

def main():
    p=argparse.ArgumentParser();p.add_argument('--track',required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();c=config(args.track);out=output(args.track)
    assert args.seed in c['seeds']
    assert sha(MODEL_ROOT/'checkpoints/last.ckpt')==MODEL_SHA and sha(MATTERSIM)==MATTERSIM_SHA
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    g=CrystalGenerator(checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT),config_overrides=[ELEMENT_MASK_OVERRIDE]),
      batch_size=1,num_batches=1,properties_to_condition_on={'dft_mag_density':.2},
      diffusion_guidance_factor=2.,guidance_schedule='constant',seed=args.seed,deterministic=True,record_trajectories=False)
    g._configure_deterministic_mode();g.prepare();g.model.eval()
    for v in g.model.parameters():v.requires_grad_(False)
    for method in c['methods']:
        target=TARGETS[args.track][method]
        if target=='post':post_refine(out,args.seed,c['F0']);continue
        dest=out/'generation'/method/str(args.seed);partial=dest.with_name(dest.name+'.partial')
        if (dest/'run_summary.json').is_file():continue
        if dest.exists() or partial.exists():raise FileExistsError(partial)
        partial.mkdir(parents=True)
        adaptive=args.track=='A5' and method in ['A0','AB']
        g.guidance_schedule='adaptive' if adaptive else 'constant'
        sc=g.load_sampling_config(batch_size=1,num_batches=1)
        assert int(sc.sampler_partial.N)==1000 and int(sc.sampler_partial.n_steps_corrector)==1
        assert float(sc.sampler_partial.guidance_scale)==2.
        kw={'pl_module':g.model}
        if adaptive:
            a=c['adaptive_cfg']
            for name,value in a.items():
                if name!='source':setattr(sc.sampler_partial,name,value)
            kw.update(guidance_trace_path=str(partial/'adaptive_cfg_trace.csv'),sample_seed=args.seed)
        if target is not None:
            f=c['F0']
            kw.update(trace_path=str(partial/'late_force_trace.csv'),sample_seed=args.seed,
               mattersim_checkpoint=str(MATTERSIM),guidance_t_max=f['guidance_t_max'],
               force_reference_ev_a=f['force_reference_ev_a'],nominal_cart_step_a=f['nominal_cartesian_step_a'],
               hard_cart_cap_a=f['hard_cartesian_cap_a'],min_distance_floor_a=f['min_periodic_distance_floor_a'])
            if target in ['random','shuffled','anti','unbounded']:
                kw['mode']=target
                if target!='unbounded':kw['reference_corrections_path']=str(out/'generation'/'G1'/str(args.seed)/'reference_corrections.json')
                target='experiments.innovation2_finalization.experimental_sampler.ExperimentalPhysicsSampler.from_pl_module'
            if args.track=='B' and method=='F1':
                f1=c['F1']
                kw.update({k:f1[k] for k in ['epsilon_ev_a','radius_init_a','radius_min_a','radius_max_a','max_retries']})
            sc.sampler_partial._target_=target
        started=time.perf_counter();sampler=instantiate(sc.sampler_partial)(**kw)
        load=time.perf_counter()-started if target is not None else 0.
        callcount={'actual_mattersim_calls':0,'failed_mattersim_calls':0}
        if hasattr(sampler,'_calculator') and not hasattr(sampler,'_actual_calls'):
            original=sampler._calculator.calculate
            def counted(*a,**kw):
                callcount['actual_mattersim_calls']+=1
                try:return original(*a,**kw)
                except BaseException:
                    callcount['failed_mattersim_calls']+=1;raise
            sampler._calculator.calculate=counted
        g.seed=args.seed;g._seed_sampling_rngs();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter()
        structures=draw_samples_from_sampler(sampler=sampler,condition_loader=g.get_condition_loader(sc),
          properties_to_condition_on={'dft_mag_density':.2},output_path=partial,cfg=g.cfg,record_trajectories=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started;metrics=dict(sampler.sampling_metrics)
        assert int(metrics['mattergen_score_calls'])==2000
        if target is not None:assert metrics['mattersim_guidance_eligible']==20
        summary={'success':True,'track':args.track,'method':method,'seed':args.seed,
          'formula':structures[0].composition.reduced_formula,'num_atoms':len(structures[0]),
          'elapsed_seconds':elapsed,'potential_load_seconds':load,'end_to_end_seconds':elapsed+load,
          'physical_gpu':os.environ.get('FORMAL32_PHYSICAL_GPU'),'adaptive_cfg_used':adaptive,
          'guidance_schedule':g.guidance_schedule,'guidance_scale':2.,'target_dft_mag_density':.2,
          'sampling_steps':1000,'corrector_steps_per_timestep':1,'checkpoint_sha256':MODEL_SHA,
          'mattersim_checkpoint_sha256':MATTERSIM_SHA if target is not None else None,
          'position_force_guidance':target is not None,'cell_guidance':False,'stress_guidance':False,
          'atomic_guidance':False,'energy_guidance':False,'dft_verified':False,
          'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
          'config_sha256':sha(out/'config.yaml'),**callcount,**metrics}
        write_json(partial/'run_summary.json',summary);os.replace(partial,dest)
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
