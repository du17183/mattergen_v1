"""Experimental ablation plumbing; frozen F0 source is never edited."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.mattersim_late_force_guidance_p0.late_force_sampler import (
    LateMatterSimForceGuidedSampler, bounded_cartesian_force_correction,
    cartesian_to_fractional_correction, position_correction_is_safe, _minimum_distance)
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector

def randomized_closed_vectors(reference, rng):
    """Random directions with the exact norm multiset and zero net displacement.

    Alternating projections preserve each labelled norm and center the vectors.
    If slow convergence occurs, a common random orthogonal rotation is an exact
    zero-sum, norm-preserving fallback; the fallback is recorded.
    """
    reference=np.asarray(reference,dtype=float)
    norms=np.linalg.norm(reference,axis=1)
    if norms.max()<1e-15:return np.zeros_like(reference),'zero_reference'
    v=rng.normal(size=reference.shape)
    for _ in range(2000):
        v-=v.mean(axis=0)
        v*=np.divide(norms,np.maximum(np.linalg.norm(v,axis=1),1e-30))[:,None]
        if np.linalg.norm(v.sum(axis=0))<1e-11:
            return v,'independent_random_then_norm_and_translation_projection'
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    return reference@q,'common_rotation_fallback'

class ExperimentalPhysicsSampler(LateMatterSimForceGuidedSampler):
    def __init__(self,*,mode,reference_corrections_path=None,**kw):
        super().__init__(**kw)
        self.mode=mode
        self._reference=(None if reference_corrections_path is None else
                         json.loads(Path(reference_corrections_path).read_text()))
        self._actual_calls=0;self._failed_calls=0;self._initial_calls=0
        self._candidate_calls=0;self._retry_calls=0;self._candidate_failures=0
        self._candidate_rows=[];self._corrections=[];self._call_category='initial'
        original=self._calculator.calculate
        def counted(*args,**kwargs):
            self._actual_calls+=1
            if self._call_category=='initial':self._initial_calls+=1
            elif self._call_category=='candidate':self._candidate_calls+=1
            else:self._retry_calls+=1
            try:return original(*args,**kwargs)
            except BaseException:
                self._failed_calls+=1
                if self._call_category!='initial':self._candidate_failures+=1
                raise
        self._calculator.calculate=counted

    def _physics(self,atoms,category='initial'):
        self._call_category=category
        with torch.enable_grad():
            atoms.calc=self._calculator
            energy=float(atoms.get_potential_energy())
            force=np.asarray(atoms.get_forces(),dtype=float)
        if not np.isfinite(force).all() or not np.isfinite(energy):
            raise FloatingPointError('nonfinite potential evaluation')
        return energy,force

    @property
    def sampling_metrics(self):
        d=dict(super().sampling_metrics)
        d.update(actual_mattersim_calls=self._actual_calls,failed_mattersim_calls=self._failed_calls,
          initial_mattersim_calls=self._initial_calls,candidate_mattersim_calls=self._candidate_calls,
          retry_mattersim_calls=self._retry_calls,failed_candidate_calls=self._candidate_failures)
        return d

    def _propose(self,atoms,cell,force,row):
        bounded=bounded_cartesian_force_correction(force,force_reference_ev_a=self._force_reference,
              nominal_cart_step_a=self._nominal_step,hard_cart_cap_a=self._hard_cap)
        if self.mode=='unbounded':
            return (force-force.mean(axis=0))*self._nominal_step/self._force_reference
        if self.mode in ['random','shuffled','anti']:
            reference=np.asarray(self._reference[str(row['sampling_step'])],dtype=float)
            if reference.shape!=bounded.shape:raise ValueError('reference shape differs from paired force arm')
            rng=np.random.default_rng(np.random.SeedSequence([self._seed,row['sampling_step'],417]))
            if self.mode=='random':
                correction,kind=randomized_closed_vectors(reference,rng)
                row['random_direction_construction']=kind
            elif self.mode=='shuffled':
                # Cyclic nonzero permutation ensures no unchanged atom indices for N>1.
                shift=int(rng.integers(1,len(reference))) if len(reference)>1 else 0
                correction=np.roll(reference,shift,axis=0)
            else:correction=-reference
            if not np.allclose(np.sort(np.linalg.norm(correction,axis=1)),
                               np.sort(np.linalg.norm(reference,axis=1)),rtol=0,atol=1e-10):
                raise AssertionError('direction-only control changed correction norm distribution')
            return correction
        return bounded

    def _evaluate_exact_score(self,x,t):
        score=GuidedPredictorCorrector._evaluate_exact_score(self,x,t)
        context=self.sampling_context;model_t=float(t[0].detach().cpu())
        if context.get('phase')!='predictor' or model_t>self._t_max+1e-8:return score
        self._eligible+=1;started=time.perf_counter()
        row={'seed':self._seed,'sampling_step':int(context.get('sampling_step',-1)),
          't_norm':model_t,'phase':'predictor','mode':self.mode,'accepted':False,'fallback':True,
          'fallback_reason':'','energy_ev':'','mean_force_ev_a':'','max_force_ev_a':'',
          'force_reference_ev_a':self._force_reference,'nominal_cart_step_a':self._nominal_step,
          'hard_cart_cap_a':self._hard_cap,'max_cart_correction_a':0.,'score_coeff_min':'',
          'min_distance_before_a':'','min_distance_after_a':'','random_direction_construction':''}
        try:
            if x.get_batch_size()!=1:raise RuntimeError('batch_size must be 1')
            atoms,cell=self._clean_atoms(x=x,score=score,t=t)
            row['min_distance_before_a']=_minimum_distance(atoms)
            if not np.isfinite(cell).all() or np.linalg.det(cell)<=0:raise ValueError('invalid clean cell')
            energy,force=self._physics(atoms);norms=np.linalg.norm(force,axis=1)
            row.update(energy_ev=energy,mean_force_ev_a=float(norms.mean()),max_force_ev_a=float(norms.max()))
            correction=self._propose(atoms,cell,force,row)
            if correction is None:raise ValueError('closed_loop_no_accepted_candidate')
            frac=cartesian_to_fractional_correction(correction,cell)
            safe,min_after=position_correction_is_safe(atoms,frac,self._min_distance_floor)
            row.update(max_cart_correction_a=float(np.linalg.norm(correction,axis=1).max()),min_distance_after_a=min_after)
            if not safe:raise ValueError('minimum_distance_safety_rejection')
            predictor=self._predictors['pos'];dt=torch.full_like(t,-1./self.N)
            _,coeff,_=predictor._get_coeffs(x=x['pos'],t=t,dt=dt,batch_idx=x.get_batch_idx('pos'),batch=x)
            row['score_coeff_min']=float(coeff.abs().min().detach().cpu())
            if not bool(torch.isfinite(coeff).all()) or bool((coeff.abs()<1e-12).any()):raise ValueError('invalid score coefficient')
            delta=torch.as_tensor(frac,device=score['pos'].device,dtype=score['pos'].dtype)/coeff
            if not bool(torch.isfinite(delta).all()):raise ValueError('nonfinite score correction')
            score=score.replace(pos=score['pos']+delta)
            row.update(accepted=True,fallback=False);self._accepted+=1
            self._corrections.append({'sampling_step':row['sampling_step'],'correction_cart_a':correction.tolist()})
        except Exception as e:
            row['fallback_reason']=f'{type(e).__name__}: {e}'[:300];self._fallbacks+=1
        elapsed=time.perf_counter()-started;self._physics_seconds+=elapsed
        row.update(physics_seconds=elapsed,actual_mattersim_calls_cumulative=self._actual_calls)
        self._rows.append(row)
        return score

    def _on_sampling_end(self,error):
        GuidedPredictorCorrector._on_sampling_end(self,error)
        if error is not None:return
        keys=list(dict.fromkeys(k for row in self._rows for k in row))
        with self._trace_path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(self._rows)
        if self._candidate_rows:
            with self._trace_path.with_name('candidate_trace.csv').open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(self._candidate_rows[0]));w.writeheader();w.writerows(self._candidate_rows)
