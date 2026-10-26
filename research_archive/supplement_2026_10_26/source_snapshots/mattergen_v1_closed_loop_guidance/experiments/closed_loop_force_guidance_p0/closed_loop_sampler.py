"""F1: clean-state residual-force acceptance; only shrink the frozen F0 proposal."""
import numpy as np
from experiments.innovation2_finalization.experimental_sampler import ExperimentalPhysicsSampler
from experiments.mattersim_late_force_guidance_p0.late_force_sampler import (
    bounded_cartesian_force_correction, cartesian_to_fractional_correction,
    position_correction_is_safe)

class ClosedLoopForceSampler(ExperimentalPhysicsSampler):
    def __init__(self,*,epsilon_ev_a=1e-4,radius_init_a=.005,radius_min_a=.00125,
                 radius_max_a=.005,max_retries=2,**kw):
        super().__init__(mode='closed_loop',**kw)
        self._epsilon=float(epsilon_ev_a);self._radius=float(radius_init_a)
        self._radius_min=float(radius_min_a);self._radius_max=float(radius_max_a)
        self._max_retries=int(max_retries)
        assert self._epsilon==1e-4 and self._radius==.005
        assert self._radius_min==.00125 and self._radius_max==.005 and self._max_retries==2

    def _propose(self,atoms,cell,force,row):
        before=float(np.linalg.norm(force,axis=1).max())
        row['closed_loop_radius_before_a']=self._radius
        for attempt in range(self._max_retries+1):
            radius=self._radius
            correction=bounded_cartesian_force_correction(force,
              force_reference_ev_a=self._force_reference,nominal_cart_step_a=radius,
              hard_cart_cap_a=self._hard_cap)
            frac=cartesian_to_fractional_correction(correction,cell)
            safe,min_after=position_correction_is_safe(atoms,frac,self._min_distance_floor)
            record={'seed':self._seed,'sampling_step':row['sampling_step'],'t_norm':row['t_norm'],
              'attempt':attempt,'radius_a':radius,'before_maxF_ev_a':before,
              'after_maxF_ev_a':'','after_mean_force_ev_a':'','after_energy_ev':'',
              'safe':safe,'min_distance_a':min_after,'accepted':False,'error':'',
              'actual_calls_before':self._actual_calls,'actual_calls_after':self._actual_calls}
            if safe:
                candidate=atoms.copy()
                candidate.set_scaled_positions(np.remainder(candidate.get_scaled_positions()+frac,1.))
                try:
                    energy,after_force=self._physics(candidate,'candidate' if attempt==0 else 'retry')
                    norms=np.linalg.norm(after_force,axis=1);after=float(norms.max())
                    record.update(after_maxF_ev_a=after,after_mean_force_ev_a=float(norms.mean()),
                                  after_energy_ev=energy,accepted=after<before-self._epsilon)
                except Exception as e:record['error']=f'{type(e).__name__}: {e}'[:250]
            record['actual_calls_after']=self._actual_calls
            self._candidate_rows.append(record)
            if record['accepted']:
                row.update(closed_loop_radius_after_a=self._radius,
                  accepted_candidate_maxF_ev_a=record['after_maxF_ev_a'],candidate_attempts=attempt+1)
                return correction
            if self._radius<=self._radius_min:break
            self._radius=max(self._radius_min,self._radius*.5)
        row.update(closed_loop_radius_after_a=self._radius,candidate_attempts=attempt+1)
        return None
