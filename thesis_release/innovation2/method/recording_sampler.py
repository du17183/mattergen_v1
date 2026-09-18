"""Record exact F0 correction vectors without changing its calculations."""
import json
import numpy as np
from experiments.mattersim_late_force_guidance_p0.late_force_sampler import (
    LateMatterSimForceGuidedSampler,bounded_cartesian_force_correction)

class RecordingFrozenSampler(LateMatterSimForceGuidedSampler):
    def __init__(self,**kw):
        super().__init__(**kw);self._reference_corrections={}

    def _evaluate_exact_score(self,x,t):
        before=len(self._rows)
        result=super()._evaluate_exact_score(x,t)
        if len(self._rows)>before:
            row=self._rows[-1]
            if row['accepted']:
                force=np.asarray(self._calculator.results['forces'],dtype=float)
                correction=bounded_cartesian_force_correction(force,
                   force_reference_ev_a=self._force_reference,nominal_cart_step_a=self._nominal_step,
                   hard_cart_cap_a=self._hard_cap)
            else:correction=np.zeros((len(x['pos']),3))
            self._reference_corrections[str(row['sampling_step'])]=correction.tolist()
        return result

    def _on_sampling_end(self,error):
        super()._on_sampling_end(error)
        if error is None:
            self._trace_path.with_name('reference_corrections.json').write_text(
                json.dumps(self._reference_corrections,indent=2)+'\n')
