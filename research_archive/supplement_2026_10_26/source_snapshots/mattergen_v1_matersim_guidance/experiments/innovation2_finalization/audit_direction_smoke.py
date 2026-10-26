"""Validate first registered direction pair without evaluating its scientific outcome."""
import csv
import json
import numpy as np
from experiments.innovation2_finalization.protocol import output,write_json

def main():
    out=output('A3');seed=741000
    traces={}
    for m in ['G1','G2','G3','G4']:
        path=out/'generation'/m/str(seed)
        with (path/'late_force_trace.csv').open() as f:traces[m]=list(csv.DictReader(f))
        summary=json.loads((path/'run_summary.json').read_text())
        assert summary['actual_mattersim_calls']==20 and summary['mattersim_guidance_accepted']==20
        assert summary['mattersim_guidance_fallbacks']==0 and len(traces[m])==20
    reference=json.loads((out/'generation/G1'/str(seed)/'reference_corrections.json').read_text())
    for i,r in enumerate(traces['G1']):
        v=np.asarray(reference[str(r['sampling_step'])]);assert np.linalg.norm(v.sum(0))<1e-10
        maximum=np.linalg.norm(v,axis=1).max()
        assert abs(maximum-float(r['max_cart_correction_a']))<1e-10
        for m in ['G2','G3','G4']:
            rr=traces[m][i]
            assert rr['sampling_step']==r['sampling_step'] and rr['t_norm']==r['t_norm']
            assert abs(float(rr['max_cart_correction_a'])-maximum)<1e-10
    result={'seed':seed,'AUDIT':'PASS','all_20_events_accepted':True,'actual_calls_each':20,
      'max_norm_and_timing_matched':True,'reference_zero_net_translation':True,
      'per_atom_norm_multiset':'checked inside each experimental proposal using immutable assertion',
      'scientific_outcome_not_examined':True,'no_resampling_or_tuning':True}
    write_json(out/'first_pair_implementation_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
