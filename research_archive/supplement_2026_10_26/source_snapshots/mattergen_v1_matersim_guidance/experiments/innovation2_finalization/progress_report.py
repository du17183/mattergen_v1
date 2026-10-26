"""Operational progress/ETA only; never inspect generated scientific outcomes."""
from datetime import datetime,timezone
import json
import statistics
import time
from experiments.innovation2_finalization.protocol import ROOT,SPECS,output,write_json

def main():
    records={};durations=[];pair_intervals=[]
    for code,(_,_,n,methods) in SPECS.items():
        out=output(code);sets={};endpoints={}
        for m in methods:
            records_m=[]
            for p in (out/'generation'/m).glob('*/run_summary.json'):
                d=json.loads(p.read_text());records_m.append(d)
                if m!='POST':durations.append(float(d['elapsed_seconds']))
                if m==methods[-1]:endpoints.setdefault(str(d['physical_gpu']),[]).append(p.stat().st_mtime)
            sets[m]={int(d['seed']) for d in records_m}
        complete=set.intersection(*sets.values()) if sets else set()
        for times in endpoints.values():
            times.sort()
            if code=='A1':pair_intervals += [b-a for a,b in zip(times,times[1:])]
        records[code]={'n':n,'completed_pairs':len(complete),'completed_by_method':{m:len(s) for m,s in sets.items()},
          'decision':json.loads((out/'decision_summary.json').read_text()).get('status') if (out/'decision_summary.json').exists() else 'PENDING',
          'active_stage':json.loads((out/'active_stage.json').read_text()) if (out/'active_stage.json').exists() else None,
          'pipeline_failure':(out/'pipeline_failure.json').exists(),'generation_failures':(out/'generation_failures.json').exists()}
    sample=statistics.median(durations) if durations else 90.
    pair=statistics.median(pair_intervals) if pair_intervals else 2*sample+30.
    setup=max(0.,pair-2*sample)
    a1=(256-records['A1']['completed_pairs'])*pair/2
    a5=(64-records['A5']['completed_pairs'])*(4*sample+setup)/2
    small=sum((records[k]['n']-records[k]['completed_pairs'])*
              (sample*(len(SPECS[k][3])-(1 if k=='A6' else 0))+setup) for k in ['A3','A4','B','A6'])
    remain=max(a1+a5,small)/3600
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'tracks':records,
      'median_sampling_seconds':sample,'A1_observed_pair_interval_seconds':pair,
      'estimated_remaining_hours_range':[round(remain+.25,1),round(remain*1.15+1.,1)],
      'estimate_note':'Projection only; adds official-evaluation margin, assumes GPU 5/6 big lane and GPU 7 small lane; failures and unusual structures can extend it.',
      'registered_generation_draws':1120,'postprocess_only_structures':32,'scientific_outcomes_not_examined':True}
    write_json(ROOT/'progress_snapshot.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
