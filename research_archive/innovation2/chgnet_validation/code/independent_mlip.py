"""Evaluate a preregistered independent CHGNet potential; never guide generation."""
import argparse
import csv
import json
import itertools
from pathlib import Path
import time
import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from experiments.innovation2_finalization.protocol import ROOT, F32, output, sha, write_json
from experiments.mattersim_late_force_guidance_p0.evaluate_p0_properties import CHGNet, CHGNET_PATH

EXPECTED='d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1'

def table(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compare(rows,base,method):
    a=sorted([r for r in rows if r['method']==base],key=lambda r:r['seed'])
    b=sorted([r for r in rows if r['method']==method],key=lambda r:r['seed'])
    assert [r['seed'] for r in a]==[r['seed'] for r in b]
    x=np.array([r['maxF_ev_a'] for r in a]);y=np.array([r['maxF_ev_a'] for r in b])
    indices=np.random.default_rng(20260915).integers(0,len(x),size=(20000,len(x)))
    absolute=x-y;boot=absolute[indices].mean(1)
    relative=boot/x[indices].mean(1)
    ci=np.quantile(relative,[.025,.975]);reduction=float((x.mean()-y.mean())/x.mean())
    wins=int((y<x-1e-12).sum());losses=int((y>x+1e-12).sum())
    return {'baseline':base,'method':method,'n':len(x),'baseline_mean':float(x.mean()),
      'method_mean':float(y.mean()),'mean_relative_reduction':reduction,
      'relative_bootstrap_ci95':ci.tolist(),'absolute_improvement':float(absolute.mean()),
      'absolute_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
      'wins':wins,'ties':len(x)-wins-losses,'losses':losses,
      'status':'PASS' if reduction>0 and wins>losses else ('FAIL' if ci[1]<0 else 'INCONCLUSIVE'),
      'ci_excludes_zero_positive':bool(ci[0]>0),'bootstrap_resamples':20000,
      'formal256_evaluation_trigger':reduction>0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if sha(CHGNET_PATH)!=EXPECTED:raise RuntimeError('Independent checkpoint changed')
    if args.cohort=='Formal32':source=F32;methods=['C0','F0']
    else:
        from experiments.innovation2_finalization.protocol import config
        source=output(args.cohort);methods=config(args.cohort)['methods']
    out=ROOT/'independent_mlip'/args.cohort
    if (out/'decision_summary.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    selection=ROOT/'independent_mlip_selection.md'
    write_json(out/'evaluator_lock.json',{'checkpoint':str(CHGNET_PATH),'sha256':EXPECTED,
      'selection_sha256':sha(selection),'model_version':'0.3.0','selection_before_inference':True})
    model=CHGNet.from_file(CHGNET_PATH).to(args.device);model.eval()
    rows=[];raw=[];coverage=set();started=time.perf_counter()
    for method in methods:
        atoms=read(source/'structures'/f'{method}_generated.extxyz',index=':')
        structures=[AseAtomsAdaptor.get_structure(a) for a in atoms]
        for a in atoms:
            coverage.update(int(z) for z in a.numbers)
            if any(int(z)<1 or int(z)>94 for z in a.numbers):raise ValueError('Unsupported atomic number')
        preds=model.predict_structure(structures,task='ef',batch_size=16)
        if isinstance(preds,dict):preds=[preds]
        assert len(preds)==len(atoms)
        for a,pr in zip(atoms,preds):
            force=np.asarray(pr['f'],dtype=float);norms=np.linalg.norm(force,axis=1)
            energy=float(pr['e']);assert np.isfinite(force).all() and np.isfinite(energy)
            seed=int(a.info['sample_seed'])
            rows.append({'method':method,'seed':seed,'formula':a.get_chemical_formula(),
              'num_atoms':len(a),'energy_ev_atom':energy,'energy_total_ev':energy*len(a),
              'mean_force_ev_a':float(norms.mean()),'maxF_ev_a':float(norms.max()),
              'evaluator':'CHGNet_0.3.0','surrogate_property_eval':True,'dft_verified':False})
            raw.append({'method':method,'seed':seed,'forces_ev_a':force.tolist()})
        print(json.dumps({'method_complete':method,'n':len(atoms)}),flush=True)
    table(out/'independent_per_structure.csv',rows);write_json(out/'raw_forces.json',raw)
    summaries=[]
    for m in methods:
        x=np.array([r['maxF_ev_a'] for r in rows if r['method']==m])
        summaries.append({'method':m,'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
         'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.9)),
         'p95':float(np.quantile(x,.95)),'max':float(x.max())})
    table(out/'independent_summary.csv',summaries)
    comparisons=[compare(rows,a,b) for a,b in itertools.combinations(methods,2)]
    write_json(out/'paired_bootstrap.json',comparisons)
    d=comparisons[0] | {'cohort':args.cohort,'elements_Z':sorted(coverage),
      'coverage_fraction':1.0,'elapsed_seconds':time.perf_counter()-started,'DFT_VERIFIED':False}
    write_json(out/'decision_summary.json',d)
    if args.cohort in ['Formal32','A1']:write_json(ROOT/'independent_mlip'/'decision_summary.json',d)
    (out/'final_report.md').write_text('# Independent CHGNet force evaluation\n\n'
      +json.dumps(d,indent=2)+'\n\nThis evaluator is independent of guidance, but previously supplied magnetic guardrails. '
      'Positive central effect is not statistical confirmation unless the interval excludes zero. DFT_VERIFIED=False.\n')
    print(json.dumps(d,indent=2))

if __name__=='__main__':main()
