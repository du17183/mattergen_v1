"""Re-audit traces and accounting independently of historical summary verdicts."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
F0=PROJECT.with_name('mattergen_v1_matersim_guidance')
MINI=PROJECT/'experiments/final_innovation1_audit/minimal_ablation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())

def main():
    checks={};detail={}
    a3=F0/'experiments/mattersim_force_direction_ablation'
    t=pd.read_csv(a3/'guidance_trace.csv',float_precision='round_trip')
    for seed in range(741000,741032):
        ref=js(a3/'generation/G1'/str(seed)/'reference_corrections.json')
        base=t[(t.seed==seed)&(t.method=='G1')].sort_values('sampling_step');assert len(base)==20
        for m in ['G1','G2','G3','G4']:
            g=t[(t.seed==seed)&(t.method==m)].sort_values('sampling_step')
            assert len(g)==20 and g.accepted.all() and not g.fallback.any()
            np.testing.assert_array_equal(base.sampling_step,g.sampling_step)
            np.testing.assert_array_equal(base.t_norm,g.t_norm)
            np.testing.assert_allclose(base.max_cart_correction_a,g.max_cart_correction_a,atol=1e-10)
            for _,r in g.iterrows():
                v=np.asarray(ref[str(r.sampling_step)],float)
                assert np.linalg.norm(v.sum(0))<1e-10
                assert abs(np.linalg.norm(v,axis=1).max()-r.max_cart_correction_a)<1e-10
    checks['A3_all_32_pairs_event_time_norm_acceptance_match']=True
    detail['A3_vector_scope']='Full G1 reference vectors available. All-arm per-atom norm matching relied on immutable runtime assertions; control vectors not all separately archived. G4 reverses reference trajectory, not own current gradient.'
    a6=F0/'experiments/mattersim_equal_budget_post_generation'
    calls=[]
    for seed in range(745000,745032):
        base=a6/'generation/C0'/str(seed);post=a6/'generation/POST'/str(seed);f0=a6/'generation/F0'/str(seed)
        p=js(post/'run_summary.json');f=js(f0/'run_summary.json');events=js(post/'post_trace.json')
        assert p['post_source_sha256']==sha(base/'generated_crystals.extxyz')
        assert p['actual_mattersim_calls']==f['actual_mattersim_calls']==20
        assert p['failed_mattersim_calls']==f['failed_mattersim_calls']==0
        assert len(events)==20 and events[-1]['calls_after']==20
        assert sum(r['calls_after']-r['calls_before'] for r in events)==20
        assert all(r['max_correction_a']<=.005+1e-12 and r['accepted'] for r in events)
        calls.append({'seed':seed,'F0_actual_calls':20,'POST_actual_calls':20,'failed_calls':0})
    checks['A6_same_C0_hash_equal_20_actual_calls_all_attempts_included']=True
    pd.DataFrame(calls).to_csv(ROOT/'equal_budget_call_audit.csv',index=False)
    hist=js(F0/'experiments/innovation2_finalization/historical_seed_audit.json')
    assert not hist['historical_matches'] and not hist['scan_errors']
    checks['A1_A3_A4_A5_A6_preregistered_fresh_seeds']=True
    cfg=js(MINI/'preregistered_config.json')
    for p,h in cfg['source_sha256'].items():assert sha(p)==h,p
    checks['minimal_ablation_preregistered_source_unchanged']=True
    h=js(MINI/'historical_seed_audit.json');assert not h['prior_hits'];checks['minimal8_fresh_seed_scan']=True
    ranges={'I1':set(range(20000,20256)),'E3':set(range(40000,40256)),'F0diagnostic':set(range(710000,710032)),
      'F0P0':set(range(710032,710048)),'F0formal32':set(range(730000,730032)),
      'A1':set(range(740000,740256)),'A3':set(range(741000,741032)),'A4':set(range(742000,742032)),
      'F1':set(range(743000,743016)),'A5':set(range(744000,744064)),'A6':set(range(745000,745032)),
      'I1mini':set(cfg['seeds'])}
    import itertools
    for (ka,a),(kb,b) in itertools.combinations(ranges.items(),2):assert not a&b,(ka,kb)
    checks['all_final_study_cohorts_pairwise_disjoint']=True
    detail['intentional_reuse']='C0 shared within paired methods, A6 POST derived from its C0, and I1 Formal256 overlaps later E3 gate training seeds 20000-20063. A0 does not use the gate; E3 formal40000-40255 excludes gate training. Not all historical projects are independent of each other.'
    if (MINI/'completion.json').exists():
        summaries=[]
        for m in cfg['methods']:
            for seed in cfg['seeds']:
                p=MINI/'generation'/m/str(seed)
                run=js(p/'run_summary.json');assert run['success'] and run['mattergen_score_calls']==2000
                if m=='C0':continue
                t=pd.read_csv(p/'adaptive_cfg_trace.csv',float_precision='round_trip')
                assert len(t)==2000 and t.phase.value_counts().to_dict()=={'corrector':1000,'predictor':1000}
                reference=pd.read_csv(MINI/'generation/A0'/str(seed)/'adaptive_cfg_trace.csv',nrows=1,float_precision='round_trip')
                cols=['delta_cell','delta_pos','delta_atomic_numbers']
                np.testing.assert_array_equal(t.loc[0,cols].to_numpy(float),reference.loc[0,cols].to_numpy(float))
                state={'predictor':None,'corrector':None};clip=0;fall=0
                for _,r in t.iterrows():
                    delta=float(r.delta_pos) if m=='POS_ONLY' else (r.delta_cell+r.delta_pos+r.delta_atomic_numbers)/3
                    phase='predictor' if m=='SHARED_EMA' else r.phase
                    beta=0 if m=='NO_EMA' else .95
                    ema=delta if state[phase] is None else beta*state[phase]+(1-beta)*delta
                    state[phase]=ema;ratio=delta/(ema+1e-6);mult=1+.5*(ratio-1)
                    raw=2*mult
                    g=raw if m=='NO_CLIP' else min(max(2*min(max(mult,.25),4),0),5)
                    np.testing.assert_allclose([r.delta,r.ema,r.ratio,r.final_guidance],[delta,ema,ratio,g],atol=1e-10,rtol=1e-10)
                    clip+=int(abs(raw-g)>1e-12);fall+=int(pd.notna(r.fallback_reason))
                summaries.append({'method':m,'seed':seed,'events':len(t),'min_scale':t.final_guidance.min(),
                  'max_scale':t.final_guidance.max(),'scale_mean':t.final_guidance.mean(),'clipped_events':clip,'fallback_events':fall,
                  'predictor_scale_mean':t[t.phase=='predictor'].final_guidance.mean(),
                  'corrector_scale_mean':t[t.phase=='corrector'].final_guidance.mean()})
        pd.DataFrame(summaries).to_csv(ROOT/'innovation1_trace_audit.csv',index=False)
        checks['minimal8_all_48_generations_and_80000_control_decisions_reproduced']=True
        checks['minimal8_nonconstant_arms_initial_residuals_identical']=True
    else:checks['minimal8_all_48_generations_and_80000_control_decisions_reproduced']=False
    original_stash=js(ROOT/'resource_audit.json')['commands'][3]['stdout']
    import subprocess
    now=subprocess.check_output(['git','stash','list'],cwd=PROJECT,text=True)
    assert now==original_stash;checks['five_stashes_unchanged']=True
    result={'checks':checks,'details':detail,'all_required_pass':all(checks.values())}
    (ROOT/'evidence_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
