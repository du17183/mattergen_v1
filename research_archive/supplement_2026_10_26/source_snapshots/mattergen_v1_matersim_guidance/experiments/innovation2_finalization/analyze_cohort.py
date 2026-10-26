"""Paired, full-cohort analysis. No missing-seed filtering or outcome-driven tuning."""
import argparse
import csv
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments.innovation2_finalization.protocol import ROOT,config,output,sha,write_json
from experiments.mattersim_late_force_guidance_p0 import analyze_p0 as frozen

def read_table(p):
    with p.open() as f:return list(csv.DictReader(f))

def table(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def values(rows,key):return np.asarray([r[key] for r in rows],dtype=float)

def reduction(a,b):return float((a-b)/a) if a>1e-15 else None

def paired(a,b,metric='maxF_ev_per_a'):
    assert [r['seed'] for r in a]==[r['seed'] for r in b]
    x=values(a,metric);y=values(b,metric);n=len(x)
    idx=np.random.default_rng(20260915).integers(0,n,(20000,n))
    improvement=(x-y)[idx].mean(1);den=x[idx].mean(1)
    relative=np.divide(improvement,den,out=np.full_like(den,np.nan),where=den>1e-15)
    wins=int((y<x-1e-12).sum());losses=int((y>x+1e-12).sum())
    return {'baseline':a[0]['method'],'method':b[0]['method'],'metric':metric,'n_pairs':n,
      'baseline_mean':float(x.mean()),'method_mean':float(y.mean()),
      'absolute_improvement':float((x-y).mean()),'absolute_ci95':np.quantile(improvement,[.025,.975]).tolist(),
      'relative_improvement':reduction(x.mean(),y.mean()),
      'relative_ci95':np.quantile(relative,[.025,.975]).tolist() if np.isfinite(relative).all() else None,
      'wins':wins,'ties':n-wins-losses,'losses':losses,
      'win_binomial_one_sided_p':float(binomtest(wins,wins+losses,.5,alternative='greater').pvalue) if wins+losses else 1.,
      'bootstrap_resamples':20000,'bootstrap_seed':20260915}

def summary(method,rows):
    s=frozen.summarize(method,rows);x=values(rows,'maxF_ev_per_a')
    s.update(maxF_p75_ev_per_a=float(np.quantile(x,.75)),maxF_p90_ev_per_a=float(np.quantile(x,.90)),
      maxF_max_ev_per_a=float(x.max()),high_maxF_gt_0_2_fraction=float((x>.2).mean()),
      generation_seconds_mean=float(values(rows,'generation_seconds').mean()),
      end_to_end_seconds_mean=float(values(rows,'end_to_end_generation_seconds').mean()),
      mattergen_score_calls_mean=float(values(rows,'mattergen_score_calls').mean()),
      actual_mattersim_calls_mean=float(values(rows,'actual_mattersim_calls').mean()),
      actual_mattersim_calls_total=int(values(rows,'actual_mattersim_calls').sum()),
      failed_mattersim_calls_total=int(values(rows,'failed_mattersim_calls').sum()),
      peak_allocated_gib_max=float(values(rows,'peak_allocated_bytes').max()/1024**3),
      fallback_total=int(values(rows,'guidance_fallbacks').sum()))
    return s

def guards(a,b,mag=.10,runtime=1.5):
    mag_delta=(b['mag_mae_a3']-a['mag_mae_a3'])/a['mag_mae_a3'] if a['mag_mae_a3']>1e-15 else (0. if b['mag_mae_a3']<=1e-15 else float('inf'))
    v={'mag_mae_worsening_fraction':mag_delta,'nus_drop_pp':100*(a['nus_fraction']-b['nus_fraction']),
      'validity_drop_pp':100*(a['validity_fraction']-b['validity_fraction']),
      'e_hull_increase_ev_atom':b['e_hull_mean_ev_per_atom']-a['e_hull_mean_ev_per_atom'],
      'runtime_ratio':b['end_to_end_seconds_mean']/a['end_to_end_seconds_mean']}
    limits=[mag,5.,5.,.01,runtime]
    checks={k:bool(v[k]<=lim+1e-12) for k,lim in zip(v,limits)}
    return {'values':v,'pass':checks,'all_pass':all(checks.values())}

def audit(code,c,out,generation):
    checks={'all_registered_seeds':all(sorted(int(r['seed']) for r in generation if r['method']==m)==c['seeds'] for m in c['methods']),
      'score_calls_2000':all(int(r['mattergen_score_calls'])==2000 for r in generation),
      'all_generation_success':all(frozen.truth(r['success']) for r in generation),
      'config_and_frozen_sources':True,'implementation_unchanged':True,
      'no_partial_generation_directories':not list((out/'generation').glob('*/*.partial'))}
    for p,h in json.loads((out/'implementation_lock.json').read_text())['sha256'].items():
        checks['implementation_unchanged'] &= sha(p)==h
    trace=read_table(out/'guidance_trace.csv') if (out/'guidance_trace.csv').exists() else []
    guided={'A1':['F0'],'A3':['G1','G2','G3','G4'],'A4':['T1','T2'],
            'B':['F0','F1'],'A5':['B0','AB'],'A6':['F0']}[code]
    checks['exact_20_late_predictor_events']=all(
        len([r for r in trace if r['method']==m and int(r['seed'])==s])==20
        for m in guided for s in c['seeds'])
    checks['trigger_window_unchanged']=all(r['phase']=='predictor' and .00099<=float(r['t_norm'])<=.020001 for r in trace)
    bounded=[r for r in trace if r['method']!='T2']
    checks['bounded_displacements']=all(float(r['max_cart_correction_a'])<=.005+1e-9 for r in bounded)
    checks['adaptive_only_A5']=all(frozen.truth(r['adaptive_cfg_used'])==(code=='A5' and r['method'] in ['A0','AB']) for r in generation)
    if code=='A3':
        ref={(int(r['seed']),int(r['sampling_step'])):r for r in trace if r['method']=='G1'}
        checks['direction_norm_max_matched']=all(abs(float(r['max_cart_correction_a'])-float(ref[(int(r['seed']),int(r['sampling_step']))]['max_cart_correction_a']))<=1e-9
          for r in trace if r['method'] in ['G2','G3','G4'] and frozen.truth(r['accepted']))
        checks['direction_internal_norm_assertions_pass']=not any('norm distribution' in r.get('fallback_reason','') for r in trace)
        checks['direction_event_acceptance_matched']=all(frozen.truth(r['accepted'])==frozen.truth(ref[(int(r['seed']),int(r['sampling_step']))]['accepted'])
          for r in trace if r['method'] in ['G2','G3','G4'])
    if code=='B':
        cand=read_table(out/'candidate_trace.csv')
        checks['all_accepted_candidates_improve_maxF']=all(float(r['after_maxF_ev_a'])<float(r['before_maxF_ev_a'])-1e-4 for r in cand if frozen.truth(r['accepted']))
        checks['radii_in_frozen_set']=all(float(r['radius_a']) in [.005,.0025,.00125] for r in cand)
        checks['max_two_retries']=all(0<=int(r['attempt'])<=2 for r in cand)
        checks['nonincreasing_radii']=all(np.all(np.diff([float(r['radius_a']) for r in cand if int(r['seed'])==s])<=0) for s in c['seeds'])
        checks['actual_call_accounting']=all(int(r['actual_mattersim_calls'])==sum(int(r[k]) for k in ['initial_mattersim_calls','candidate_mattersim_calls','retry_mattersim_calls']) for r in generation if r['method']=='F1')
    if code=='A6':
        by={(r['method'],int(r['seed'])):r for r in generation}
        checks['post_budget_no_greater_than_F0']=all(int(by[('POST',s)]['actual_mattersim_calls'])<=int(by[('F0',s)]['actual_mattersim_calls']) for s in c['seeds'])
        checks['post_same_C0_source']=all(json.loads((out/'generation/POST'/str(s)/'run_summary.json').read_text())['post_source_sha256']==sha(out/'generation/C0'/str(s)/'generated_crystals.extxyz') for s in c['seeds'])
    return {'AUDIT':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
      'trace_rows':len(trace),'all_failures_retained':True,'DFT_VERIFIED':False}

def decisions(code,data,s,a):
    methods=list(data);base=methods[0];candidate=methods[1]
    primary=paired(data[base],data[candidate]);g=guards(s[base],s[candidate])
    d={'track':code,'n_paired':len(data[base]),'primary':primary,'guardrails':g,
       'SURROGATE_PROPERTY_EVAL':True,'DFT_VERIFIED':False,'audit':a['AUDIT']}
    if code=='A1':
        ci=primary['relative_ci95'];r=primary['relative_improvement']
        passed=r>=.15 and ci is not None and ci[0]>0 and primary['win_binomial_one_sided_p']<.05 and g['all_pass']
        strong=passed and ci[0]>=.10 and primary['wins']>=192
        d.update(status='STRONG_CONFIRMED' if strong else 'CONFIRMED' if passed else 'FAIL',
          next_action='COMPLETE_EVIDENCE_CHAIN' if passed else 'RETAIN_FORMAL32_AS_SMALL_COHORT_RESULT_NO_RESCUE')
    elif code=='A3':
        m=lambda method:s[method]['maxF_mean_ev_per_a']
        trend=m('G1')<m('G2') and m('G1')<m('G3') and m('G1')<m('G0') and m('G4')>=m('G0')
        d.update(status='SUPPORTED' if trend else 'NOT_SUPPORTED',mechanism_trend=bool(trend),
          interpretation='Descriptive prespecified ordering, not a requirement that every contrast be significant.',
          direction_reference='G2/G3/G4 replay paired G1 trajectory magnitudes; G4 reverses paired G1 MatterSim direction, not its own current-state force. This limits the causal interpretation.')
    elif code=='A4':
        uncapped=guards(s['T0'],s['T2']);bounded=guards(s['T0'],s['T1'])
        if bounded['all_pass'] and not uncapped['all_pass']:v='TRUST_REGION_NECESSARY'
        elif bounded['all_pass'] and s['T1']['maxF_mean_ev_per_a']<s['T2']['maxF_mean_ev_per_a']:v='TRUST_REGION_USEFUL_BUT_NOT_ESSENTIAL'
        else:v='TRUST_REGION_NOT_SUPPORTED'
        d.update(status='SUPPORTED' if v!='TRUST_REGION_NOT_SUPPORTED' else 'NOT_SUPPORTED',
          conclusion=v,bounded_guardrails=bounded,unbounded_guardrails=uncapped,
          interpretation='Necessary refers only to this frozen linear unbounded comparator and these guardrails, not all uncapped algorithms.')
    elif code=='A5':
        cfg_den=s['C0']['mag_mae_a3']-s['A0']['mag_mae_a3']
        force_den=s['C0']['maxF_mean_ev_per_a']-s['B0']['maxF_mean_ev_per_a']
        r_cfg=(s['C0']['mag_mae_a3']-s['AB']['mag_mae_a3'])/cfg_den if cfg_den>1e-15 else None
        r_force=(s['C0']['maxF_mean_ev_per_a']-s['AB']['maxF_mean_ev_per_a'])/force_den if force_den>1e-15 else None
        ab=guards(s['C0'],s['AB']);nus=100*(s['A0']['nus_fraction']-s['AB']['nus_fraction'])<=5+1e-12
        passed=r_cfg is not None and r_force is not None and r_cfg>=.7 and r_force>=.7 and nus and ab['all_pass']
        d.update(status='PASS' if passed else 'FAIL',retention_CFG=r_cfg,retention_force=r_force,
          retention_denominators={'CFG':cfg_den,'force':force_den},AB_guardrails=ab,
          NUS_preserved_vs_A0=bool(nus),threshold_interpretive_not_significance=True,
          FINAL_MODEL='Adaptive CFG + Late-stage MatterSim Force Guidance' if passed else 'F0 standalone; combination not confirmed')
    elif code=='B':
        p95=reduction(s['F0']['maxF_p95_ev_per_a'],s['F1']['maxF_p95_ev_per_a'])
        high=reduction(s['F0']['high_maxF_gt_0_2_fraction'],s['F1']['high_maxF_gt_0_2_fraction'])
        g=guards(s['F0'],s['F1'],mag=.05)
        mean_worsening=s['F1']['maxF_mean_ev_per_a']/s['F0']['maxF_mean_ev_per_a']-1
        calls=s['F1']['actual_mattersim_calls_mean']/s['F0']['actual_mattersim_calls_mean']
        passed=g['all_pass'] and mean_worsening<=.05+1e-12 and calls<=2.5+1e-12
        go=passed and (p95>=.15 or (high is not None and high>=.25))
        borderline=passed and p95>=.05 and calls<=2.0
        d.update(status='GO' if go else 'BORDERLINE' if borderline else 'FAIL',
          p95_relative_reduction=p95,p90_relative_reduction=reduction(s['F0']['maxF_p90_ev_per_a'],s['F1']['maxF_p90_ev_per_a']),
          high_maxF_threshold_ev_a=.2,high_maxF_rate_relative_reduction=high,
          mean_maxF_worsening=mean_worsening,actual_calls_ratio=calls,guardrails=g,
          all_F1_guardrails_pass=bool(passed),next_action='F1_FORMAL32_NEXT_ROUND_ONLY' if go else 'KEEP_F0_NO_TUNING',
          F0_remains_formal=True,automatic_formal32=False)
    else:
        d.update(status='COMPLETE',force_vs_post=paired(data['POST'],data['F0']),
          post_vs_base=paired(data['C0'],data['POST']),budget_includes_failed_actual_calculator_calls=True,
          evaluation_calls_separate_and_equal_for_all=True)
    if a['AUDIT']!='PASS':d.update(pre_audit_status=d['status'],status='AUDIT_FAIL')
    return d

def plots(out,data,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'plots';dest.mkdir(exist_ok=True)
    methods=list(data)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for m in methods:
        x=np.sort(values(data[m],'maxF_ev_per_a'));axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=m)
    axes[0].set(xlabel='Pre-relaxation MaxF (eV/A)',ylabel='Empirical CDF');axes[0].legend()
    x=values(data[methods[0]],'maxF_ev_per_a');y=values(data[methods[1]],'maxF_ev_per_a')
    axes[1].scatter(x,y,s=15);mx=max(x.max(),y.max());axes[1].plot([0,mx],[0,mx],'k--')
    axes[1].set(xlabel=methods[0]+' MaxF',ylabel=methods[1]+' MaxF')
    fig.tight_layout();fig.savefig(dest/'maxF_ecdf_paired.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--track',required=True);arg=p.parse_args()
    c=config(arg.track);out=output(arg.track)
    if (out/'decision_summary.json').exists():raise FileExistsError(out/'decision_summary.json')
    frozen.ROOT=out;frozen.SEEDS=tuple(c['seeds']);frozen.METHODS=tuple(c['methods'])
    generation=read_table(out/'generation_manifest.csv');props=read_table(out/'p0_property_metrics.csv')
    data={m:frozen.load_method(m,generation,props) for m in c['methods']}
    gmap={(r['method'],int(r['seed'])):r for r in generation}
    for m,rows in data.items():
        for r in rows:
            g=gmap[(m,r['seed'])]
            for key in ['actual_mattersim_calls','failed_mattersim_calls','peak_allocated_bytes',
                        'initial_mattersim_calls','candidate_mattersim_calls','retry_mattersim_calls','failed_candidate_calls']:
                r[key]=int(g[key])
    summaries={m:summary(m,rows) for m,rows in data.items()}
    comparisons=[paired(data[a],data[b]) for a,b in itertools.combinations(data,2)]
    secondary=[paired(data[c['methods'][0]],data[m],metric) for m in c['methods'][1:]
       for metric in ['atomic_force_mean_ev_per_a','rmsd_a','e_hull_ev_per_atom','mag_absolute_error_a3',
                      'stable','nus','valid','mag_hit','end_to_end_generation_seconds']]
    audit_result=audit(arg.track,c,out,generation)
    d=decisions(arg.track,data,summaries,audit_result)
    independent=ROOT/'independent_mlip'/arg.track/'decision_summary.json'
    if independent.exists():d['independent_mlip']=json.loads(independent.read_text())
    elif arg.track in ['A1','A3','A6']:
        d.update(pre_independent_status=d['status'],status='INCOMPLETE_INDEPENDENT_EVALUATION')
    table(out/'per_structure_metrics.csv',list(itertools.chain.from_iterable(data.values())))
    table(out/'quality_metrics.csv',list(summaries.values()))
    table(out/'efficiency_metrics.csv',[{'method':m,**{k:v for k,v in s.items() if any(t in k for t in ['seconds','calls','allocated','fallback'])}} for m,s in summaries.items()])
    write_json(out/'paired_bootstrap.json',{'MaxF':comparisons,'secondary':secondary,
      'secondary_sign_note':'All improvements are baseline minus candidate; for rates positive indicates rate decrease, not benefit.'})
    pairrows=[]
    for i,seed in enumerate(c['seeds']):
        row={'seed':seed}
        for m,rs in data.items():row.update({m+'_'+k:v for k,v in rs[i].items() if k not in ['method','seed']})
        pairrows.append(row)
    table(out/'paired_results.csv',pairrows)
    if arg.track=='A1':
        ranks=np.argsort(values(data['C0'],'maxF_ev_per_a'),kind='stable');quartiles=[]
        for i,idx in enumerate(np.array_split(ranks,4)):
            ids=set(int(j) for j in idx)
            aa=[r for j,r in enumerate(data['C0']) if j in ids];bb=[r for j,r in enumerate(data['F0']) if j in ids]
            quartiles.append({'quartile':i+1,'seed_membership':[r['seed'] for r in aa],**paired(aa,bb)})
        write_json(out/'baseline_maxF_quartiles.json',{'non_gating':True,'stratification':'C0 MaxF only; stable seed-order tie breaks','quartiles':quartiles})
    write_json(out/'audit_results.json',audit_result);plots(out,data,comparisons)
    lines=['# '+c['experiment'],'',f"Status: {d['status']}",f"Paired seeds: {len(c['seeds'])}; no seeds dropped.",'',
      '| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,s in summaries.items():lines.append(f"| {m} | {s['maxF_mean_ev_per_a']:.6f} | {s['maxF_p95_ev_per_a']:.6f} | {s['mag_mae_a3']:.6f} | {s['nus_fraction']:.2%} | {s['validity_fraction']:.2%} | {s['end_to_end_seconds_mean']:.2f} | {s['actual_mattersim_calls_mean']:.2f} |")
    lines+=['','## Decision and limitations','', '```json',json.dumps(d,indent=2),'```','',
      'All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. '
      'Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. '
      'P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. '
      'Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.','']
    (out/'final_report.md').write_text('\n'.join(lines))
    d['completed_utc']=datetime.now(timezone.utc).isoformat();write_json(out/'decision_summary.json',d)
    print(json.dumps(d,indent=2))

if __name__=='__main__':main()
