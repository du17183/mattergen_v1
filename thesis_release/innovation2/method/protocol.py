"""Preregister and audit the finalization cohorts without touching F0 artifacts."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
F32 = PROJECT / 'experiments/mattersim_late_force_guidance_formal32'
P0 = PROJECT / 'experiments/mattersim_late_force_guidance_p0'
CLOSED = Path('/mnt/datasets-livsyn/dxl/mattergen_v1_closed_loop_guidance')
PYTHON = '/mnt/datasets-livsyn/dxl/alm/.venv/bin/python'
SPECS = {
    'A1': ('mattersim_late_force_guidance_formal256', 740000, 256, ['C0','F0']),
    'A3': ('mattersim_force_direction_ablation', 741000, 32, ['G0','G1','G2','G3','G4']),
    'A4': ('mattersim_trust_region_ablation', 742000, 32, ['T0','T1','T2']),
    'B': ('closed_loop_force_guidance_p0', 743000, 16, ['F0','F1']),
    'A5': ('adaptive_cfg_force_guidance_compatibility', 744000, 64, ['C0','A0','B0','AB']),
    'A6': ('mattersim_equal_budget_post_generation', 745000, 32, ['C0','F0','POST']),
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def output(code):
    return (CLOSED if code=='B' else PROJECT)/'experiments'/SPECS[code][0]

def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def config(code):
    with (ROOT/'experiment_registry.csv').open() as f:
        registered=next(row for row in csv.DictReader(f) if row['track']==code)
    if sha(output(code)/'config.yaml')!=registered['config_sha256']:
        raise RuntimeError(f'Registered configuration changed: {code}')
    value=yaml.safe_load((output(code)/'config.yaml').read_text())
    for path,expected in value['immutable_source_sha256'].items():
        if sha(path)!=expected: raise RuntimeError(f'Frozen source changed: {path}')
    return value

def historical_scan():
    worktrees=[Path(x[9:]) for x in subprocess.check_output(
        ['git','worktree','list','--porcelain'],cwd=PROJECT,text=True).splitlines()
        if x.startswith('worktree ')]
    requested={s for _,start,n,_ in SPECS.values() for s in range(start,start+n)}
    pattern=re.compile(rb'(?<![A-Za-z0-9.])(74[0-5][0-9]{3})(?![A-Za-z0-9.])')
    scanned=[]; hits=[]; files=0; errors=[]
    excluded=[ROOT]+[output(code) for code in SPECS]
    suffixes={'.csv','.json','.jsonl','.md','.py','.sh','.txt','.yaml','.yml'}
    for wt in worktrees:
        for name in ['experiments','diagnostics','research','reports','thesis']:
            scope=wt/name
            if not scope.exists(): continue
            scanned.append(str(scope))
            paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore',str(scope)],text=True).splitlines()
            selected=[Path(x) for x in paths if Path(x).suffix.lower() in suffixes
                      and not any(Path(x)==e or e in Path(x).parents for e in excluded)]
            files+=len(selected)
            for name_path in paths:
                np=Path(name_path)
                if any(np==x or x in np.parents for x in excluded):continue
                for part in np.parts:
                    if part.isdigit() and int(part) in requested:
                        hits.append({'path':str(np),'seed':int(part),'kind':'path'})
            # Ripgrep scans the byte stream in native code; Python validates only matches.
            cmd=['rg','--hidden','--no-ignore','-l',r'(^|[^A-Za-z0-9.])74[0-5][0-9]{3}([^A-Za-z0-9.]|$)']
            for suffix in sorted(suffixes):cmd += ['-g','*'+suffix]
            cmd.append(str(scope))
            found=subprocess.run(cmd,text=True,capture_output=True)
            if found.returncode not in (0,1):errors.append({'path':str(scope),'error':found.stderr})
            for p in [Path(x) for x in found.stdout.splitlines()]:
                if any(p==x or x in p.parents for x in excluded): continue
                try:
                    if p.name.isdigit() and int(p.name) in requested:
                        hits.append({'path':str(p),'seed':int(p.name),'kind':'path'})
                    if not p.is_file() or p.suffix.lower() not in suffixes: continue
                    # Stream large CSV/log files; retain overlap across chunk boundaries.
                    with p.open('rb') as f:
                        carry=b''
                        for chunk in iter(lambda:f.read(1024*1024),b''):
                            b=carry+chunk
                            for m in pattern.finditer(b):
                                seed=int(m.group(1))
                                if seed in requested: hits.append({'path':str(p),'seed':seed,'kind':'text'})
                            carry=b[-32:]
                except OSError as e: errors.append({'path':str(p),'error':str(e)})
            print(json.dumps({'seed_scan_scope':str(scope),'files':len(selected),'matches':len(hits)}),flush=True)
    result={'scopes':scanned,'files_scanned':files,'historical_matches':hits,
            'scan_errors':errors,'reserved_count':len(requested),
            'utc':datetime.now(timezone.utc).isoformat(),
            'prior_known_ranges_also_checked':[[20000,20255],[40000,40255],[67000,67255],[710000,710047],[730000,730031]]}
    if hits or errors: raise RuntimeError(json.dumps(result))
    return result

def main():
    if (ROOT/'experiment_registry.csv').exists(): raise FileExistsError('Already registered')
    formal=yaml.safe_load((F32/'formal32_config.yaml').read_text())
    assert json.loads((F32/'audit_results.json').read_text())['AUDIT']=='PASS'
    assert json.loads((F32/'decision_summary.json').read_text())['MATTERSIM_FORCE_GUIDANCE_FORMAL32']=='STRONG_CONFIRMED'
    sources={str(p):sha(p) for p in [F32/'formal32_config.yaml', F32/'decision_summary.json',
        F32/'audit_results.json',F32/'guidance_trace.csv',F32/'generate_formal32_pair.py',
        P0/'late_force_sampler.py',P0/'guidance_config.yaml',P0/'analyze_p0.py',
        P0/'evaluate_p0_properties.py',P0/'run_p0_relaxation.py',P0/'run_p0_quality.py']}
    for name,expected in formal['source']['sha256'].items():
        assert sha(formal['source']['files'][name])==expected
    history=historical_scan(); write_json(ROOT/'historical_seed_audit.json',history)
    registry=[]
    for code,(name,start,n,methods) in SPECS.items():
        out=output(code); out.mkdir(parents=True,exist_ok=True)
        seeds=list(range(start,start+n))
        value={'experiment':name,'track':code,'frozen':True,'methods':methods,'seeds':seeds,
            'common':formal['common'],'F0':formal['F0'],
            'actual_trigger_model_t':formal['actual_trigger_model_t'],
            'immutable_source_sha256':sources,'bootstrap':{'resamples':20000,'seed':20260915},
            'guardrails':formal['guardrails'],'surrogate_property_eval':True,'dft_verified':False,
            'failure_policy':'retain every registered seed; never replace failed seeds; incomplete metrics block confirmatory verdict',
            'registered_utc':datetime.now(timezone.utc).isoformat()}
        if code=='A1':
            value['decision']={'mean_maxf_reduction_min':.15,'ci_low_strict_positive':True,
              'wins_clearly_exceed_losses':'exact one-sided binomial p<0.05, ties omitted',
              'strong_ci_low_min':.10,'strong_wins_min':192,'all_guardrails_required':True}
        if code=='B':
            value['F1']={'epsilon_ev_a':1e-4,'radius_init_a':.005,'radius_min_a':.00125,
              'radius_max_a':.005,'shrink_factor':.5,'max_retries':2,
              'radius_state':'per_sample_nonincreasing_across_events',
              'first_attempt_each_event':'current_radius; accepted radius retained; no growth',
              'all_attempts_fail':'zero physics correction for this event; no sample early stopping',
              'high_maxf_threshold_ev_a':.2,'tail_go_p95_reduction':.15,
              'tail_go_high_rate_relative_reduction':.25,'mean_worsening_max':.05,
              'mag_mae_worsening_max':.05,'calls_ratio_max':2.5,
              'runtime_ratio_max':1.5,'borderline_p95_reduction_min':.05,
              'borderline_low_cost_calls_ratio_max':2.0,
              'zero_F0_high_rate':'rate reduction criterion unavailable; use P95',
              'no_automatic_formal32':True,'budget_allocation':False}
        if code=='A4':
            value['unbounded_definition']='remove norm-dependent saturation: delta=0.005*centered_force/force_reference; retain safety and exact score map; no T3 unless numerical unrunability'
        if code=='A3':
            value['direction_control']='per-atom magnitudes replayed from paired G1 frozen trajectory at identical timestep; separate RNG; directions only; enforce zero net translation with norm-preserving projection'
        if code=='A5':
            value['adaptive_cfg']={'guidance_schedule':'adaptive','guidance_scale':2.,
              'guidance_min_scale':0.,'guidance_max_scale':5.,'guidance_adaptive_alpha':.5,
              'guidance_adaptive_ema':.95,'guidance_adaptive_eps':1e-6,
              'source':'origin/feature/a0-e3g-formal256:reports/a0_e3g_formal256/frozen_manifest.json'}
            value['retention']={'threshold':.7,'property_metric':'Mag MAE reduction vs C0',
               'force_metric':'Mean MaxF reduction vs C0','nonpositive_denominator':'NOT_ESTIMABLE; never PASS'}
        (out/'config.yaml').write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
        seed_data={'paired_seeds':[{'seed':s,'historical_search_status':'NO_REFERENCE_FOUND_BEFORE_REGISTRATION'} for s in seeds],
            'historical_audit':str(ROOT/'historical_seed_audit.json'),'overlap_count':0}
        write_json(out/('formal256_seeds.json' if code=='A1' else 'seeds.json'),seed_data)
        if code=='A1': (out/'formal256_config.yaml').write_text((out/'config.yaml').read_text())
        registry.append({'track':code,'experiment':name,'seed_first':start,'seed_last':start+n-1,
          'n_paired':n,'methods':';'.join(methods),'output':str(out),'config_sha256':sha(out/'config.yaml')})
    with (ROOT/'experiment_registry.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)
    write_json(ROOT/'initial_audit.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'stash_list':subprocess.check_output(['git','stash','list'],cwd=PROJECT,text=True).splitlines(),
        'frozen_source_sha256':sources,'resource_pool':[5,6,7]})
    (ROOT/'frozen_method_definition.md').write_text(
        '# Frozen method definition\n\nF0 is the exact Formal32/P0 LateMatterSimForceGuidedSampler. '
        'Target magnetic density 0.2, constant CFG 2, 1000 steps, one corrector; 20 predictor guidance events at t=0.020..0.001. '
        'Clean x0 position forces only; nominal 0.005 Å, hard 0.01 Å, minimum distance 0.5 Å; no weight updates.\n\n'
        'A5 alone composes the existing frozen adaptive CFG with F0. F1 is a separate worktree and changes only candidate acceptance/shrinking. '
        'F1 epsilon=1e-4 eV/Å; radii 0.005/0.0025/0.00125 Å; up to two retries; high-force threshold 0.2 eV/Å. '
        'F1 does not allocate budget across samples or automatically start Formal32.\n\n'
        'A4 removes both effective saturation and the redundant hard cap: merely removing 0.01 Å would leave the 0.005 Å saturation intact. '
        'The uncapped linear force scale remains 0.005/0.07795149218357911 Å per (eV/Å).\n\n'
        'Source hashes and all seed registrations are in config.yaml and experiment_registry.csv. '
        'SURROGATE_PROPERTY_EVAL=True; DFT_VERIFIED=False.\n',encoding='utf-8')
    print(json.dumps({'registered':registry,'scan_files':history['files_scanned']},indent=2))

if __name__=='__main__': main()
