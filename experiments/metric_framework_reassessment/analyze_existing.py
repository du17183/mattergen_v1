"""CPU-only retrospective statistics. Reads frozen files/Git blobs; no model execution."""
from pathlib import Path
import io
import json
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from ase.io import read as read_atoms

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = 'origin/archive/thesis-analysis-package-v1'
Q3 = 'origin/feature/q3-e3-pcr-formal256'
RNG_SEED = 20260908
REPS = 20000
sources = set()

def git_text(ref, path):
    commit = subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
    sources.add(f'git:{commit}:{path}')
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT, text=True)

def csv(path, ref=None):
    if ref:
        return pd.read_csv(io.StringIO(git_text(ref, path)))
    sources.add(str(path))
    return pd.read_csv(ROOT / path)

def json_file(path):
    sources.add(str(path))
    return json.loads((ROOT / path).read_text())

def boolean(v):
    if v.dtype == object:
        return v.astype(str).str.lower().map({'true': 1., 'false': 0., '1.0': 1., '0.0': 0.}).to_numpy()
    return v.to_numpy(dtype=float)

METRICS = ['e_hull', 'nus', 'stable', 'novel', 'unique', 'rmsd', 'atomic_force', 'max_force', 'steps', 'post_relax_max_force']
HIGHER = {'nus', 'stable', 'novel', 'unique'}
major = {}

def wide_cohort(name, path, prefixes, ref):
    d = csv(path, ref)
    for arm, prefix in prefixes.items():
        f = pd.DataFrame({'seed': d.seed})
        for metric, suffix in [('e_hull','ehull'), ('max_force','max_force'), ('rmsd','rmsd'),
                               ('stable','stable'), ('nus','nus'), ('novel','novel'), ('unique','unique')]:
            f[metric] = boolean(d[f'{prefix}_{suffix}']) if metric in HIGHER else d[f'{prefix}_{suffix}']
        major[(name, arm)] = f

wide_cohort('Adaptive Formal256', 'thesis_archive/data/innovation1/per_seed_metrics.csv',
            {'C0':'c0','Adaptive CFG':'a0'}, ARCHIVE)
for arm, prefix in [('C0','C0'),('Adaptive CFG','A0')]:
    # maximum_force_ev_ang is relaxation output, NOT pre_relax_max_force_ev_ang.
    f = major[('Adaptive Formal256', arm)]
    raw = csv(f'thesis_archive/data/innovation1/source/{prefix}_official_metrics_per_structure.csv', ARCHIVE)
    raw = raw.set_index('seed').loc[f.seed]
    np.testing.assert_allclose(f.max_force, raw.maximum_force_ev_ang, atol=1e-12)
    f.rename(columns={'max_force':'post_relax_max_force'}, inplace=True)
    f['steps'] = raw.steps.to_numpy()
wide_cohort('E3-PCR Formal256', 'thesis_archive/experiments/innovation2_e3_pcr_formal256/per_seed_metrics.csv',
            {'C0':'c0','E3-A':'e3a','E3-G':'e3g'}, ARCHIVE)
for arm, directory in [('C0','C0'),('E3-A','ALWAYS_ON'),('E3-G','Q3_E3_PCR')]:
    d = csv(f'reports/q3_e3_pcr/formal256/{directory}/official_metrics_per_structure.csv', Q3)
    major[('E3-PCR Formal256', arm)]['steps'] = d.steps.to_numpy()

d = csv('experiments/corrector_residual_distillation_formal256/quality_per_seed.csv')
for arm, g in d.groupby('method', sort=False):
    major[('Corrector V2 Formal256', arm)] = g.rename(columns={'force_mean':'atomic_force','force_max':'max_force','relaxation_steps':'steps'}).reset_index(drop=True)

def cached_cohort(name, folder, arms):
    for arm in arms:
        detail = pd.read_json(ROOT / f'experiments/{folder}/quality/{arm}/official_detailed.json.gz')
        sources.add(f'experiments/{folder}/quality/{arm}/official_detailed.json.gz')
        r = json_file(f'experiments/{folder}/relaxation/{arm}/relaxation_summary.json')
        p = f'experiments/{folder}/relaxation/{arm}/initial_with_properties.extxyz'
        sources.add(p)
        atoms = read_atoms(ROOT / p, index=':')
        norms = [np.linalg.norm(a.get_forces(), axis=1) for a in atoms]
        f = pd.DataFrame({'seed': r['sample_seeds'], 'atomic_force': [v.mean() for v in norms],
                          'max_force': [v.max() for v in norms], 'steps': r['relaxation_steps']})
        for metric, key in [('e_hull','energy_above_hull_per_atom'),('nus','novel_unique_stable'),
                            ('stable','stable'),('novel','novel'),('unique','unique'),('rmsd','rmsd_from_relaxation')]:
            f[metric] = boolean(detail[key]) if metric in HIGHER else detail[key].to_numpy()
        major[(name, arm)] = f

cached_cohort('TCL Formal256', 'tcl_formal256', ['C0','FT0','TCL'])
cached_cohort('GBSA P0', 'gbsa_tcl_p0', ['C0','TCL','GBSA-TCL'])
d = csv('experiments/gbsa_tcl_p1/per_seed_results.csv').rename(columns={
    'e_hull_ev_per_atom':'e_hull','rmsd_angstrom':'rmsd',
    'atomic_force_mean_ev_per_angstrom':'atomic_force',
    'structure_max_force_ev_per_angstrom':'max_force','relaxation_steps':'steps'})
for arm, g in d.groupby('method', sort=False):
    g = g.copy()
    for k in HIGHER: g[k] = boolean(g[k])
    major[('GBSA P1', arm)] = g.reset_index(drop=True)

abs_rows = []
for (study, arm), d in major.items():
    row = {'study':study, 'method':arm, 'n':len(d)}
    for k in METRICS:
        if k not in d: continue
        v = d[k].to_numpy(dtype=float)
        assert np.isfinite(v).all(), (study, arm, k)
        row[k] = v.mean()
        for name, q in [('median',.5),('p95',.95),('p99',.99),('max',1.)]: row[k+'_'+name] = np.quantile(v,q)
    for k, threshold, label in [('max_force',1,'force_gt1'),('max_force',2,'force_gt2'),
                                ('rmsd',.5,'rmsd_gt0p5'),('steps',200,'steps_gt200'),('steps',400,'steps_gt400')]:
        if k in d: row[label] = int((d[k] > threshold).sum())
    abs_rows.append(row)
absolute = pd.DataFrame(abs_rows)
absolute.to_csv(OUT / 'major_absolute_metrics.csv',index=False)

paired_rows = []
for (study, arm), candidate in major.items():
    if arm == 'C0': continue
    baselines = ['C0'] + (['TCL','FT0'] if study == 'GBSA P1' and arm == 'GBSA-TCL' else [])
    for baseline in baselines:
        b = major[(study, baseline)].sort_values('seed').reset_index(drop=True)
        c = candidate.sort_values('seed').reset_index(drop=True)
        assert c.seed.equals(b.seed)
        indices = np.random.default_rng(RNG_SEED).integers(0,len(b),(REPS,len(b)))
        for k in METRICS:
            if k not in b or k not in c: continue
            scale = 100 if k in HIGHER else 1
            delta = (c[k].to_numpy(float)-b[k].to_numpy(float))*scale
            boot = delta[indices].mean(axis=1)
            lo,hi = np.quantile(boot,[.025,.975])
            favorable = delta if k in HIGHER else -delta
            eps = 1e-6 if k in {'atomic_force','max_force'} else 1e-12
            paired_rows.append(dict(study=study,candidate=arm,baseline=baseline,metric=k,n=len(b),
                delta=delta.mean(),ci95_low=lo,ci95_high=hi,wins=int((favorable>eps).sum()),
                ties=int((np.abs(favorable)<=eps).sum()),losses=int((favorable < -eps).sum()),
                bootstrap_resamples=REPS,bootstrap_seed=RNG_SEED,
                interpretation='retrospective; fixed sample uniqueness labels; unadjusted CI'))
paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT/'retrospective_paired_statistics.csv',index=False)

# Baseline campaigns remain separate: these quantify sampling uncertainty, not allowable harm.
variation = []
for (study, arm), d in major.items():
    if arm != 'C0' or len(d) != 256: continue
    for n in [32,64,256]:
        rng=np.random.default_rng(RNG_SEED)
        ia=rng.integers(0,len(d),(REPS,n)); ib=rng.integers(0,len(d),(REPS,n))
        for k in ['e_hull','nus','rmsd','atomic_force','max_force']:
            if k not in d: continue
            v=d[k].to_numpy(float)*(100 if k=='nus' else 1)
            diff=v[ia].mean(1)-v[ib].mean(1)
            lo,hi=np.quantile(diff,[.025,.975])
            variation.append(dict(study=study,resample_n=n,metric=k,c0_mean=v.mean(),
                                  null_difference_low=lo,null_difference_high=hi))
pd.DataFrame(variation).to_csv(OUT/'baseline_sampling_variation.csv',index=False)

# GBSA influence uses ALL leave-one-pair-out cases; no sample is removed from the main endpoint.
g=major[('GBSA P1','GBSA-TCL')].sort_values('seed').reset_index(drop=True)
diagnostic={'p1_n':32,'all_samples_retained':True,'comparisons':{},'seed85014':[], 'historical_baselines':[]}
for arm in ['C0','TCL','FT0']:
    b=major[('GBSA P1',arm)].sort_values('seed').reset_index(drop=True)
    idx=np.random.default_rng(RNG_SEED).integers(0,32,(REPS,32))
    item={}
    for k in ['e_hull','nus','rmsd','atomic_force','max_force','steps']:
        x=g[k].to_numpy(float); y=b[k].to_numpy(float); delta=x-y
        loo=(delta.sum()-delta)/31
        ratio=x.mean()/y.mean()
        # Rates can have zero resampled denominators; use absolute differences.
        ratio_ci = None
        if k in {'rmsd','atomic_force','max_force','steps'}:
            denominators = y[idx].mean(1)
            assert (denominators > 0).all()
            ratio_ci = np.quantile(x[idx].mean(1)/denominators,[.025,.975]).tolist()
        i=int(np.where(g.seed==85014)[0][0])
        item[k]={'mean_ratio':ratio,'ratio_ci95':ratio_ci,
                 'all_leave_one_pair_out_delta_range':[loo.min(),loo.max()],
                 'seed85014_contribution_to_mean_delta':delta[i]/32,
                 'seed85014_share_of_gbsa_total':x[i]/x.sum(),
                 'paired_delta_excluding_85014_sensitivity_only':loo[i],
                 'candidate_mean_excluding_85014_sensitivity_only':np.delete(x,i).mean(),
                 'baseline_mean_excluding_85014_sensitivity_only':np.delete(y,i).mean()}
    diagnostic['comparisons'][arm]=item
for arm in ['C0','FT0','TCL','GBSA-TCL']:
    d=major[('GBSA P1',arm)]
    for k,t in [('max_force',1),('max_force',2)]:
        count=int((d[k]>t).sum())
        lo=0 if count==0 else beta.ppf(.025,count,33-count)
        hi=1 if count==32 else beta.ppf(.975,count+1,32-count)
        diagnostic.setdefault('exact_binomial_tail_intervals',[]).append(dict(method=arm,threshold=t,count=count,n=32,ci95=[lo,hi]))
    p=f'experiments/gbsa_tcl_p1/relaxation/{arm}/relaxed.extxyz'
    sources.add(p)
    final_atoms=read_atoms(ROOT/p,index=':')
    initial_atoms=read_atoms(ROOT/f'experiments/gbsa_tcl_p1/relaxation/{arm}/initial_with_properties.extxyz',index=':')
    i=14; a=initial_atoms[i]; f=final_atoms[i]
    dist=a.get_all_distances(mic=True); np.fill_diagonal(dist,np.inf)
    ff=f.get_forces()
    row=d.loc[d.seed==85014].iloc[0].to_dict()
    row.update(method=arm,num_atoms=len(a),initial_minimum_distance=float(dist.min()),
               final_max_force=float(np.linalg.norm(ff,axis=1).max()),
               initial_volume_per_atom=float(a.get_volume()/len(a)),
               final_volume_per_atom=float(f.get_volume()/len(f)))
    diagnostic['seed85014'].append(row)
    diagnostic.setdefault('nus_decomposition',[]).append(dict(method=arm,
        stable_count=int(d.stable.sum()),nus_count=int(d.nus.sum()),
        novel_unique_among_stable=float(d.nus.sum()/d.stable.sum()),
        unstable_novel_count=int(((d.stable==0)&(d.novel==1)).sum())))
for (study, arm),d in major.items():
    if arm != 'C0' or len(d)!=256 or 'max_force' not in d: continue
    k=int((d.max_force>2).sum()); k6=int((d.max_force>=6.342232233).sum())
    diagnostic['historical_baselines'].append(dict(study=study,n=len(d),max_force=d.max_force.max(),
        gt2_count=k,ge_6p342232_count=k6,prob_at_least_one_gt2_in32_plugin=1-(1-k/len(d))**32))
diagnostic['gbsa_nus_discordance_vs_c0_exact_p']=binomtest(10,12,.5).pvalue
diagnostic['p99_n32_top_weight']=.69
(OUT/'gbsa_tail_diagnostics.json').write_text(json.dumps(diagnostic,indent=2,allow_nan=False,default=lambda o: o.item() if isinstance(o,np.generic) else str(o))+'\n')

# All recoverable local summary tables, with explicit column semantics and no silent atom weighting.
aliases={
 'e_hull':['e_hull','e_hull_mean_ev_per_atom','avg_energy_above_hull_per_atom_ev','avg_energy_above_hull_per_atom'],
 'nus':['nus','nus_rate','frac_novel_unique_stable','frac_novel_unique_stable_structures'],
 'stable':['stable','stable_rate','frac_stable','frac_stable_structures'],
 'novel':['novel','novel_rate','frac_novel','frac_novel_structures'],
 'unique':['unique','unique_rate','frac_unique','frac_unique_structures'],
 'rmsd':['rmsd','rmsd_mean_angstrom','rmsd_mean_a','avg_rmsd_from_relaxation_a','avg_rmsd_from_relaxation'],
 'atomic_force':['structure_force_mean_ev_per_a','pre_relaxation_structure_mean_force_mean_ev_per_a',
                 'force_mean','atomic_force_mean_mean_ev_per_angstrom','atomic_force_mean_ev_per_angstrom','atomic_force_mean_ev_per_a'],
 'max_force':['force_max','structure_max_force_mean_ev_per_angstrom','structure_max_force_mean_ev_per_a',
              'pre_relaxation_structure_max_force_mean_ev_per_a','pre_relaxation_max_force_mean_ev_per_a','pre_relaxation_max_force_mean'],
 'steps':['relaxation_steps_mean']}
historical=[]
main_methods={'cross_field_interaction_p0':{'CFI'},'cross_field_interaction_p1':{'CFI'},
 'distribution_balanced_quality_adapter':{'M2-DB'},'distribution_constrained_quality_adapter_p0':{'M2'},
 'distribution_constrained_quality_adapter_p1':{'M2'},'distribution_constrained_quality_adapter_p1b':{'M2-L1','M2-L2'},
 'gbsa_tcl_p0':{'GBSA-TCL'},'gbsa_tcl_p1':{'GBSA-TCL'},'global_transformer_adapter_p0':{'Transformer'},
 'tcl_dml_p0':{'TCL','DML'},'tcl_p1':{'TCL'},'tcl_p2':{'TCL'},'tcl_formal256':{'TCL'},
 'corrector_residual_distillation_v3_anchor':{'V3_AnchorK16'}}
categories={'cross_field_interaction_p0':'A-like exploratory; n=8 only',
 'cross_field_interaction_p1':'D: E-hull/NUS tradeoff; inferior to MLP',
 'distribution_balanced_quality_adapter':'E-like: primary adverse',
 'distribution_constrained_quality_adapter_p0':'A-like exploratory; n=8 only',
 'distribution_constrained_quality_adapter_p1':'A-like E-hull signal; no NUS gain; later replication adverse',
 'distribution_constrained_quality_adapter_p1b':'D: inconsistent primary/geometry tradeoff',
 'gbsa_tcl_p0':'A-like exploratory; safety unconfirmed at n=8',
 'gbsa_tcl_p1':'B-like: NUS signal; E-hull NI and force safety unconfirmed',
 'global_transformer_adapter_p0':'B-like exploratory; RMSD/relaxation risk; MLP unresolved',
 'tcl_dml_p0':'TCL neutral vs C0; DML E-like',
 'tcl_formal256':'D/E: no primary gain plus geometry deterioration',
 'tcl_p1':'D: weak signal vs C0; repair vs FT0',
 'tcl_p2':'D: neutral/adverse vs C0; repair vs FT0',
 'corrector_residual_distillation_v1':'D/E: arm-specific; see final report',
 'corrector_residual_distillation_v2':'D: speed gain, adverse primary point estimates',
 'corrector_residual_distillation_v3_anchor':'D/E: failed independent repair'}
verdicts={'gbsa_tcl_p1':'FAIL','gbsa_tcl_p0':'GO','tcl_formal256':'FAIL','tcl_p1':'CLEAR GO','tcl_p2':'GO',
          'tcl_dml_p0':'TCL GO / DML FAIL','cross_field_interaction_p0':'GO',
          'cross_field_interaction_p1':'FAIL','global_transformer_adapter_p0':'FAIL',
          'distribution_constrained_quality_adapter_p0':'GO',
          'distribution_constrained_quality_adapter_p1':'BORDERLINE',
          'distribution_constrained_quality_adapter_p1b':'FAIL','distribution_balanced_quality_adapter':'FAIL',
          'corrector_residual_distillation_v1':'CONTINUE development / no Formal',
          'corrector_residual_distillation_v2':'GO to Formal',
          'corrector_residual_distillation_v3_anchor':'FAIL'}
paths=sorted(ROOT.glob('experiments/*/quality_results.csv'))
paths += [ROOT/f'experiments/{f}/quality_metrics.csv' for f in ['corrector_residual_distillation_v1','corrector_residual_distillation_v2']]
paths += [ROOT/'experiments/corrector_residual_distillation_v3_anchor/stage_c_results.csv']
for path in paths:
    df=csv(str(path.relative_to(ROOT)))
    study=path.parent.name
    base=df.loc[df.method=='C0'].iloc[0]
    for _,r in df.iterrows():
        if r['method']=='C0':continue
        row={'study':study,'method':r['method'],'n':r.get('n',np.nan),
             'historical_verdict':verdicts[study], 'comparator':'C0', 'source':str(path.relative_to(ROOT))}
        row['retrospective_category'] = categories[study]
        if study in main_methods and r['method'] not in main_methods[study]:
            row['historical_verdict'] = 'CONTROL; study verdict: '+verdicts[study]
            row['retrospective_category'] = 'Comparator; see within-cohort deltas'
        if r['method']=='FT0' and study=='tcl_formal256':
            row['retrospective_category'] = 'E: primary and geometry deterioration vs C0'
        if study=='corrector_residual_distillation_v3_anchor' and r['method'].startswith('V2'):
            row['retrospective_category'] = 'D: speed replicates, RMSD benefit does not'
        for k,options in aliases.items():
            col=next((a for a in options if a in df),None)
            if col:
                row[k]=r[col]; row['delta_'+k]=(r[col]-base[col])*(100 if k in HIGHER else 1)
                row[k+'_source_column']=col
        # The P0 quality adapter has only atom-weighted force in its summary; leave structure mean missing.
        if study=='distribution_constrained_quality_adapter_p0':
            row['atomic_force']=np.nan;row['delta_atomic_force']=np.nan
        for k in ['structure_max_force_gt1_count','structure_max_force_gt2_count','relaxation_steps_gt400_count','speedup']:
            if k in r: row[k]=r[k]
        if 'n_input' in r:row['attempts']=r['n_input']
        tails=[]
        for label,options in {
          'MaxF P95':['structure_max_force_p95_ev_per_angstrom','structure_max_force_p95_ev_per_a','pre_relaxation_max_force_p95','force_max_p95'],
          'MaxF P99':['structure_max_force_p99_ev_per_angstrom','structure_max_force_p99_ev_per_a','force_max_p99'],
          'MaxF maximum':['structure_max_force_max_ev_per_angstrom','structure_max_force_max_ev_per_a','pre_relaxation_max_force_max','force_max_max'],
          'MaxF>1 count':['structure_max_force_gt1_count','force_gt1_count'],
          'MaxF>2 count':['structure_max_force_gt2_count','force_gt2_count']}.items():
            col=next((a for a in options if a in df),None)
            if col: tails.append(f'{label} {base[col]:.4g} -> {r[col]:.4g}')
        row['tail_status'] = '; '.join(tails) or 'Not recovered from summary; not assumed safe'
        row['speed_cost_note']='No controlled end-to-end speed claim'
        if study=='corrector_residual_distillation_v2':
            speed=csv('experiments/corrector_residual_distillation_v2/stage_c_speed_metrics.csv')
            matched=speed.loc[speed.method==r['method']]
            if len(matched):row['speedup']=matched.iloc[0].paired_speedup_mean
            row['speed_cost_note']='Historical Stage-C sampling benchmark, not end-to-end discovery cost'
        historical.append(row)
for study,arm in [('Adaptive Formal256','Adaptive CFG'),('E3-PCR Formal256','E3-A'),('E3-PCR Formal256','E3-G'),
                  ('Corrector V2 Formal256','V2_Frozen_Atomic75_Late30')]:
    b=absolute[(absolute.study==study)&(absolute.method=='C0')].iloc[0]
    a=absolute[(absolute.study==study)&(absolute.method==arm)].iloc[0]
    row=dict(study=study,method=arm,n=a['n'],comparator='C0',
             historical_verdict=('BORDERLINE' if study.startswith('Corrector') else
                                 'E3-A primary/quality pass (ablation)' if arm=='E3-A' else
                                 'E3_G_FORMAL_CONFIRMED' if arm=='E3-G' else 'FORMAL_INNOVATION1_CONFIRMED=True'),
             source='major_absolute_metrics.csv; original sources in sources.md')
    for k in METRICS:
        row[k]=a.get(k,np.nan);row['delta_'+k]=(a.get(k,np.nan)-b.get(k,np.nan))*(100 if k in HIGHER else 1)
    if study.startswith('Corrector'):row['speedup']=1.324642
    row['retrospective_category']=('C-like acceleration; quality NI/tails uncertain' if study.startswith('Corrector') else
                                  'C: primary preserved + physical robustness improvement' if study.startswith('E3') else
                                  'U: primary positive directions; uncertain superiority/missing pre-force')
    row['tail_status']=('Pre-force unavailable; archived force is post-relax' if study.startswith('Adaptive') else
                       f"MaxF P95 {b.max_force_p95:.4g}->{a.max_force_p95:.4g}; P99 {b.max_force_p99:.4g}->{a.max_force_p99:.4g}; >1 {b.force_gt1:.0f}->{a.force_gt1:.0f}; >2 {b.force_gt2:.0f}->{a.force_gt2:.0f}")
    row['speed_cost_note']=('Single H20 batch=1 n=16 sampling-only; CI [1.267466,1.376734]' if study.startswith('Corrector') else
                            'Extra CHGNet refiner/gate cost; no end-to-end speedup claim' if study.startswith('E3') else
                            'Online CFG; no acceleration claim here')
    historical.append(row)
history=pd.DataFrame(historical)
history.to_csv(OUT/'historical_metrics.csv',index=False)
cols=['study','method','n','historical_verdict','retrospective_category']+['delta_'+k for k in METRICS]+['tail_status','speedup','speed_cost_note']
(OUT/'historical_table.md').write_text('# Existing-data retrospective table\n\nRates use percentage points; other deltas retain original units. Missing means not recovered, never zero. All comparisons are within the named cohort.\n\n'+history.reindex(columns=cols).to_markdown(index=False,floatfmt='.5f')+'\n')

# Read source reports for the evidence list without modifying any historical artifact.
for path in sorted(ROOT.glob('experiments/*/*report.md')):
    if path.parent==OUT:continue
    sources.add(str(path.relative_to(ROOT)))
    path.read_text()
for path,ref in [('thesis_archive/reports/innovation1/formal_final_report.md',ARCHIVE),
                 ('thesis_archive/data/innovation1/source_manifest.json',ARCHIVE),
                 ('reports/q3_e3_pcr/formal256/final_report.md',Q3),
                 ('reports/q3_e3_pcr/formal256/gate_mechanism_summary.json',Q3),
                 ('docs/experiments/negative_results_summary.md',ARCHIVE)]:git_text(ref,path)
(OUT/'sources.md').write_text('# Sources read\n\nGit sources are pinned to the existing local commit. Only the new reassessment directory is written. Historical verdicts, checkpoints and reports are untouched.\n\n'+ '\n'.join('- `'+s+'`' for s in sorted(sources))+'\n')
print(absolute[['study','method','n']+METRICS].to_string(index=False))
print('\nAdaptive retrospective intervals\n'+paired[paired.study=='Adaptive Formal256'].to_string(index=False))
print('\nGBSA severe seed\n'+pd.DataFrame(diagnostic['seed85014']).to_string(index=False))
print('\nHistorical baselines\n'+pd.DataFrame(diagnostic['historical_baselines']).to_string(index=False))
print(f'Wrote existing-data analysis to {OUT}; no training, generation, forward or relaxation.')
