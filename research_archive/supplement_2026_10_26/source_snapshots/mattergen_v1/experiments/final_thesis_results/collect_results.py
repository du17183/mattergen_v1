"""Source-pinned final thesis data collection and supplementary paired statistics.

Does not execute or modify historical generation/evaluation. Re-running only
refreshes this finalization's derived exports. Original decisions stay unchanged.
"""
from pathlib import Path
import datetime, hashlib, io, itertools, json, subprocess, sys
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
F0=PROJECT.with_name('mattergen_v1_matersim_guidance')
HIST=F0/'experiments/innovation2_finalization'
MINI=PROJECT/'experiments/final_innovation1_audit/minimal_ablation'
ARCHIVE='origin/archive/thesis-analysis-package-v1'
E3REF='origin/feature/q3-e3-pcr-formal256'
SPECS={'A1':('mattersim_late_force_guidance_formal256',740000,256,['C0','F0']),
 'A3':('mattersim_force_direction_ablation',741000,32,['G0','G1','G2','G3','G4']),
 'A4':('mattersim_trust_region_ablation',742000,32,['T0','T1','T2']),
 'A5':('adaptive_cfg_force_guidance_compatibility',744000,64,['C0','A0','B0','AB']),
 'A6':('mattersim_equal_budget_post_generation',745000,32,['C0','F0','POST'])}
MAP={'maxF_ev_per_a':'MaxF','atomic_force_mean_ev_per_a':'Atomic Force','rmsd_a':'RMSD',
 'e_hull_ev_per_atom':'E-hull','stable':'Stable','novel':'Novel','unique':'Unique','nus':'NUS',
 'mag_absolute_error_a3':'Property MAE','mag_hit':'Hit','valid':'Validity','end_to_end_generation_seconds':'Runtime'}
HIGHER={'Stable','Novel','Unique','NUS','Hit','Validity','Composition validity'}
UNITS={'MaxF':'eV/Angstrom','Atomic Force':'eV/Angstrom','Independent MaxF':'eV/Angstrom',
 'Independent mean force':'eV/Angstrom','RMSD':'Angstrom','E-hull':'eV/atom','Property MAE':'mu_B/Angstrom^3','Runtime':'seconds'}
SOURCES={};GIT_SOURCES=[];COHORTS={};AUDITS={};ORIGINAL={};STATS=[];DISTRIBUTIONS=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def record(p):
    p=Path(p);SOURCES[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def js(p):return json.loads(record(p).read_text())
def frame(p):return pd.read_csv(record(p),float_precision='round_trip')
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def git_source(ref,path):
    commit=subprocess.check_output(['git','rev-parse',ref],cwd=PROJECT,text=True).strip()
    raw=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=PROJECT)
    dest=ROOT/'sources'/commit/path;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_bytes()==raw
    else:dest.write_bytes(raw)
    record(dest);GIT_SOURCES.append({'commit':commit,'path':path,'exported_to':str(dest),'sha256':sha(dest)})
    return dest
def md_table(df):return df.fillna('NA').to_markdown(index=False,floatfmt='.8g')+'\n'
def tables(name,df):
    df.to_csv(ROOT/(name+'.csv'),index=False,float_format='%.17g',na_rep='NA')
    (ROOT/(name+'.md')).write_text(md_table(df))
def pair(df,base,method,metric,seed=20260915,tolerance=1e-12):
    x=df[df.method==base].sort_values('seed');y=df[df.method==method].sort_values('seed')
    assert np.array_equal(x.seed,y.seed)
    a=x[metric].to_numpy(float);b=y[metric].to_numpy(float);assert np.isfinite(a).all() and np.isfinite(b).all()
    idx=np.random.default_rng(seed).integers(0,len(a),(20000,len(a)))
    delta=b-a;improvement=delta if metric in HIGHER else -delta
    boot=delta[idx].mean(1);denom=a[idx].mean(1)
    rel=None;relci=None
    if np.all(denom>0):
        sign=1 if metric in HIGHER else -1
        rel=float(sign*delta.mean()/a.mean());relci=np.quantile(sign*boot/denom,[.025,.975]).tolist()
    return {'cohort':str(df.cohort.iloc[0]),'baseline':base,'method':method,'metric':metric,
      'units':UNITS.get(metric,'fraction'),'n':len(a),'baseline_mean':float(a.mean()),'method_mean':float(b.mean()),
      'baseline_median':float(np.median(a)),'method_median':float(np.median(b)),
      'delta_method_minus_baseline':float(delta.mean()),'paired_median_delta':float(np.median(delta)),
      'delta_ci95_low':float(np.quantile(boot,.025)),'delta_ci95_high':float(np.quantile(boot,.975)),
      'relative_improvement':rel,'relative_ci95_low':None if relci is None else relci[0],
      'relative_ci95_high':None if relci is None else relci[1],
      'wins':int((improvement>tolerance).sum()),'ties':int((np.abs(improvement)<=tolerance).sum()),
      'losses':int((improvement < -tolerance).sum()),'tie_tolerance':tolerance,
      'bootstrap_resamples':20000,'bootstrap_seed':seed,
      'evidence':'supplementary finalization statistics; not a replacement for frozen original tests; unadjusted CI'}

def add_cohort(name,df,comparisons=None,metrics=None):
    df=df.copy();df['cohort']=name;assert not df.duplicated(['method','seed']).any();COHORTS[name]=df
    metrics=metrics or [m for m in list(MAP.values())+['Independent MaxF','Independent mean force'] if m in df]
    comparisons=comparisons or list(itertools.combinations(df.method.unique(),2))
    for base,method in comparisons:
        for metric in metrics:STATS.append(pair(df,base,method,metric))
    for method,g in df.groupby('method',sort=False):
        for metric in metrics:
            x=g[metric].to_numpy(float);assert np.isfinite(x).all()
            DISTRIBUTIONS.append({'Cohort':name,'Method':method,'Metric':metric,'n':len(x),'units':UNITS.get(metric,'fraction'),
              'mean':float(x.mean()),'median':float(np.median(x)),'p90':float(np.quantile(x,.90)),
              'p95':float(np.quantile(x,.95)),'max':float(x.max())})

def historical_f0():
    registry=frame(HIST/'experiment_registry.csv').set_index('track')
    for track,(dirname,first,n,methods) in SPECS.items():
        out=F0/'experiments'/dirname;c=yaml.safe_load(record(out/'config.yaml').read_text())
        assert sha(out/'config.yaml')==registry.loc[track,'config_sha256']
        for p,h in c['immutable_source_sha256'].items():
            if p not in SOURCES:record(p)
            assert SOURCES[p]['sha256']==h,p
        assert c['seeds']==list(range(first,first+n)) and c['methods']==methods
        audit=js(out/'audit_results.json');assert audit['AUDIT']=='PASS'
        decision=js(out/'decision_summary.json');ORIGINAL[track]=decision
        raw=frame(out/'per_structure_metrics.csv');summary=frame(out/'quality_metrics.csv')
        for method in methods:
            g=raw[raw.method==method].sort_values('seed');assert g.seed.tolist()==c['seeds']
            assert np.isfinite(g[list(MAP)].to_numpy(float)).all()
            assert np.all(g.mattergen_score_calls==2000)
            np.testing.assert_allclose(g.maxF_ev_per_a.mean(),summary.set_index('method').loc[method,'maxF_mean_ev_per_a'],atol=1e-14)
        manifest=js(out/'structures/manifest.json')
        for p,h in manifest['source_sha256'].items(): assert sha(p)==h,p
        # Independent raw vectors reproduce both force endpoints exactly.
        ind=HIST/'independent_mlip'/track
        lock=js(ind/'evaluator_lock.json');assert sha(lock['checkpoint'])==lock['sha256']
        independent=frame(ind/'independent_per_structure.csv')
        rawforce=js(ind/'raw_forces.json');index=independent.set_index(['method','seed'])
        assert len(rawforce)==len(raw)==len(independent)
        for r in rawforce:
            norms=np.linalg.norm(np.asarray(r['forces_ev_a'],float),axis=1);row=index.loc[(r['method'],r['seed'])]
            np.testing.assert_allclose([norms.max(),norms.mean()],[row.maxF_ev_a,row.mean_force_ev_a],atol=1e-12)
        original_ind=js(ind/'paired_bootstrap.json');original_boot=js(out/'paired_bootstrap.json')
        ORIGINAL[track+'_independent']=original_ind;ORIGINAL[track+'_bootstrap']=original_boot
        joined=raw.rename(columns=MAP).merge(independent[['method','seed','maxF_ev_a','mean_force_ev_a']].rename(columns={
            'maxF_ev_a':'Independent MaxF','mean_force_ev_a':'Independent mean force'}),on=['method','seed'],validate='one_to_one')
        add_cohort(track,joined)
        # Check new calculations of the already frozen primary endpoint reproduce original intervals.
        for r in original_boot['MaxF']:
            s=next(v for v in STATS if v['cohort']==track and v['baseline']==r['baseline'] and v['method']==r['method'] and v['metric']=='MaxF')
            np.testing.assert_allclose([s['relative_improvement'],s['relative_ci95_low'],s['relative_ci95_high']],
              [r['relative_improvement'],*r['relative_ci95']],atol=1e-12)
        AUDITS[track]={'complete':True,'pairs':n,'arms':methods,'source_structure_hashes_verified':len(manifest['source_sha256']),
          'independent_raw_vectors_reproduced':len(rawforce),'original_audit':audit,'original_verdict':decision.get('status',decision.get('decision')),
          'primary_bootstrap_reproduced':True,'config':c,'result_dir':str(out)}
    for code,dirname in [('P0','mattersim_late_force_guidance_p0'),('Formal32','mattersim_late_force_guidance_formal32')]:
        out=F0/'experiments'/dirname
        if code=='P0':
            raw=frame(out/'heldout_metrics.csv').rename(columns=MAP)
        else:
            wide=frame(out/'paired_results.csv');rows=[]
            for method in ['C0','F0']:
                for _,r in wide.iterrows():
                    rows.append({'method':method,'seed':int(r.seed),**{v:r[method+'_'+k] for k,v in MAP.items()}})
            raw=pd.DataFrame(rows)
        add_cohort(code,raw)
        ORIGINAL[code]=js(out/'decision_summary.json')
    ORIGINAL['diagnostic']=js(F0/'experiments/mattersim_late_force_guidance_p0/diagnostic_summary.json')
    record(F0/'experiments/mattersim_late_force_guidance_p0/diagnostic_stability.csv')
    AUDITS['preregistered_seed_scan']=js(HIST/'historical_seed_audit.json')
    AUDITS['regression']=js(HIST/'regression_audit.json')

def innovation1():
    prefix='thesis_archive/'
    paths=['data/innovation1/source_manifest.json','data/innovation1/source/formal_per_seed_records.csv',
      'data/innovation1/source/C0_official_metrics_per_structure.csv','data/innovation1/source/A0_official_metrics_per_structure.csv',
      'data/innovation1/per_seed_metrics.csv','reports/innovation1/formal_final_report.json',
      'reports/innovation1/formal_final_report.md','configs/adaptive_cfg_final.yaml',
      'experiments/innovation1_adaptive_cfg/experiment_manifest.json']
    exported={p:git_source(ARCHIVE,prefix+p) for p in paths}
    source=js(exported['data/innovation1/source_manifest.json'])
    for r in source['sources']:assert sha(exported[r['archived_source']])==r['sha256']
    data=frame(exported['data/innovation1/source/formal_per_seed_records.csv'])
    rows=[]
    for method in ['C0','A0']:
        raw=frame(exported[f'data/innovation1/source/{method}_official_metrics_per_structure.csv'])
        assert sorted(raw.seed)==list(range(20000,20256))
        merged=data.merge(raw,on='seed',validate='one_to_one')
        for _,r in merged.iterrows():
            rows.append({'method':method,'seed':int(r.seed),'E-hull':r['energy_above_hull_per_atom'],'Stable':r['stable_'+method],
              'NUS':r['NUS_'+method],'Novel':r['novel_'+method],'Unique':r['unique_'+method],
              'Validity':r['basic_structure_valid_'+method],'Composition validity':r['composition_valid_'+method],
              'RMSD':r.rmsd_from_relaxation,'Runtime':r['elapsed_'+method],
              'Post-relax MaxF':r.maximum_force_ev_ang})
    add_cohort('Innovation1 Formal256',pd.DataFrame(rows))
    final=js(exported['reports/innovation1/formal_final_report.json']);ORIGINAL['Innovation1']=final
    assert final['FORMAL_256_COMPLETED'] and final['innovation1']['FORMAL_INNOVATION1_CONFIRMED']
    assert not final['FORMAL_PARAMETERS_RETUNED']
    for metric,old in zip(['E-hull','Stable','NUS'],final['innovation1']['paired_statistics']):
        r=next(s for s in STATS if s['cohort']=='Innovation1 Formal256' and s['metric']==metric)
        np.testing.assert_allclose(r['delta_method_minus_baseline'],old['paired_mean_difference_right_minus_left'],atol=1e-14)
    AUDITS['Innovation1']={'formal_complete':True,'pairs':256,'raw_source_sha256_match':True,
       'frozen_config':yaml.safe_load(exported['configs/adaptive_cfg_final.yaml'].read_text()),
       'historical_statistics':'Original three primary CIs all include zero; original confirmation was directional guardrail based.',
       'bootstrap_provenance':'Original CI and p-values preserved. Exact original RNG seed and resample count not present in available formal archive; supplementary 20k paired bootstrap is explicitly separate.',
       'missing_historical_fields':{'Property MAE':'Not evaluated in original Formal256; no raw structures available locally for complete retrospective target validation.',
          'Hit':'Not evaluated in original Formal256. New n=8 and A5 n=64 are different cohorts, never backfilled.',
          'MaxF':'Archived maximum_force_ev_ang is POST-relax force, cannot be relabeled pre-relax.',
          'Atomic Force':'Pre-relax atom forces not preserved in original formal archive.'}}
    if (MINI/'completion.json').exists():
        assert js(MINI/'completion.json')['complete'];raw=frame(MINI/'per_structure_metrics.csv').rename(columns=MAP)
        for method in ['C0','A0','POS_ONLY','NO_EMA','SHARED_EMA','NO_CLIP']:
            assert sorted(raw[raw.method==method].seed)==list(range(746000,746008))
        add_cohort('Innovation1 minimal8',raw,comparisons=[('C0','A0')]+[('A0',m) for m in ['POS_ONLY','NO_EMA','SHARED_EMA','NO_CLIP']])
        AUDITS['Innovation1_minimal8']={'complete':True,'preregistration':js(MINI/'preregistered_config.json'),'seed_audit':js(MINI/'historical_seed_audit.json')}
    else:AUDITS['Innovation1_minimal8']={'complete':False,'status':js(MINI/'pipeline_status.json')}

def e3g():
    prefix='reports/q3_e3_pcr/formal256/'
    exports={n:git_source(E3REF,prefix+n) for n in ['formal_frozen_manifest.json','final_summary.json',
       'primary_statistics.json','gate_mechanism_summary.json','formal_seed_audit.json','quality_summary.json',
       'artifacts/results/refinement_manifest.csv','artifacts/results/refinement_summary.json']}
    rows=[]
    for method,directory in [('C0','C0'),('E3-A','ALWAYS_ON'),('E3-G','Q3_E3_PCR')]:
        p=git_source(E3REF,prefix+directory+'/official_metrics_per_structure.csv');raw=frame(p)
        assert sorted(raw.candidate_seed)==list(range(40000,40256))
        for _,r in raw.iterrows():
            rows.append({'method':method,'seed':int(r.candidate_seed),'MaxF':r.pre_relax_max_force_ev_ang,
             'RMSD':r.rmsd_from_relaxation,'E-hull':r.energy_above_hull_per_atom,'Stable':r.stable,
             'Novel':r['novel'],'Unique':r['unique'],'NUS':r.novel_unique_stable,'Validity':r.structure_validity,
             'Composition validity':r.comp_validity,'Relaxation seconds':r.relax_elapsed_seconds})
    add_cohort('E3 Formal256',pd.DataFrame(rows))
    ORIGINAL['E3']=js(exports['final_summary.json']);ORIGINAL['E3_primary']=js(exports['primary_statistics.json'])
    ORIGINAL['E3_gate']=js(exports['gate_mechanism_summary.json'])
    assert ORIGINAL['E3']['Q3_E3_PCR_FORMAL256_COMPLETED']
    c=js(exports['formal_frozen_manifest.json']);assert c['seed_audit']['passed']
    for m,old in ORIGINAL['E3_primary'].items():
        g=COHORTS['E3 Formal256'];s=pair(g,'C0',m,'MaxF',seed=c['statistics']['bootstrap_seed'],tolerance=0.)
        np.testing.assert_allclose([s['delta_ci95_low'],s['delta_ci95_high']],old['bootstrap_95_ci'],atol=1e-12)
    AUDITS['E3']={'complete':True,'frozen_manifest':c,'primary_bootstrap_reproduced':True,
     'runtime_note':'Archived relaxation seconds are NOT generation+refinement runtime. Missing end-to-end runtime is NA.',
     'missing_fields':['Property MAE','Hit','Atomic Force','Runtime'],'cohort_comparison':'Different seeds and target from F0; no absolute head-to-head ranking.'}

def registry():
    rows=[]
    for eid,name,track,directory,priority in [
      ('EXP-01','Equal-budget online vs POST','A6','equal_budget_online_vs_post',1),
      ('EXP-02','Independent MLIP Formal256','A1','final_independent_mlip_validation',2),
      ('EXP-03','Innovation1 formal and minimal ablation','I1','final_innovation1_audit',3),
      ('EXP-04','Force direction','A3','final_force_direction_audit',4),
      ('EXP-05','Bounded correction','A4','final_bounded_correction_audit',5),
      ('EXP-06','Adaptive CFG + F0 interaction','A5','final_interaction_analysis',6),
      ('EXP-07','DFT environment audit','DFT','dft_verification',7),
      ('EXP-08','E3-G baseline','E3','final_e3g_baseline_audit',8),
      ('EXP-09','Final artifacts','FINAL','final_thesis_results',9)]:
        status='COMPLETED_EXISTING' if track in SPECS or track=='E3' else 'RUNNING'
        if track=='DFT':status='SKIPPED_NOT_APPLICABLE'
        if track=='I1':status='COMPLETED' if AUDITS['Innovation1_minimal8']['complete'] else AUDITS['Innovation1_minimal8']['status']['status']
        wt=F0 if track in SPECS else PROJECT
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=wt,text=True).strip()
        rows.append({'experiment_id':eid,'experiment_name':name,'priority':priority,'worktree':str(wt),'branch':branch,
          'tmux_session':'nohup/setsid pid 1988095' if track=='I1' else 'not applicable: existing evidence or foreground CPU analysis',
          'gpu_ids':'5;6;7' if track=='I1' else 'not used this turn',
          'command':f'{sys.executable} {MINI.parent}/minimal_ablation.py --pipeline' if track=='I1' else f'{sys.executable} {ROOT}/collect_results.py',
          'start_time':json.loads((MINI/'preregistered_config.json').read_text())['registered_utc'] if track=='I1' else '',
          'end_time':str(ORIGINAL.get(track,{}).get('completed_utc','')) if track in SPECS else '',
          'status':status,'exit_code':0 if status.startswith('COMPLETED') else '',
          'result_dir':str(PROJECT/'experiments'/directory),
          'notes':'Historical data reused without rerun; scientific negative result does not mean execution failed.' if status=='COMPLETED_EXISTING' else ''})
    pd.DataFrame(rows).to_csv(ROOT/'run_registry.csv',index=False)
    return rows

def export():
    main=[]
    for name in ['Innovation1 Formal256','E3 Formal256','A1','Formal32','P0','A6','A3','A4','A5']+(['Innovation1 minimal8'] if 'Innovation1 minimal8' in COHORTS else []):
        df=COHORTS[name]
        for method,g in df.groupby('method',sort=False):
            row={'Method':method,'Cohort':name,'Seed range':f'{int(g.seed.min())}-{int(g.seed.max())}','n':len(g)}
            for metric in ['Property MAE','Hit','MaxF','Atomic Force','RMSD','E-hull','Stable','Novel','Unique','NUS','Validity','Runtime']:
                row[metric]=float(g[metric].mean()) if metric in g else np.nan
            row['Target dft_mag_density']=0.1 if name in ['Innovation1 Formal256','E3 Formal256','Innovation1 minimal8'] else 0.2
            row['Method label']=('Official MatterGen' if method=='C0' or (name=='P0' and method=='B0') else
                'Adaptive CFG' if method=='A0' else 'RC-NFGD' if method=='F0' or (name=='A5' and method=='B0') else method)
            row['Runtime scope']='NA (only relaxation time archived)' if name=='E3 Formal256' else 'sampling+force-potential load where applicable; excludes evaluator runtime'
            main.append(row)
    tables('main_results',pd.DataFrame(main))
    header='# Unified main results\n\nUnits: force=eV/Angstrom; RMSD=Angstrom; E-hull=eV/atom; magnetic density error=mu_B/Angstrom^3; rates=fraction; runtime=seconds. NA means unavailable, never imputed. Cohorts/targets differ; compare each method with its paired baseline, not absolute cross-cohort ranking.\n\n'
    (ROOT/'main_results.md').write_text(header+(ROOT/'main_results.md').read_text())
    tables('ablation_results',pd.DataFrame(main)[pd.DataFrame(main).Cohort.isin(['A3','A4','A5','A6','Innovation1 minimal8'])])
    pd.concat(COHORTS.values(),ignore_index=True).to_csv(ROOT/'all_per_seed_metrics.csv',index=False,float_format='%.17g',na_rep='NA')
    pd.DataFrame(STATS).to_csv(ROOT/'paired_statistics.csv',index=False,float_format='%.17g',na_rep='NA')
    pd.DataFrame(DISTRIBUTIONS).to_csv(ROOT/'distribution_statistics.csv',index=False,float_format='%.17g')
    save(ROOT/'source_audits.json',AUDITS);save(ROOT/'original_decisions.json',ORIGINAL)
    rows=[]
    for s in STATS:
        if (s['cohort'],s['baseline'],s['method']) not in [('Innovation1 Formal256','C0','A0'),('A1','C0','F0'),('E3 Formal256','C0','E3-G'),('A6','C0','POST'),('A6','C0','F0'),('A6','F0','POST')]:continue
        if s['metric'] not in (['E-hull','Stable','NUS'] if s['cohort']=='Innovation1 Formal256' else ['MaxF','Independent MaxF']):continue
        row={k:s[k] for k in ['cohort','baseline','method','metric','relative_improvement','relative_ci95_low','relative_ci95_high','wins','ties','losses']}
        row['CI interpretation']='Supplementary 20k paired percentile CI; original formal tests preserved separately.'
        rows.append(row)
    tables('relative_improvement_table',pd.DataFrame(rows))
    # No fake pseudopotential may be mistaken for a production configuration.
    dft=PROJECT/'experiments/dft_verification';audit=js(dft/'environment_audit.json')
    assert all(audit['binaries'][n] is None for n in ['vasp_std','vasp_gam','vasp_ncl','vasp','pw.x','abinit','castep'])
    assert all('/ase/test/testdata/vasp/fake_pseudos/' in p for p in audit['candidate_files'])
    audit.update(DFT_VERIFICATION='NOT_RUN_ENVIRONMENT_UNAVAILABLE',DFT_VERIFIED=False,
      software='No runnable VASP/QE/ABINIT/CASTEP found in audited scope',pseudopotential='Only ASE fake test POTCAR files; no production pseudopotential set',
      workflow='No executable DFT workflow/validated numerical configuration located',
      attempted='Environment audited; no single-point calculation attempted; n=0',
      optional_nonblocking=True)
    save(dft/'environment_audit.json',audit)
    record(dft/'environment_audit.json')
    (dft/'environment_audit.md').write_text('# DFT 环境审计\n\nDFT_VERIFICATION=NOT_RUN_ENVIRONMENT_UNAVAILABLE\n\n实际检查 PATH、dxl 项目目录、/opt、/usr/local/bin、/usr/bin。未找到 VASP、Quantum ESPRESSO、ABINIT 或 CASTEP 可执行程序。检出文件仅为 ASE fake_pseudos 测试 POTCAR，不可用于真实计算。未验证许可授权；不能因没找到许可文件断言用户无许可。未找到已验证计算工作流或调度器；CPU/RAM 充足本身不等于具备 DFT 运行条件。\n\n完成的是环境审计，不是 DFT 计算尝试。实际完成单点数=0；不生成虚构的 16-pair 选择/结果。可选 DFT 不阻止收尾，DFT_VERIFIED=False。\n\n详见 environment_audit.json 的原始二进制和文件检查列表。\n')
    versions={}
    from importlib import metadata
    for n in ['torch','numpy','pandas','mattersim','pymatgen','ase','matplotlib','scipy']:
        try:versions[n]=metadata.version(n)
        except metadata.PackageNotFoundError:versions[n]='not found'
    versions['CHGNet']='0.3.0 (legacy package path; frozen checkpoint independently SHA256 verified)'
    checkpoints=[PROJECT/'checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt',
      PROJECT/'checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth',
      PROJECT/'.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar']
    for p in checkpoints:record(p)
    environment=js(ROOT/'resource_audit.json')
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'thesis_title':'基于深度学习的材料逆向生成',
      'mattergen_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
      'worktrees':environment['worktrees'],'all_branches':environment['commands'][5]['stdout'],
      'versions':versions,'python':sys.executable,'sources':SOURCES,'git_sources':GIT_SOURCES,
      'configs':{k:v.get('config',v.get('frozen_config',v.get('frozen_manifest'))) for k,v in AUDITS.items() if isinstance(v,dict)},
      'checkpoint_sha256':{str(p):sha(p) for p in checkpoints},
      'cohorts':{n:{'seeds':sorted(d.seed.unique().tolist()),'methods':d.method.unique().tolist()} for n,d in COHORTS.items()},
      'metrics':{'MaxF':'mean over structures of max_i norm(F_i); raw pre-relax structure',
       'Atomic Force':'mean over structures of within-structure mean_i norm(F_i); NOT atom-count pooled',
       'E-hull':'MatterSim-relaxed surrogate energy relative to frozen Alex-MP reference, eV/atom',
       'Stable':'surrogate E-hull <=0.1 eV/atom','NUS':'novel AND unique AND stable, labels computed in each fixed cohort',
       'Property MAE':'CHGNet sum(abs atomic magmom)/volume error relative to conditioned target; surrogate only',
       'Hit':'absolute surrogate magnetic density error <=0.01','Validity':'geometric structure validity; not composition validity or synthesizability'},
      'statistics':{'original':'Unmodified archived decisions and CIs in original_decisions.json; original I1 bootstrap RNG/count unavailable in supplied archive.',
       'f0_P0':{'seed':20260913,'resamples':20000},'f0_Formal32':{'seed':20260914,'resamples':20000},
       'A1_A3_A4_A5_A6_independent':{'seed':20260915,'resamples':20000},'E3':{'seed':20260728,'resamples':20000},
       'supplementary_finalization':{'seed':20260915,'resamples':20000,'method':'paired seed bootstrap percentile 95%; ratio of resampled means, not mean per-seed percent; no multiple testing adjustment; all seeds retained'},
       'uniqueness_warning':'Bootstrap conditions on full-cohort unique/novel labels, does not rerun deduplication on resampled structures.',
       'wins_warning':'E3 original primary uses strict signs; its harm epsilon is 1e-6; supplementary table uses 1e-12. Do not conflate these.'},
      'SURROGATE_PROPERTY_EVAL':True,'DFT_VERIFIED':False}
    save(ROOT/'reproducibility_manifest.json',manifest)
    rows=registry();(ROOT/'master_status.md').write_text('# Final thesis status\n\n'+md_table(pd.DataFrame(rows)[['experiment_id','experiment_name','status','result_dir']])+'\nFinal tables exported; plots and chapters pending final completion.\n')
    print(json.dumps({'cohorts':list(COHORTS),'statistics':len(STATS),'minimal_complete':AUDITS['Innovation1_minimal8']['complete'],'sources':len(SOURCES)}))

def main():
    historical_f0();innovation1();e3g();export()
if __name__=='__main__':main()
