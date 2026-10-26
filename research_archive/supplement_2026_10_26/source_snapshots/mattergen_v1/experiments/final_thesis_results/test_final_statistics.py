"""CPU-only unit tests for endpoint semantics and paired resampling."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('final_collect',Path(__file__).with_name('collect_results.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def data(metric,baseline,method):
    return pd.DataFrame({'cohort':['test']*6,'method':['C0']*3+['F0']*3,
        'seed':[1,2,3]*2,metric:baseline+method})

def test_ratio_of_means_not_mean_relative_changes():
    d=data('MaxF',[1.,2.,9.],[.5,2.,8.])
    s=c.pair(d,'C0','F0','MaxF')
    assert np.isclose(s['relative_improvement'],1.5/12.)
    assert s['wins']==2 and s['ties']==1 and s['losses']==0
    assert np.isclose(s['delta_method_minus_baseline'],-.5)
    assert np.isclose(s['paired_median_delta'],-.5)
    assert not np.isclose(s['relative_improvement'],np.mean([.5,0,1/9]))

def test_higher_is_better_and_zero_baseline_rate():
    d=data('Stable',[0.,0.,0.],[1.,0.,1.]);s=c.pair(d,'C0','F0','Stable')
    assert s['wins']==2 and s['losses']==0 and s['relative_improvement'] is None
    assert s['delta_method_minus_baseline']>0

def test_seed_pairing_and_bootstrap_determinism():
    d=data('MaxF',[1.,2.,9.],[.5,2.,8.])
    assert c.pair(d,'C0','F0','MaxF')==c.pair(d.sample(frac=1,random_state=5),'C0','F0','MaxF')

def test_identity_is_all_ties_zero_interval():
    s=c.pair(data('MaxF',[1.,2.,9.],[1.,2.,9.]),'C0','F0','MaxF')
    assert s['ties']==3 and s['delta_ci95_low']==s['delta_ci95_high']==0

def test_incompatible_seed_join_rejected():
    d=data('MaxF',[1.,2.,9.],[.5,2.,8.]);d.loc[5,'seed']=4
    try:c.pair(d,'C0','F0','MaxF')
    except AssertionError:return
    raise AssertionError('Unpaired seeds were not rejected')
