"""Synthetic gate boundary checks, independent of generated outcomes."""
import unittest
from experiments.innovation2_finalization.analyze_cohort import decisions,guards,paired

def rows(method,n=256,value=.2):
    return [{'method':method,'seed':s,'maxF_ev_per_a':value} for s in range(n)]

def stats(force=.2,mag=.02,nus=.2,calls=20,p95=.3,high=.25):
    return {'maxF_mean_ev_per_a':force,'mag_mae_a3':mag,'nus_fraction':nus,'validity_fraction':1.,
      'e_hull_mean_ev_per_atom':0.,'end_to_end_seconds_mean':90.,'maxF_p95_ev_per_a':p95,
      'maxF_p90_ev_per_a':p95,'high_maxF_gt_0_2_fraction':high,'actual_mattersim_calls_mean':calls}

class Gates(unittest.TestCase):
    def test_formal256(self):
        d={'C0':rows('C0'),'F0':rows('F0',value=.15)}
        s={'C0':stats(),'F0':stats(force=.15)}
        self.assertEqual(decisions('A1',d,s,{'AUDIT':'PASS'})['status'],'STRONG_CONFIRMED')
        self.assertEqual(decisions('A1',d,s,{'AUDIT':'FAIL'})['status'],'AUDIT_FAIL')
        s['F0']['mag_mae_a3']=.03
        self.assertEqual(decisions('A1',d,s,{'AUDIT':'PASS'})['status'],'FAIL')

    def test_no_effect_not_confirmation(self):
        p=paired(rows('C0'),rows('F0'));self.assertEqual(p['wins'],0)
        self.assertEqual(p['relative_ci95'],[0.,0.])

    def test_closed_loop_budget_and_tail(self):
        d={'F0':rows('F0',16),'F1':rows('F1',16,value=.19)}
        s={'F0':stats(),'F1':stats(force=.19,p95=.24,calls=40)}
        self.assertEqual(decisions('B',d,s,{'AUDIT':'PASS'})['status'],'GO')
        s['F1']['actual_mattersim_calls_mean']=51
        self.assertEqual(decisions('B',d,s,{'AUDIT':'PASS'})['status'],'FAIL')

    def test_nonpositive_retention_denominator(self):
        d={m:rows(m,64) for m in ['C0','A0','B0','AB']}
        s={m:stats() for m in d}
        v=decisions('A5',d,s,{'AUDIT':'PASS'})
        self.assertIsNone(v['retention_CFG']);self.assertIsNone(v['retention_force'])
        self.assertEqual(v['status'],'FAIL')

if __name__=='__main__':unittest.main(verbosity=2)
