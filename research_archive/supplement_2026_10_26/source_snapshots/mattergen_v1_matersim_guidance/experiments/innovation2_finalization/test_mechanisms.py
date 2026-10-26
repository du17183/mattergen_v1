"""CPU-only invariance and acceptance tests; no cohort outcomes are used."""
import unittest
from types import MethodType
import numpy as np
from ase import Atoms
from experiments.innovation2_finalization.experimental_sampler import randomized_closed_vectors
from experiments.mattersim_late_force_guidance_p0.late_force_sampler import (
    bounded_cartesian_force_correction,cartesian_to_fractional_correction)
from experiments.closed_loop_force_guidance_p0.closed_loop_sampler import ClosedLoopForceSampler

class Mechanisms(unittest.TestCase):
    def test_norm_and_translation(self):
        rng=np.random.default_rng(307)
        for n in [1,2,3,5,16,32]:
            force=rng.normal(size=(n,3))
            ref=bounded_cartesian_force_correction(force,force_reference_ev_a=.07795149218357911,
                nominal_cart_step_a=.005,hard_cart_cap_a=.01)
            v,_=randomized_closed_vectors(ref,rng)
            np.testing.assert_allclose(np.linalg.norm(v,axis=1),np.linalg.norm(ref,axis=1),atol=1e-10)
            np.testing.assert_allclose(v.sum(axis=0),0,atol=1e-10)
            self.assertLessEqual(np.linalg.norm(v,axis=1).max(),.005+1e-12)

    def test_periodic_mapping(self):
        cell=np.array([[5.,0,0],[1.,6.,0],[.2,.3,7.]])
        delta=np.array([[.001,.003,-.004],[-.001,-.003,.004]])
        frac=cartesian_to_fractional_correction(delta,cell)
        np.testing.assert_allclose(frac@cell,delta,atol=1e-15)
        for coeff in [.0001,.02,4.]:
            np.testing.assert_allclose((frac/coeff)*coeff@cell,delta,atol=1e-15)

    def fake(self,after):
        obj=object.__new__(ClosedLoopForceSampler)
        for k,v in dict(_epsilon=1e-4,_radius=.005,_radius_min=.00125,_radius_max=.005,
            _max_retries=2,_force_reference=.07795149218357911,_hard_cap=.01,
            _min_distance_floor=.5,_seed=99,_actual_calls=0,_candidate_rows=[]).items():setattr(obj,k,v)
        values=iter(after)
        def physics(self,atoms,category):
            self._actual_calls+=1
            x=next(values)
            if isinstance(x,Exception):raise x
            return 0.,np.array([[x,0,0],[-x,0,0]])
        obj._physics=MethodType(physics,obj)
        return obj

    def propose(self,obj):
        a=Atoms('Fe2',positions=[[1,1,1],[4,4,4]],cell=[8,8,8],pbc=True)
        return obj._propose(a,a.cell.array,np.array([[.2,0,0],[-.2,0,0]]),
                           {'sampling_step':980,'t_norm':.02})

    def test_accept_first(self):
        o=self.fake([.18]);d=self.propose(o)
        self.assertIsNotNone(d);self.assertEqual(o._actual_calls,1);self.assertEqual(o._radius,.005)

    def test_retries_shrink_and_keep_radius(self):
        o=self.fake([.21,.20,.18,.17]);d=self.propose(o)
        self.assertIsNotNone(d);self.assertEqual(o._actual_calls,3);self.assertEqual(o._radius,.00125)
        self.assertEqual([r['radius_a'] for r in o._candidate_rows],[.005,.0025,.00125])
        self.assertLessEqual(np.linalg.norm(d,axis=1).max(),.00125+1e-12)
        self.propose(o);self.assertEqual(o._radius,.00125);self.assertEqual(o._actual_calls,4)

    def test_epsilon_and_failure_no_correction(self):
        o=self.fake([.19995,RuntimeError('mock failure'),.22]);self.assertIsNone(self.propose(o))
        self.assertEqual(o._actual_calls,3);self.assertEqual(o._radius,.00125)
        self.assertFalse(any(r['accepted'] for r in o._candidate_rows))

if __name__=='__main__':unittest.main(verbosity=2)
