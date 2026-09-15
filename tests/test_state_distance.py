"""Independent tests for distances on a single length-measurement algebra."""
import unittest
import numpy as np
from scipy.linalg import expm, eigvalsh
from scipy.optimize import linprog
from scipy.stats import norm, wasserstein_distance
from src.geodesic import edge_hamiltonian,prepare
from src.state_distance import (state_at,pair_statistics,length_velocity,
                                quantum_fidelity,match_mean,time_reversal)


class StateDistanceTests(unittest.TestCase):
    def test_delta_reference_and_general_pair(self):
        p=np.array([.25,.5,.25,0.]);r=np.array([1.,0.,0.,0.]);eps=.3
        out=pair_statistics(p,r,eps)
        self.assertAlmostEqual(out['W1_length'],.3)
        self.assertLess(out['dual_residual'],1e-14)

    def test_same_mean_different_distribution(self):
        p=np.array([0.,1.,0.]);r=np.array([.5,0.,.5]);eps=.3
        out=pair_statistics(p,r,eps)
        self.assertEqual(out['mean_difference'],0.)
        self.assertAlmostEqual(out['W1_length'],.3)
        self.assertAlmostEqual(out['fisher_rao'],np.pi)

    def test_linear_program_independent_transport(self):
        rng=np.random.default_rng(753)
        for _ in range(4):
            p=rng.dirichlet(np.ones(5));r=rng.dirichlet(np.ones(5));eps=.2
            xx=eps*np.arange(5);cost=abs(xx[:,None]-xx).ravel()
            A=np.zeros((10,25))
            for i in range(5):A[i,5*i:5*i+5]=1;A[5+i,i::5]=1
            lp=linprog(cost,A_eq=A,b_eq=np.r_[p,r],bounds=(0,None),method='highs')
            self.assertTrue(lp.success)
            out=pair_statistics(p,r,eps)
            self.assertAlmostEqual(out['W1_length'],lp.fun,places=12)
            self.assertAlmostEqual(out['W1_length'],wasserstein_distance(xx,xx,p,r),places=12)
            self.assertLessEqual(out['W1_length'],out['concentration_upper_bound']+1e-13)
            self.assertLess(out['dual_residual'],1e-13)
            self.assertLessEqual(np.max(abs(np.diff(out['optimal_function']))),eps+1e-14)

    def test_scaling_wasserstein_but_not_fisher(self):
        p=np.array([.2,.5,.3]);r=np.array([.5,.2,.3])
        a=pair_statistics(p,r,.2);b=pair_statistics(p,r,.8)
        self.assertAlmostEqual(b['W1_length'],4*a['W1_length'])
        self.assertAlmostEqual(b['fisher_rao'],a['fisher_rao'])

    def test_gaussian_equal_mean_formula(self):
        x=np.linspace(0,20,20001);p=norm.pdf(x,loc=10,scale=.7);r=norm.pdf(x,loc=10,scale=1.2)
        p/=p.sum();r/=r.sum();out=pair_statistics(p,r,x[1]-x[0])
        self.assertAlmostEqual(out['W1_length'],np.sqrt(2/np.pi)*.5,places=6)
        expected=2*np.arccos(np.sqrt(2*.7*1.2/(.7**2+1.2**2)))
        self.assertAlmostEqual(out['fisher_rao'],expected,places=9)

    def test_probe_bound(self):
        p=np.array([.3,.4,.3]);r=np.array([.5,0.,.5]);eps=.2
        out=pair_statistics(p,r,eps);ell=eps*np.arange(3)
        for delta in (.02,.2,1.,10.):
            bound=abs(np.exp(-delta*ell)@(p-r))/delta
            self.assertLessEqual(bound,out['W1_length']+1e-13)

    def test_arbitrary_diagonal_phase_is_invisible(self):
        rng=np.random.default_rng(753)
        first=rng.normal(size=8)+1j*rng.normal(size=8);first/=np.linalg.norm(first)
        second=np.exp(1j*rng.normal(size=8))*first
        pair=pair_statistics(abs(first)**2,abs(second)**2,.1)
        self.assertLess(pair['W1_length'],1e-14)
        self.assertLess(quantum_fidelity(first,second),.99)

    def test_backward_state_against_dense(self):
        eps=.2;K=edge_hamiltonian(eps,1.,50);seed,_=prepare(K,2.)
        forward=state_at(K,seed,.6);reverse=expm(.6j*K.toarray())@seed
        np.testing.assert_allclose(forward.conj(),reverse,atol=3e-14)
        ell=eps*np.arange(51)
        self.assertAlmostEqual(length_velocity(K,forward,ell),-length_velocity(K,reverse,ell),places=12)

    def test_current_and_derivative_of_mean(self):
        eps=.15;K=edge_hamiltonian(eps,1.,60);seed,_=prepare(K,3.);t=.7
        psi=state_at(K,seed,t);ell=eps*np.arange(61);b=-K.diagonal(1)
        current=2*b*np.imag(psi[:-1].conj()*psi[1:])
        self.assertAlmostEqual(eps*sum(current),length_velocity(K,psi,ell),places=12)
        h=1e-5
        finite_difference=((abs(state_at(K,seed,t+h))**2-abs(state_at(K,seed,t-h))**2)@ell)/(2*h)
        self.assertAlmostEqual(finite_difference,length_velocity(K,psi,ell),places=8)

    def test_pure_trace_distance(self):
        p=np.array([1.,0.],complex);r=np.array([.6,.8],complex)
        diff=np.outer(p,p.conj())-np.outer(r,r.conj())
        direct=.5*abs(eigvalsh(diff)).sum()
        self.assertAlmostEqual(direct,np.sqrt(1-quantum_fidelity(p,r)),places=14)

    def test_projected_dynamics_not_closed(self):
        out=time_reversal(.1,200,beta=5.,time=1.,step=.2)
        self.assertLess(out['initial_W1'],1e-13)
        self.assertLess(out['quantum_fidelity'],.9)
        self.assertGreater(out['W1_after_common_step'],.01)
        self.assertAlmostEqual(out['quantum_fidelity'],out['fidelity_after_common_step'],places=10)

    def test_equal_mean_thermal_pair(self):
        out=match_mean(.05,360)['row']
        self.assertLess(out['mean_difference'],1e-9)
        self.assertGreater(out['W1_length'],.01)
        self.assertLess(out['W1_length'],.02)
        self.assertLess(abs(out['gaussian_W1_relative_deviation']),.01)
        self.assertLess(out['scipy_W1_difference'],1e-10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):pair_statistics(np.array([1.,0.]),np.array([.8,0.]),.1)
        with self.assertRaises(ValueError):pair_statistics(np.array([1.,0.]),np.array([1.,0.]),0.)
        with self.assertRaises(ValueError):pair_statistics(np.array([1.1,-.1]),np.array([1.,0.]),.1)


if __name__=='__main__':unittest.main()
