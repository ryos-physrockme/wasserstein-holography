"""Checks for bilocal-probe reconstruction of neighboring coherences."""
import unittest
import numpy as np
from src.geodesic import edge_hamiltonian
from src.boundary_probe import (evolved_state, probe_value, direct_link_generators,
                                exact_derivatives, derivative_reconstruction,
                                coefficient_uniqueness_demo)

class BoundaryProbeTests(unittest.TestCase):
    def test_exact_generator_identities_random_state(self):
        rng=np.random.default_rng(753)
        eps=.17;omega=1.2;nmax=24;K=edge_hamiltonian(eps,omega,nmax)
        psi=rng.normal(size=nmax+1)+1j*rng.normal(size=nmax+1)
        psi/=np.linalg.norm(psi)
        for delta in (.1,.7,2.3):
            direct=direct_link_generators(psi,K,eps,delta,omega)
            deriv=exact_derivatives(psi,K,eps,delta)
            rec=derivative_reconstruction(**deriv,epsilon=eps,delta=delta,omega=omega)
            self.assertAlmostEqual(direct['current_generator'],rec['current_generator'],places=12)
            self.assertAlmostEqual(direct['hopping_generator'],rec['hopping_generator'],places=12)
            self.assertLess(abs(direct['coherence_generator']-rec['coherence_generator']),1e-13)

    def test_time_derivative_by_finite_difference(self):
        eps=.1;beta=3.;time=.7;nmax=90;delta=.8;K=edge_hamiltonian(eps,1.,nmax)
        psi=evolved_state(K,beta,time)
        direct=direct_link_generators(psi,K,eps,delta)
        h=2e-5
        gp=probe_value(evolved_state(K,beta,time+h),eps,delta)
        gm=probe_value(evolved_state(K,beta,time-h),eps,delta)
        predicted=(gp-gm)/(2*h)/(np.exp(-eps*delta)-1)
        self.assertAlmostEqual(predicted,direct['current_generator'],places=7)

    def test_beta_derivative_by_finite_difference(self):
        eps=.1;beta=3.;time=.7;nmax=90;delta=.8;K=edge_hamiltonian(eps,1.,nmax)
        psi=evolved_state(K,beta,time)
        direct=direct_link_generators(psi,K,eps,delta)
        h=1e-3
        gp=probe_value(evolved_state(K,beta+h,time),eps,delta)
        gm=probe_value(evolved_state(K,beta-h,time),eps,delta)
        db=(gp-gm)/(2*h);G=probe_value(psi,eps,delta);E=float(np.vdot(psi,K@psi).real)
        z=np.exp(-eps*delta);B=1/eps
        predicted=2*((E-2*B)*G-db)/(1+z)
        self.assertLess(abs(predicted-direct['hopping_generator']),2e-7)

    def test_delta_zero_limits_directly(self):
        eps=.08;beta=4.;time=.6;nmax=120;K=edge_hamiltonian(eps,1.,nmax)
        psi=evolved_state(K,beta,time)
        direct=direct_link_generators(psi,K,eps,0.)
        length_velocity=eps*direct['current_generator']
        n=np.arange(nmax+1);ell=eps*n
        velocity=-1j*(K@psi)
        explicit=float(2*np.real(np.vdot(psi*ell,velocity)))
        self.assertAlmostEqual(length_velocity,explicit,places=12)
        self.assertLess(abs(direct['hopping_generator']-(direct['energy']-2/eps)),1e-12)

    def test_finite_polynomial_uniqueness_demo(self):
        out=coefficient_uniqueness_demo(nlinks=7)
        self.assertLess(out['max_error'],1e-10)
        self.assertGreater(out['condition_number'],1e5)

    def test_probe_value_delta_zero(self):
        eps=.2;K=edge_hamiltonian(eps,1.,50);psi=evolved_state(K,2.,.4)
        self.assertAlmostEqual(probe_value(psi,eps,0.),1.,places=12)

if __name__=='__main__': unittest.main()
