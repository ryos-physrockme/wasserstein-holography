"""Independent tests for thermal preparation, clocks, length and transport."""
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from src.geodesic import (thermal_saddle, classical_length, geodesic_growth,
                           edge_hamiltonian, prepare, thermal_evolution,
                           sector_parameters)
from src.krylov import physical_parameters


class GeodesicTests(unittest.TestCase):
    def test_saddle_equation_and_limits(self):
        self.assertEqual(thermal_saddle(0), np.pi/2)
        for s in (0.1, 1., 10., 100.):
            u=thermal_saddle(s)
            self.assertAlmostEqual(s*np.sin(u), np.pi-2*u, places=11)
        self.assertAlmostEqual(thermal_saddle(1e6)*1e6/np.pi, 1., places=5)

    def test_finite_cutoff_from_embedding(self):
        A=3.0; T=.7; tleft=.3; tright=.2
        kappa=2*np.pi*T
        XR=np.array([A, np.sqrt(A*A-1)*np.sinh(kappa*tright),
                      np.sqrt(A*A-1)*np.cosh(kappa*tright)])
        XL=np.array([A, np.sqrt(A*A-1)*np.sinh(kappa*tleft),
                     -np.sqrt(A*A-1)*np.cosh(kappa*tleft)])
        metric=np.array([-1.,-1.,1.])
        self.assertAlmostEqual(np.sum(metric*XR*XR),-1.)
        invariant=-np.sum(metric*XR*XL)
        computed=np.arccosh(invariant)-2*np.arccosh(A)
        self.assertAlmostEqual(computed, geodesic_growth(tleft+tright,T,A),places=12)

    def test_time_zero_radial_integral(self):
        # r=r_h*cosh(u) removes the endpoint singularity in dr/sqrt(r^2-r_h^2).
        A=7.; radial=2*quad(lambda u: 1.,0,np.arccosh(A))[0]
        self.assertAlmostEqual(radial,np.arccosh(2*A*A-1),places=13)

    def test_boundary_clock_factor(self):
        t=.3; T=.7
        self.assertAlmostEqual(geodesic_growth(2*t,T),2*np.log(np.cosh(2*np.pi*T*t)))
        self.assertGreater(abs(geodesic_growth(2*t,T)-geodesic_growth(t,T)),.1)

    def test_cutoff_convergence(self):
        times=np.linspace(0,1,21)
        errors=[np.max(abs(geodesic_growth(times,1,A)-geodesic_growth(times,1))) for A in (10.,100.,1000.)]
        self.assertLess(errors[1],errors[0]/90)
        self.assertLess(errors[2],errors[1]/90)

    def test_preparation_against_dense_exponential(self):
        K=edge_hamiltonian(.2,1.,50)
        seed,logz=prepare(K,3.,chunk=.17)
        raw=expm(-1.5*K.toarray())[:,0]
        np.testing.assert_allclose(seed,raw/np.linalg.norm(raw),atol=2e-14)
        self.assertAlmostEqual(logz,2*np.log(np.linalg.norm(raw)),places=12)
        other,_=prepare(K,3.,chunk=10.)
        np.testing.assert_allclose(seed,other,atol=2e-14)

    def test_evolution_against_dense(self):
        eps=.2; beta=1.; nmax=60; times=np.linspace(0,.4,9)
        run=thermal_evolution(eps,beta,times,nmax)
        K=edge_hamiltonian(eps,1.,nmax)
        exact=expm(-1j*times[-1]*K.toarray())@run['initial_seed']
        np.testing.assert_allclose(run['probability'][-1],abs(exact)**2,atol=3e-14)
        for k in ('norm_error','energy_drift','acceleration_identity_error','transport_identity_error'):
            self.assertLess(run['summary'][k],1e-10)

    def test_zero_temperature_preparation_parameter_means_infinite_temperature(self):
        # beta=0, not T=0: original delta_0 reference is recovered.
        K=edge_hamiltonian(.2,1.,30)
        seed,logz=prepare(K,0.)
        self.assertEqual(seed[0],1.)
        self.assertEqual(logz,0.)
        self.assertEqual(np.sum(seed[1:]**2),0.)

    def test_beta_nonzero_is_not_delta_reference(self):
        run=thermal_evolution(.2,1.,np.linspace(0,.2,3),60)
        self.assertGreater(run['length'][0],0.1)
        self.assertEqual(run['w1_thermal'][0],0.)

    def test_rescaling_preserves_probability(self):
        times=np.linspace(0,.3,7)
        first=thermal_evolution(.2,.5,times,60,omega=1.)
        second=thermal_evolution(.2,.25,times/2,60,omega=2.)
        np.testing.assert_allclose(first['probability'],second['probability'],atol=3e-14)

    def test_classical_growth_solves_energy_equation(self):
        beta=7.; omega=1.3; t=.3; h=1e-4
        ell=classical_length(t,beta,omega)
        dot=(classical_length(t+h,beta,omega)-classical_length(t-h,beta,omega))/(2*h)
        u=thermal_saddle(beta*omega)
        self.assertAlmostEqual(dot**2,4*omega**2*(np.sin(u)**2-np.exp(-ell)),places=8)

    def test_sector_normalizations(self):
        for charge in (0.,.2,.4):
            p=sector_parameters(.002,charge,10)
            old=physical_parameters(.002,charge,10)
            self.assertAlmostEqual(p['a'],old['a'],places=12)
            self.assertAlmostEqual(np.exp(-p['epsilon']),old['q'],places=14)
            # q_Heller^2=q, so 2|log q_Heller|=epsilon.
            self.assertAlmostEqual(-2*np.log(np.sqrt(old['q'])),p['epsilon'],places=14)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError): thermal_saddle(-1)
        with self.assertRaises(ValueError): geodesic_growth(1,1,.5)
        with self.assertRaises(ValueError): edge_hamiltonian(0,1,10)
        with self.assertRaises(ValueError): sector_parameters(.01,.5,10)


if __name__=='__main__':
    unittest.main()
