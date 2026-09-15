"""Independent exact algebra and finite-dimensional verification."""
import unittest
import numpy as np
from scipy.linalg import expm
from src.geodesic import edge_hamiltonian,prepare,thermal_saddle
from src.phase_space import (lowering,coherence_data,reconstruct,fidelity,
                            classical_momentum,ambiguous_pair,mixed_ambiguity)


class PhaseSpaceTests(unittest.TestCase):
    def test_matrix_algebra_with_endpoint(self):
        eps=.2;M=17;A=lowering(eps,M).toarray();Ad=A.T
        n=np.arange(M+1);q=np.exp(-eps);L=np.diag(eps*n)
        X=(A+Ad)/2;Y=(A-Ad)/(2j);proj=np.zeros_like(A);proj[-1,-1]=1
        np.testing.assert_allclose(Ad@A,np.diag(1-q**n),atol=3e-16)
        np.testing.assert_allclose(A@Ad,np.diag(1-q**(n+1))-(1-q**(M+1))*proj,atol=3e-16)
        np.testing.assert_allclose(L@X-X@L,-1j*eps*Y,atol=5e-16)
        np.testing.assert_allclose(L@Y-Y@L,1j*eps*X,atol=5e-16)
        np.testing.assert_allclose(X@X+Y@Y,np.eye(M+1)-.5*(1+q)*np.diag(q**n)-.5*(1-q**(M+1))*proj,atol=5e-16)
        np.testing.assert_allclose(X@Y-Y@X,.5j*((1-q)*np.diag(q**n)-(1-q**(M+1))*proj),atol=3e-16)

    def test_velocity_operator(self):
        eps=.3;omega=1.7;M=20;K=edge_hamiltonian(eps,omega,M).toarray()
        L=np.diag(eps*np.arange(M+1));A=lowering(eps,M).toarray();Y=(A-A.T)/(2j)
        np.testing.assert_allclose(1j*(K@L-L@K),2*omega*Y,atol=5e-15)

    def test_orthonormalization(self):
        eps=.2;M=8;q=np.exp(-eps);weights=np.r_[1.,np.cumprod(1-q**np.arange(1,M+1))]
        recurrence=np.diag(np.ones(M),1)+np.diag(1-q**np.arange(1,M+1),-1)
        D=np.diag(np.sqrt(weights))
        orth=np.linalg.solve(D,recurrence@D)
        A=lowering(eps,M).toarray()
        np.testing.assert_allclose(orth,A+A.T,atol=3e-16)

    def test_continuity_energy_and_quadratures(self):
        rng=np.random.default_rng(748);eps=.2;M=30;omega=1.3
        K=edge_hamiltonian(eps,omega,M)
        s=rng.normal(size=M+1)+1j*rng.normal(size=M+1);s/=np.linalg.norm(s)
        d=coherence_data(s,K,eps,omega)
        for name in ['continuity_error','velocity_error','energy_error','quadrature_identity_error']:
            self.assertLess(d[name],3e-14)
        self.assertGreaterEqual(d['variance_X_plus_Y'],-1e-14)

    def test_pure_reconstruction(self):
        rng=np.random.default_rng(991);s=rng.normal(size=20)+1j*rng.normal(size=20)
        s/=np.linalg.norm(s);c=s[:-1].conj()*s[1:]
        rebuilt,details=reconstruct(abs(s)**2,c,threshold=0.)
        self.assertAlmostEqual(fidelity(s,rebuilt),1.,places=13)
        self.assertAlmostEqual(details['retained_mass'],1.,places=14)

    def test_support_zero_is_not_bridged(self):
        with self.assertRaises(ValueError):
            reconstruct(np.array([.5,0.,.5]),np.zeros(2,complex),threshold=0.)

    def test_positive_cosine_reconstruction(self):
        rng=np.random.default_rng(55);p=rng.dirichlet(np.ones(15));angles=rng.uniform(-.5,.5,size=14)
        s=np.sqrt(p)*np.exp(1j*np.r_[0.,np.cumsum(angles)])
        c=s[:-1].conj()*s[1:]
        rebuilt,_=reconstruct(p,c,positive_cosine=True)
        self.assertAlmostEqual(fidelity(s,rebuilt),1.,places=14)

    def test_madelung_phase_equation(self):
        eps=.2;M=12;K=edge_hamiltonian(eps,1.,M)
        rng=np.random.default_rng(37);p=rng.dirichlet(np.ones(M+1));S=rng.uniform(-2.,2.,M+1)
        psi=np.sqrt(p)*np.exp(1j*S);rhs=-1j*(K@psi)
        phase_dot=np.imag(rhs/psi);b=-K.diagonal(1);c=np.cos(np.diff(S))
        formula=-K.diagonal().copy()
        formula[:-1]+=b*np.sqrt(p[1:]/p[:-1])*c
        formula[1:]+=b*np.sqrt(p[:-1]/p[1:])*c
        np.testing.assert_allclose(phase_dot,formula,atol=3e-14)

    def test_classical_hamilton_equations(self):
        eps=.1;beta=10.;omega=1.3;u=thermal_saddle(beta*omega);t=1.2;h=1e-5
        ell=-2*np.log(np.sin(u))+2*np.log(np.cosh(omega*t*np.sin(u)))
        p=float(classical_momentum(t,beta,omega));r=np.sqrt(1-np.exp(-ell))
        v=2*omega*r*np.sin(p);pdot=omega*np.exp(-ell)*np.cos(p)/r
        finite=(classical_momentum(t+h,beta,omega)-classical_momentum(t-h,beta,omega))/(2*h)
        self.assertAlmostEqual(v,2*omega*np.sin(u)*np.tanh(omega*t*np.sin(u)),places=13)
        # h=2Omega(1-r cos p), so -dh/dell=+Omega exp(-ell) cos p/r.
        self.assertAlmostEqual(float(finite),pdot,places=9)

    def test_ambiguity_survives_time_evolution(self):
        out=ambiguous_pair(nmax=40)
        self.assertAlmostEqual(out['fidelity_initial'],.25,places=14)
        for name in ['probability_max_difference','current_max_difference','energy_max_difference']:
            self.assertLess(out[name],2e-12)
        self.assertAlmostEqual(out['energy_first'],10.,places=13)
        self.assertAlmostEqual(out['full_coherence_reconstruction_fidelity'],1.,places=13)

    def test_antiunitary_relation_independent_dense(self):
        eps=.2;omega=1.1;M=12;B=omega/eps
        K=edge_hamiltonian(eps,omega,M).toarray();D=np.diag((-1.)**np.arange(M+1))
        np.testing.assert_allclose(D@K@D,4*B*np.eye(M+1)-K,atol=1e-14)
        rng=np.random.default_rng(13);s=rng.normal(size=M+1)+1j*rng.normal(size=M+1);s/=np.linalg.norm(s)
        t=.37;left=expm(-1j*K*t)@D@s.conj()
        right=np.exp(-4j*B*t)*D@(expm(-1j*K*t)@s).conj()
        np.testing.assert_allclose(left,right,atol=4e-15)

    def test_mixed_ambiguity(self):
        d=mixed_ambiguity();self.assertGreater(d['min_eigenvalue'],0.)
        self.assertEqual(d['diagonal_difference'],0.)
        self.assertEqual(d['nearest_coherence_difference'],0.)
        self.assertAlmostEqual(d['trace_distance'],.3,places=14)

    def test_half_line_shift_not_unitary(self):
        M=8;S=np.diag(np.ones(M),-1)
        self.assertEqual((S@S.T)[0,0],0.)
        self.assertEqual((S.T@S)[0,0],1.)


if __name__=='__main__':unittest.main()
