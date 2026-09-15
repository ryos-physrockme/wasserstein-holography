"""Independent analytic and numerical regression tests (no fitted parameters)."""
import unittest

import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.stats import wasserstein_distance

from src.krylov import (
    classical_mds_spectrum,
    discrete_w1_matrix,
    distance_geometry,
    evolve,
    links,
    lower_bound,
    matched_curve,
    observables,
    pairwise_example,
    physical_parameters,
    poisson_limit_probability,
    probability_current,
)


class KrylovTests(unittest.TestCase):
    def test_physical_b1(self):
        pars = physical_parameters(0.001, 0.0, 50)
        self.assertAlmostEqual(pars["a"]**2, np.exp(0.001)/0.001, places=9)
        self.assertAlmostEqual(links(pars["q"], pars["a"], 10)[0], pars["a"])

    def test_series_symbolically(self):
        q, t = sp.symbols("q t", real=True)
        nmax = 4
        b = [0]+[sp.sqrt(sum(q**k for k in range(n))) for n in range(1, nmax+1)]
        generator = sp.zeros(nmax+1)
        for n in range(1, nmax+1):
            generator[n, n-1], generator[n-1, n] = b[n], -b[n]
        terms = [sp.eye(nmax+1)[:, 0]]
        for j in range(1, 7):
            terms.append(generator*terms[-1]/j)
        c = sp.expand(sum(n*sum(terms[j][n]*t**j for j in range(7))**2
                          for n in range(nmax+1)))
        expected = {2: 1, 4: (q-1)/6, 6: (q-1)**2*(3*q+5)/180}
        for order, coefficient in expected.items():
            self.assertEqual(sp.simplify(c.coeff(t, order)-coefficient), 0)

    def test_sparse_against_dense(self):
        times = np.linspace(0, 0.7, 11)
        chi, generator = evolve(0.6, 1.2, times, 40)
        exact = expm(times[-1]*generator.toarray())[:, 0]
        np.testing.assert_allclose(chi[-1], exact, atol=2e-14)

    def test_invariants_and_bound(self):
        for q in (0.2, 0.6, 0.9):
            times = np.linspace(0, 3, 81)
            chi, generator = evolve(q, 1, times, 140)
            obs = observables(q, 1, chi, generator)
            for name in ("norm_error", "w1_error", "fisher_relative_error", "acceleration_relative_error"):
                self.assertLess(obs[name], 1e-10)
            self.assertGreaterEqual(np.min(obs["mean"]-lower_bound(q, 1, times)), -1e-10)
            self.assertGreaterEqual(np.min(times**2-obs["mean"]), -1e-10)

    def test_transport_against_independent_library(self):
        chi, _ = evolve(0.6, 1, np.linspace(0, 2, 21), 100)
        prob = chi[-1]**2
        n = np.arange(len(prob))
        direct = wasserstein_distance(n, [0], prob, [1])
        self.assertAlmostEqual(direct, prob@n, places=11)

    def test_sixth_order_excludes_exact_logcosh(self):
        q = 0.6
        exact_sixth = (1-q)**2*(3*q+5)/180
        matched_sixth = 2*(1-q)**2/45
        self.assertAlmostEqual(exact_sixth-matched_sixth, -(1-q)**3/60)
        self.assertLess(exact_sixth, matched_sixth)

    def test_pairwise_distance_not_mean_difference(self):
        result = pairwise_example()
        self.assertGreater(result["gap"], 0.007)
        self.assertGreater(result["P0_t2"], result["P0_t1"])

    def test_gap_equals_twice_cdf_reversal(self):
        times = np.linspace(0, 2.5, 101)
        chi, _ = evolve(0.2, 1.0, times, 180)
        probability = chi**2
        p, r = probability[80], probability[100]
        cdf_difference = np.cumsum(p-r)[:-1]
        w1 = np.sum(np.abs(cdf_difference))
        mean_difference = (r-p) @ np.arange(len(p))
        predicted_gap = 2*np.sum(np.maximum(-cdf_difference, 0.0))
        self.assertAlmostEqual(w1-mean_difference, predicted_gap, places=12)

    def test_probability_current_is_cdf_flux(self):
        times = np.linspace(0, 1.3, 27)
        chi, generator = evolve(0.6, 1.0, times, 80)
        row = 19
        velocity = generator @ chi[row]
        probability_derivative = 2*chi[row]*velocity
        cdf_derivative = np.cumsum(probability_derivative)[:-1]
        current = probability_current(0.6, 1.0, chi[[row]])[0]
        np.testing.assert_allclose(cdf_derivative, -current, atol=2e-13)

    def test_q1_poisson_limit_is_exact_line_metric(self):
        times = np.linspace(0, 3, 31)
        probability = poisson_limit_probability(1.0, times, 100)
        distance = discrete_w1_matrix(probability)
        mean = probability @ np.arange(probability.shape[1])
        np.testing.assert_allclose(distance, np.abs(mean[:, None]-mean[None, :]), atol=2e-12)
        eigenvalues = classical_mds_spectrum(distance)
        self.assertGreater(eigenvalues[0], 1.0)
        self.assertLess(np.max(np.abs(eigenvalues[1:])), 2e-10)

    def test_distance_geometry_detects_finite_q_backflow(self):
        x = np.linspace(0, 4, 201)
        q = 0.2
        times = x/np.sqrt(1-q)
        chi, _ = evolve(q, 1.0, times, 140)
        summary, _, _ = distance_geometry(chi**2)
        self.assertGreater(summary["max_pairwise_gap"], 0.02)
        self.assertGreater(summary["maximum_cdf_reversal"], 0.01)
        self.assertLess(summary["gap_identity_error"], 1e-10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            physical_parameters(0.1, 0.5, 10)
        with self.assertRaises(ValueError):
            evolve(0.9, 1, np.array([0, 1, 3]), 100)
        with self.assertRaises(ValueError):
            links(1, 1, 10)


if __name__ == "__main__":
    unittest.main()
