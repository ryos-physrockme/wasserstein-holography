"""Self-contained exact checks used in the literature reassessment."""
import math
import unittest
from src.metric_benchmarks import instanton_checks, gaussian_w1_tangent

class MetricBenchmarkTests(unittest.TestCase):
    def test_instanton_integrals_and_vanishing_cubic(self):
        values=instanton_checks()
        self.assertEqual(values['normalization'],'1')
        self.assertEqual(values['second_radial_moment'],'2')
        self.assertEqual(values['fisher_translation_component_at_unit_scale'],'16/5')
        self.assertEqual(values['fisher_scale_component_at_unit_scale'],'16/5')
        self.assertEqual(values['cubic_translation_translation_scale'],'0')
        self.assertEqual(values['cubic_scale_scale_scale'],'0')

    def test_gaussian_w1_is_not_quadratic(self):
        f=gaussian_w1_tangent
        self.assertEqual(f(1,0),1)
        self.assertAlmostEqual(f(0,1),math.sqrt(2/math.pi))
        self.assertAlmostEqual(f(1,1),f(1,-1))
        self.assertGreater(abs(2*f(1,1)**2-2*f(1,0)**2-2*f(0,1)**2),.1)

    def test_w1_homogeneity_and_invalid_input(self):
        f=gaussian_w1_tangent
        self.assertAlmostEqual(f(3,6),3*f(1,2))
        with self.assertRaises(ValueError): f(math.nan,1)

if __name__=='__main__': unittest.main()
