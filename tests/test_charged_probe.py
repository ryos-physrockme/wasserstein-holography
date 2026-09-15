import unittest
import numpy as np
from src.charged_probe import (berkooz_Bs,fixed_charge_even_odd,gst_even_odd_from_Q,
                               asymmetry_log_ratio,light_probe_log_ratio,boundary_green_ratio)

class ChargedProbeTests(unittest.TestCase):
    def test_B_decomposition_and_GST_dictionary(self):
        N=3000.;pB=47.;s=35.;d=9.;Q=.21;n=-Q*N
        pM=(s+d)/2;pbar=(s-d)/2
        B1,B2=berkooz_Bs(N,n,pB,pM,pbar)
        even,odd=fixed_charge_even_odd(Q,pB,s,d,N)
        ge,go,mu=gst_even_odd_from_Q(Q,pB,s,d,N)
        self.assertAlmostEqual(B1,even+odd,places=14)
        self.assertAlmostEqual(B2,even-odd,places=14)
        self.assertAlmostEqual(even,ge,places=14)
        self.assertAlmostEqual(odd,go,places=14)
        self.assertAlmostEqual(np.tanh(mu),-2*Q,places=14)

    def test_charge_parities(self):
        args=(80.,25.,7.,4000.)
        ep,op=fixed_charge_even_odd(.17,*args)
        em,om=fixed_charge_even_odd(-.17,*args)
        self.assertAlmostEqual(ep,em,places=14)
        self.assertAlmostEqual(op,-om,places=14)
        _,oflip=fixed_charge_even_odd(.17,args[0],args[1],-args[2],args[3])
        self.assertAlmostEqual(op,-oflip,places=14)

    def test_neutral_probe_is_blind_to_odd_sector(self):
        _,odd=fixed_charge_even_odd(.22,60.,40.,0.,5000.)
        self.assertEqual(odd,0.)
        self.assertEqual(asymmetry_log_ratio(.22,40.,0.,5000.),0.)

    def test_light_unit_probe_matches_boundary_green_asymmetry(self):
        for Q in (-.3,-.1,.1,.3):
            ratio=np.exp(light_probe_log_ratio(Q,1.))
            self.assertAlmostEqual(ratio,boundary_green_ratio(Q),places=14)

    def test_asymmetry_odd_in_background_and_probe_charge(self):
        val=asymmetry_log_ratio(.2,30.,5.,10000.)
        self.assertAlmostEqual(val,-asymmetry_log_ratio(-.2,30.,5.,10000.),places=14)
        self.assertAlmostEqual(val,-asymmetry_log_ratio(.2,30.,-5.,10000.),places=14)

    def test_invalid_charge(self):
        with self.assertRaises(ValueError): fixed_charge_even_odd(.5,1,1,1,10)

if __name__=='__main__': unittest.main()
