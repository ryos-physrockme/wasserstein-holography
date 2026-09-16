"""Analytic baseline checks, not a new holographic dictionary.

Fisher convention: g_ij = E[partial_i log p partial_j log p].
Instanton probability on R^4: p=6/pi^2 * rho^4/(|x-a|^2+rho^2)^4.
The checks below use rho=1,a=0; covariance restores all parameter values.
The cubic vanishing is already BNT hep-th/0108122 Eq.(2.38).
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sympy as sp


def instanton_checks() -> dict[str, str]:
    """Integrate the independent rotationally invariant coefficients exactly."""
    r = sp.symbols('r', positive=True)
    radial = 12*r**3/(1+r**2)**4  # includes the S^3 angular volume
    def avg(f):
        return sp.simplify(sp.integrate(radial*f, (r, 0, sp.oo)))
    values = {
        'normalization': avg(1),
        'second_radial_moment': avg(r**2),
        'fisher_translation_component_at_unit_scale': avg(16*r**2/(1+r**2)**2),
        'fisher_scale_component_at_unit_scale': avg(16*(r**2-1)**2/(1+r**2)**2),
        'cubic_translation_translation_scale': avg(64*r**2*(r**2-1)/(1+r**2)**3),
        'cubic_scale_scale_scale': avg(64*(r**2-1)**3/(1+r**2)**3),
    }
    expected = [1, 2, sp.Rational(16, 5), sp.Rational(16, 5), 0, 0]
    for (key, value), target in zip(values.items(), expected):
        if sp.simplify(value-target) != 0:
            raise AssertionError(f'{key}: {value} != {target}')
    return {key: str(value) for key, value in values.items()}


def gaussian_w1_tangent(location_velocity: float, scale_velocity: float) -> float:
    """E|u+v Z| for Z~N(0,1): the W1 norm on a Gaussian location-scale family."""
    u,v=float(location_velocity),abs(float(scale_velocity))
    if not math.isfinite(u) or not math.isfinite(v):
        raise ValueError('finite velocities required')
    if v == 0:
        return abs(u)
    return v*math.sqrt(2/math.pi)*math.exp(-u*u/(2*v*v))+abs(u)*math.erf(abs(u)/(math.sqrt(2)*v))


def run(output: Path) -> dict:
    out = {'meaning': 'Standard exact benchmarks; no numerical inference of a bulk theory.',
           'source_fisher_and_cubic': 'Blau-Narain-Thompson hep-th/0108122 Eqs.(2.12)-(2.16),(2.38)',
           'instanton': instanton_checks()}
    # A quadratic norm must obey the parallelogram identity.
    u=gaussian_w1_tangent(1,0);v=gaussian_w1_tangent(0,1)
    residual=2*gaussian_w1_tangent(1,1)**2-2*u*u-2*v*v
    out['gaussian_W1_parallelogram_residual']=residual
    out['instanton_W2_squared']='|a-b|^2+2*(rho-sigma)^2 for the Euclidean ground cost'
    out['instanton_Fisher_line_element']='(16/5)*(|da|^2+drho^2)/rho^2'
    out['instanton_all_alpha_connections']='Levi-Civita, since the cubic tensor is zero; curvature is not zero'
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/metric_benchmarks'))
    run(parser.parse_args().output)
