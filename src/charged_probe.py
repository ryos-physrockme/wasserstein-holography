"""Charge-even/odd decomposition of complex-SYK probe factors.

Input formulas:
- Berkooz, Narovlansky, Raj, arXiv:2006.13983v2, Eqs. (7.21)-(7.23), (7.43).
- Gubankova, Sachdev, Tarnopolsky, arXiv:2512.05294v2, Eqs. (3.48)-(3.50).

Conventions:
Berkooz p_B counts p_B fermions and p_B antifermions in the Hamiltonian,
whereas GST p_G counts the total Hamiltonian size, so p_G=2 p_B.
The fixed sector has physical charge density Q=-n/N.
The probe has size s=p_M+pbar_M and charge d=p_M-pbar_M.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def berkooz_Bs(N: float, n: float, pB: float, pM: float, pbar: float):
    if N <= 0 or abs(2*n) >= N:
        raise ValueError("require N>0 and |2n|<N")
    B1=(pB*pM/N)*(2*n)/(N-2*n) - (pB*pbar/N)*(2*n)/(N+2*n)
    B2=-(pB*pM/N)*(2*n)/(N+2*n) + (pB*pbar/N)*(2*n)/(N-2*n)
    return float(B1),float(B2)


def fixed_charge_even_odd(Q: float, pB: float, probe_size: float, probe_charge: float, N: float):
    if abs(Q) >= .5 or N <= 0:
        raise ValueError("require |Q|<1/2 and N>0")
    den=1-4*Q*Q
    even=4*pB*probe_size*Q*Q/(N*den)
    odd=-2*pB*probe_charge*Q/(N*den)
    return float(even),float(odd)


def gst_even_odd_from_Q(Q: float, pB: float, probe_size: float, probe_charge: float, N: float):
    """GST Eq.(3.50), after p_G=2 p_B and tanh(mu)=-2Q."""
    if abs(Q) >= .5:
        raise ValueError("require |Q|<1/2")
    mu=np.arctanh(-2*Q)
    pG=2*pB
    tilde_lambda=pG*probe_size/N
    lambda_mu=pG*probe_charge/N
    even=-tilde_lambda/4*(1-np.cosh(2*mu))
    odd=lambda_mu/4*np.sinh(2*mu)
    return float(even),float(odd),float(mu)


def asymmetry_log_ratio(Q: float, probe_size: float, probe_charge: float, N: float):
    """BNR Eq.(7.43) rewritten with Q=-n/N."""
    if abs(Q) >= .5 or N <= 0:
        raise ValueError("require |Q|<1/2 and N>0")
    return float(probe_charge*(np.log((1-2*Q)/(1+2*Q))
                               + 4*Q*probe_size/(N*(1-4*Q*Q))))


def light_probe_log_ratio(Q: float, probe_charge: float):
    if abs(Q) >= .5:
        raise ValueError("require |Q|<1/2")
    return float(probe_charge*np.log((1-2*Q)/(1+2*Q)))


def boundary_green_ratio(Q: float):
    """GST Eq.(2.4): G(0+)/G(beta-) for a unit-charge fermion."""
    if abs(Q) >= .5:
        raise ValueError("require |Q|<1/2")
    return float((.5-Q)/(.5+Q))


def run(output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    N=10000.;pB=100.;probe_size=80.;probe_charge=20.
    qs=np.linspace(-.45,.45,181)
    rows=[]
    for Q in qs:
        n=-Q*N
        pM=(probe_size+probe_charge)/2
        pbar=(probe_size-probe_charge)/2
        B1,B2=berkooz_Bs(N,n,pB,pM,pbar)
        even,odd=fixed_charge_even_odd(Q,pB,probe_size,probe_charge,N)
        ge,go,mu=gst_even_odd_from_Q(Q,pB,probe_size,probe_charge,N)
        rows.append({"Q":Q,"B1":B1,"B2":B2,"B_even":even,"B_odd":odd,
                     "GST_even":ge,"GST_odd":go,"mu_saddle":mu,
                     "even_match_error":abs(even-ge),"odd_match_error":abs(odd-go),
                     "decomposition_error":max(abs(B1-even-odd),abs(B2-even+odd)),
                     "log_asymmetry":asymmetry_log_ratio(Q,probe_size,probe_charge,N),
                     "light_log_asymmetry":light_probe_log_ratio(Q,probe_charge)})
    with (output/"scan.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    testQ=.23
    evenp,oddp=fixed_charge_even_odd(testQ,pB,probe_size,probe_charge,N)
    evenm,oddm=fixed_charge_even_odd(-testQ,pB,probe_size,probe_charge,N)
    neutral_even,neutral_odd=fixed_charge_even_odd(testQ,pB,probe_size,0.,N)
    light_ratio=np.exp(light_probe_log_ratio(testQ,1.))
    report={
      "conventions":{"N":N,"p_B":pB,"p_GST":2*pB,"probe_size":probe_size,"probe_charge":probe_charge},
      "max_even_match_error":max(r["even_match_error"] for r in rows),
      "max_odd_match_error":max(r["odd_match_error"] for r in rows),
      "max_decomposition_error":max(r["decomposition_error"] for r in rows),
      "parity":{"even_Q_flip_error":abs(evenp-evenm),"odd_Q_flip_error":abs(oddp+oddm)},
      "neutral_probe":{"B_even":neutral_even,"B_odd":neutral_odd,"log_asymmetry":asymmetry_log_ratio(testQ,probe_size,0.,N)},
      "light_unit_probe":{"Q":testQ,"ratio_from_BNR":light_ratio,"GST_boundary_ratio":boundary_green_ratio(testQ),
                          "difference":abs(light_ratio-boundary_green_ratio(testQ))},
      "finite_size_correction":{"Q":testQ,"full_log_ratio":asymmetry_log_ratio(testQ,probe_size,1.,N),
                                "light_log_ratio":light_probe_log_ratio(testQ,1.)}
    }
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    ax.plot(qs,[r['B_even'] for r in rows],label='charge-even part')
    ax.plot(qs,[r['B_odd'] for r in rows],label='charge-odd part')
    ax.set(xlabel=r'background charge density $Q$',ylabel=r'$B_{\rm even}, B_{\rm odd}$')
    ax.legend();fig.tight_layout();fig.savefig(output/'even_odd.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    size=80.;charge=1.
    full=np.array([asymmetry_log_ratio(Q,size,charge,N) for Q in qs])
    light=np.array([light_probe_log_ratio(Q,charge) for Q in qs])
    ax.plot(qs,full,label='fixed-charge chord result')
    ax.plot(qs,light,'--',label='light-probe limit')
    ax.set(xlabel=r'background charge density $Q$',ylabel=r'$\log R_M$')
    ax.legend();fig.tight_layout();fig.savefig(output/'asymmetry.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('results/charged_probe'))
    run(ap.parse_args().output)
