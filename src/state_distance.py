"""Distances between length measurement distributions of thermal chord states.

Uses the common basis and Hamiltonian of src/geodesic.py, not a new Lanczos
basis per temperature. All comparisons are within one fixed epsilon, Omega.
Real-time vectors are never renormalized. Normalized overlap ratios are used
only when reporting classical/quantum fidelities; norm errors are recorded.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq, linprog
from scipy.sparse.linalg import expm_multiply
from scipy.stats import wasserstein_distance

try:
    from .geodesic import edge_hamiltonian, prepare
except ImportError:  # python src/state_distance.py
    from geodesic import edge_hamiltonian, prepare


def state_at(K, seed: np.ndarray, time: float) -> np.ndarray:
    """A single real-time state; negative times are allowed."""
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    return expm_multiply(-1j*time*K, seed,
                         traceA=-1j*time*float(K.diagonal().sum()))


def length_statistics(psi: np.ndarray, lengths: np.ndarray) -> dict:
    """Moments on ell_n=epsilon*n; no probability renormalization."""
    prob = abs(psi)**2
    mean = float(prob@lengths)
    return {"mean": mean,
            "variance": float(prob@((lengths-mean)**2)),
            "mad": float(prob@abs(lengths-mean)),
            "norm_error": float(abs(prob.sum()-1)),
            "tail_probability": float(prob[-20:].sum())}


def pair_statistics(p: np.ndarray, r: np.ndarray, epsilon: float) -> dict:
    """W1, explicit optimal Lipschitz function, and Fisher--Rao distance.

    Tail sums avoid cancellation of CDFs close to one. The optimal function
    has f(0)=0 and consecutive slopes sign(S_p-S_r), where S(k)=Pr(n>k).
    It saturates the discrete Kantorovich dual even for tiny norm errors.
    """
    p, r = np.asarray(p, float), np.asarray(r, float)
    if (p.ndim != 1 or p.shape != r.shape or len(p)<2
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r))
            or np.any(p<0) or np.any(r<0) or epsilon<=0 or not np.isfinite(epsilon)):
        raise ValueError("finite nonnegative equal-sized vectors and epsilon>0 required")
    if max(abs(p.sum()-1),abs(r.sum()-1))>1e-8:
        raise ValueError("probabilities must sum to one within 1e-8")
    lengths = epsilon*np.arange(len(p))
    tail_difference = np.cumsum((p-r)[::-1])[::-1][1:]
    w1 = float(epsilon*abs(tail_difference).sum())
    f = np.r_[0., np.cumsum(epsilon*np.sign(tail_difference))]
    mean_p,mean_r=float(p@lengths),float(r@lengths)
    mp=float(p@abs(lengths-mean_p));mr=float(r@abs(lengths-mean_r))
    overlap = float(np.sqrt(p*r).sum()/np.sqrt(p.sum()*r.sum()))
    fisher = float(2*np.arccos(np.clip(overlap,0.,1.)))
    upper = abs(mean_p-mean_r)+mp+mr
    return {"W1_length":w1, "mean_difference":abs(mean_p-mean_r),
            "excess_over_mean_difference":w1-abs(mean_p-mean_r),
            "concentration_upper_bound":upper,
            "dual_residual":abs(w1-float((p-r)@f)),
            "optimal_function":f, "fisher_rao":fisher,
            "classical_overlap":overlap}


def length_velocity(K, psi: np.ndarray, lengths: np.ndarray) -> float:
    return float(2*np.real(np.vdot(psi*lengths, -1j*(K@psi))))


def quantum_fidelity(first: np.ndarray, second: np.ndarray) -> float:
    den=float(np.vdot(first,first).real*np.vdot(second,second).real)
    return float(np.clip(abs(np.vdot(first,second))**2/den,0.,1.))


def match_mean(epsilon: float, nmax: int, beta_a: float=5., beta_b: float=10.,
               omega: float=1.) -> dict:
    """Match <ell> of (beta_a,t_a) with (beta_b,0), solving for t_a.

    The matching is an experimental constraint, NOT a fitted gravitational
    time or length normalization. The two states share the same Hamiltonian.
    """
    K=edge_hamiltonian(epsilon,omega,nmax)
    lengths=epsilon*np.arange(nmax+1)
    seed_a,_=prepare(K,beta_a,chunk=1/omega)
    seed_b,_=prepare(K,beta_b,chunk=1/omega)
    target=float(seed_b**2@lengths)
    def objective(t):
        return float(abs(state_at(K,seed_a,t))**2@lengths)-target
    right=1/omega
    while objective(right)<=0:
        right*=2
        if right>32/omega: raise RuntimeError("equal-mean root not bracketed")
    time_a=float(brentq(objective,0.,right,xtol=2e-12))
    psi_a=state_at(K,seed_a,time_a)
    p,r=abs(psi_a)**2,seed_b**2
    A=length_statistics(psi_a,lengths); B=length_statistics(seed_b,lengths)
    pair=pair_statistics(p,r,epsilon)
    sa,sb=np.sqrt(A['variance']),np.sqrt(B['variance'])
    gaussian_w1=float(np.sqrt(2/np.pi)*abs(sa-sb))
    gaussian_fr=float(2*np.arccos(np.sqrt(2*sa*sb/(sa*sa+sb*sb))))
    deltas=np.geomspace(.02,10.,241)
    probe_diff=np.exp(-deltas[:,None]*lengths)@(p-r)
    probe_bound=abs(probe_diff)/deltas
    ibest=int(np.argmax(probe_bound))
    row={"epsilon":epsilon,"q":float(np.exp(-epsilon)),"omega":omega,
         "beta_a":beta_a,"beta_b":beta_b,"time_a":time_a,"time_b":0.,"nmax":nmax,
         "mean_a":A['mean'],"mean_b":B['mean'],"sigma_a":float(sa),"sigma_b":float(sb),
         **{k:v for k,v in pair.items() if k!='optimal_function'},
         "W1_over_sqrt_epsilon":pair['W1_length']/np.sqrt(epsilon),
         "gaussian_W1":gaussian_w1,"gaussian_fisher_rao":gaussian_fr,
         "gaussian_W1_relative_deviation":pair['W1_length']/gaussian_w1-1.,
         "absolute_deviation_observable_gap":abs(A['mad']-B['mad']),
         "probe_delta_1_difference":float(np.exp(-lengths)@(p-r)),
         "probe_best_delta_in_scan":float(deltas[ibest]),
         "probe_best_lower_bound":float(probe_bound[ibest]),
         "quantum_fidelity":quantum_fidelity(psi_a,seed_b),
         "norm_error":max(A['norm_error'],B['norm_error']),
         "tail_probability":max(A['tail_probability'],B['tail_probability']),
         "scipy_W1_difference":abs(pair['W1_length']-float(wasserstein_distance(lengths,lengths,p,r)))}
    return {"row":row,"probability_a":p,"probability_b":r,"lengths":lengths,
            "optimal_function":pair['optimal_function'],"probe_deltas":deltas,
            "probe_lower_bounds":probe_bound}


def time_reversal(epsilon: float, nmax: int, beta: float=10., time: float=1.,
                  step: float=.2, omega: float=1.) -> dict:
    """Compare psi(+t) and psi(-t), and a common subsequent evolution."""
    K=edge_hamiltonian(epsilon,omega,nmax)
    lengths=epsilon*np.arange(nmax+1)
    seed,_=prepare(K,beta,chunk=1/omega)
    forward=state_at(K,seed,time)
    reverse=state_at(K,seed,-time)  # independent numerical exponential
    pair=pair_statistics(abs(forward)**2,abs(reverse)**2,epsilon)
    later_f=state_at(K,seed,time+step)
    later_r=state_at(K,seed,-time+step)
    later=pair_statistics(abs(later_f)**2,abs(later_r)**2,epsilon)
    probabilities=[abs(z)**2 for z in (forward,reverse,later_f,later_r)]
    return {"epsilon":epsilon,"beta":beta,"omega":omega,"nmax":nmax,
            "time":time,"common_step":step,
            "initial_W1":pair['W1_length'],"initial_fisher_rao":pair['fisher_rao'],
            "conjugation_error":float(np.max(abs(reverse-forward.conj()))),
            "probability_L1_difference":float(np.sum(abs(probabilities[0]-probabilities[1]))),
            "quantum_fidelity":quantum_fidelity(forward,reverse),
            "pure_state_trace_distance":float(np.sqrt(1-quantum_fidelity(forward,reverse))),
            "length_velocity_forward":length_velocity(K,forward,lengths),
            "length_velocity_reverse":length_velocity(K,reverse,lengths),
            "W1_after_common_step":later['W1_length'],
            "fidelity_after_common_step":quantum_fidelity(later_f,later_r),
            "norm_error":max(float(abs(p.sum()-1)) for p in probabilities),
            "tail_probability":max(float(p[-20:].sum()) for p in probabilities)}


def run(output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    cases=[.05,.02,.01,.005]
    rows, reversals, detailed=[],[],[]
    for epsilon in cases:
        nmax=int(np.ceil(18/epsilon))
        pair=match_mean(epsilon,nmax)
        doubled=match_mean(epsilon,2*nmax)
        row=pair['row']; drow=doubled['row']
        row.update({"doubled_cutoff_W1_error":abs(row['W1_length']-drow['W1_length']),
                    "doubled_cutoff_time_error":abs(row['time_a']-drow['time_a']),
                    "doubled_cutoff_fisher_error":abs(row['fisher_rao']-drow['fisher_rao']),
                    "doubled_cutoff_mean_error":abs(row['mean_a']-drow['mean_a']),
                    "doubled_cutoff_probability_L1":max(float(abs(pair[key]-doubled[key][:nmax+1]).sum())
                                                        for key in ('probability_a','probability_b'))})
        rev=time_reversal(epsilon,nmax)
        rev2=time_reversal(epsilon,2*nmax)
        rev['doubled_cutoff_fidelity_error']=abs(rev['quantum_fidelity']-rev2['quantum_fidelity'])
        rev['doubled_cutoff_W1_after_error']=abs(rev['W1_after_common_step']-rev2['W1_after_common_step'])
        rows.append(row);reversals.append(rev);detailed.append(pair)
        print(json.dumps({'equal_mean':row,'time_reversal':rev}),flush=True)
    # A completely independent transport linear program on a small toy example.
    p=np.array([.25,.5,.25,0.]);r=np.array([.5,0.,.5,0.]);h=.2
    grid=np.arange(4)*h;cost=abs(grid[:,None]-grid).ravel()
    constraints=np.zeros((8,16))
    for i in range(4):constraints[i,4*i:4*(i+1)]=1;constraints[4+i,i::4]=1
    lp=linprog(cost,A_eq=constraints,b_eq=np.r_[p,r],bounds=(0,None),method='highs')
    if not lp.success: raise RuntimeError(lp.message)
    exact=pair_statistics(p,r,h)
    summary={'conventions':'ell_n=epsilon*n; Omega=1; common Hamiltonian per pair; no real-time normalization',
             'input':'src/geodesic.py; operator dictionary conditional on arXiv:2412.17785v2 Eqs.(5)-(8)',
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
             'equal_mean_pairs':rows,'time_reversal_pairs':reversals,
             'linear_program_test':{'cost':float(lp.fun),'cdf_cost':exact['W1_length'],'dual_residual':exact['dual_residual']}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for name,data in [('equal_mean.csv',rows),('time_reversal.csv',reversals)]:
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # Save distributions and the actual optimal test functions, not only moments.
    for d in detailed:
        name=f"pair_epsilon_{d['row']['epsilon']:.3f}.csv"
        np.savetxt(output/name,np.column_stack([d['lengths'],d['probability_a'],d['probability_b'],d['optimal_function']]),
                   delimiter=',',header='length,probability_a,probability_b,optimal_lipschitz_function',comments='')
    lines=[r'\begin{tabular}{rrrrrr}\toprule',
           r'$\epsilon$ & $t_A\Omega$ & $\mathsf D_{\ell}$ & $\mathsf D_{\ell}/\sqrt{\epsilon}$ & $d_{\rm FR}$ & Gaussian近似の偏差 [\%]\\\midrule']
    for r in rows:
        lines.append(f"{r['epsilon']:.3f} & {r['time_a']:.6f} & {r['W1_length']:.6f} & {r['W1_over_sqrt_epsilon']:.6f} & {r['fisher_rao']:.6f} & {100*r['gaussian_W1_relative_deviation']:.3f} "+chr(92)*2)
    lines.append(r'\bottomrule\end{tabular}')
    (output/'equal_mean_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{rrrrr}\toprule',r'$\epsilon$ & $\mathsf D_{\ell}(+t,-t)$ & $\mathcal F_{\rm q}$ & $\dot m(+t)/\Omega$ & 共通時間発展後の$\mathsf D_{\ell}$\\\midrule']
    for r in reversals:
        lines.append(f"{r['epsilon']:.3f} & {r['initial_W1']:.1e} & {r['quantum_fidelity']:.6g} & {r['length_velocity_forward']:.6f} & {r['W1_after_common_step']:.6f} "+chr(92)*2)
    lines.append(r'\bottomrule\end{tabular}')
    (output/'time_reversal_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Each chart has its own figure. Matplotlib's default color cycle is used.
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    eps=np.array(cases);wv=np.array([r['W1_length'] for r in rows])
    ax.loglog(eps,wv,'o-',label='Exact length-distribution distance')
    ax.loglog(eps,[r['gaussian_W1'] for r in rows],'x--',label='Gaussian formula from measured widths')
    ax.set(xlabel=r'$\epsilon$',ylabel=r'$\mathsf{D}_{\ell}$')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'equal_mean_scaling.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    for d in detailed:
        eps=d['row']['epsilon'];mu=d['row']['mean_a'];z=(d['lengths']-mu)/np.sqrt(eps)
        ax.plot(z,d['probability_a']/np.sqrt(eps),label=fr'$\beta_A\Omega=5,\ \epsilon={eps}$')
        if eps==cases[-1]: ax.plot(z,d['probability_b']/np.sqrt(eps),'--',label=r'$\beta_B\Omega=10,\ \epsilon=0.005$')
    ax.set(xlim=(-4,4),xlabel=r'$(\ell-\langle\ell\rangle)/\sqrt{\epsilon}$',ylabel='Probability density in the rescaled variable')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'rescaled_distributions.pdf');plt.close(fig)
    d=detailed[1]; fig,ax=plt.subplots(figsize=(6.3,4.0))
    mu=d['row']['mean_a'];ax.plot(d['lengths'],d['optimal_function'])
    mask=(d['lengths']>mu-.8)&(d['lengths']<mu+.8)
    vals=d['optimal_function'][mask]
    ax.set(xlim=(mu-.8,mu+.8),ylim=(float(vals.min())-.06,float(vals.max())+.06),
           xlabel=r'$\ell$',ylabel=r'Optimal $f(\ell)$ with $f(0)=0$')
    fig.tight_layout();fig.savefig(output/'optimal_observable.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    ax.semilogx(d['probe_deltas'],d['probe_lower_bounds']/d['row']['W1_length'])
    ax.set(xlabel=r'Probe parameter $\Delta$',ylabel=r'$|G_A(\Delta)-G_B(\Delta)|/(\Delta\mathsf{D}_{\ell})$')
    fig.tight_layout();fig.savefig(output/'probe_bound.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/state_distance'))
    run(parser.parse_args().output)
