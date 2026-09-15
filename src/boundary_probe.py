"""Recover neighboring chord coherences from derivatives of the bilocal probe.

Conventions:
  K = 2 B I - B(A+A^dagger),
  A|n> = sqrt(1-q^n)|n-1>, q=exp(-epsilon), Omega=epsilon*B,
  ell = epsilon*N, Q_Delta=exp(-Delta*ell)=z^N, z=q**Delta.
The normalized thermally prepared state is
  |psi_beta(t)> = Z_beta^{-1/2} exp(-i K t) exp(-beta K/2)|0>.
No real-time renormalization is applied.
"""
from __future__ import annotations
import argparse, csv, json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse.linalg import expm_multiply
try:
    from .geodesic import edge_hamiltonian, prepare
except ImportError:
    from geodesic import edge_hamiltonian, prepare


def evolved_state(K, beta: float, time: float, omega: float=1.0):
    seed,_=prepare(K,beta,chunk=1/omega)
    return expm_multiply(-1j*time*K,seed,traceA=-1j*time*float(K.diagonal().sum()))


def probe_value(psi, epsilon: float, delta: float) -> float:
    if epsilon<=0 or delta<0: raise ValueError('epsilon>0 and delta>=0 required')
    z=np.exp(-epsilon*delta); n=np.arange(len(psi))
    return float(np.abs(psi)**2 @ (z**n))


def direct_link_generators(psi,K,epsilon: float,delta: float,omega: float=1.0):
    """Direct link-current, link-energy, and coherence generating functions."""
    if epsilon<=0 or omega<=0 or delta<0: raise ValueError('positive scales and delta>=0 required')
    B=omega/epsilon; z=np.exp(-epsilon*delta)
    r=-K.diagonal(1)/B
    c=psi[:-1].conj()*psi[1:]
    weight=z**np.arange(len(c))
    current=2*B*r*c.imag
    hopping=-2*B*r*c.real
    C=np.sum(weight*r*c)
    return {'z':float(z),'coherence_generator':complex(C),
            'current_generator':float(weight@current),
            'hopping_generator':float(weight@hopping),
            'energy':float(np.vdot(psi,K@psi).real),
            'current':current,'hopping':hopping,'coherence':c,'r':r}


def derivative_reconstruction(G: float, dG_dt: float, dG_dbeta: float,
                              energy: float, epsilon: float, delta: float,
                              omega: float=1.0):
    """Exact algebraic reconstruction from B_Delta and its t,beta derivatives."""
    B=omega/epsilon; z=np.exp(-epsilon*delta)
    if delta==0: raise ValueError('finite-delta formula has removable 0/0 in the current sector')
    J=dG_dt/(z-1)
    H=2*((energy-2*B)*G-dG_dbeta)/(1+z)
    C=(-H+1j*J)/(2*B)
    return {'current_generator':float(J),'hopping_generator':float(H),
            'coherence_generator':complex(C)}


def exact_derivatives(psi,K,epsilon: float,delta: float):
    """Exact derivatives from commutator/anticommutator; used for identities."""
    z=np.exp(-epsilon*delta);n=np.arange(len(psi));qdiag=z**n
    Qpsi=qdiag*psi; Kpsi=K@psi
    energy=float(np.vdot(psi,Kpsi).real)
    G=float(np.vdot(psi,Qpsi).real)
    dtime=float((1j*(np.vdot(Kpsi,Qpsi)-np.vdot(Qpsi,Kpsi))).real)
    anti=(np.vdot(Kpsi,Qpsi)+np.vdot(Qpsi,Kpsi)).real
    dbeta=float(-.5*anti+energy*G)
    return {'G':G,'dG_dt':dtime,'dG_dbeta':dbeta,'energy':energy}


def finite_difference_derivatives(epsilon: float,beta: float,time:float,delta:float,
                                  nmax:int,omega:float=1.0,ht:float=2e-5,hb:float=2e-4):
    K=edge_hamiltonian(epsilon,omega,nmax)
    gp=probe_value(evolved_state(K,beta,time+ht,omega),epsilon,delta)
    gm=probe_value(evolved_state(K,beta,time-ht,omega),epsilon,delta)
    bp=probe_value(evolved_state(K,beta+hb,time,omega),epsilon,delta)
    bm=probe_value(evolved_state(K,beta-hb,time,omega),epsilon,delta)
    psi=evolved_state(K,beta,time,omega)
    return {'dG_dt':(gp-gm)/(2*ht),'dG_dbeta':(bp-bm)/(2*hb),
            'G':probe_value(psi,epsilon,delta),
            'energy':float(np.vdot(psi,K@psi).real)}


def coefficient_uniqueness_demo(epsilon:float=.1,nlinks:int=7):
    """Finite polynomial demo: samples at nlinks z-values recover link coefficients.

    This is not proposed as a stable inversion scheme for the full physical chain.
    It only verifies algebraic uniqueness on a finite polynomial.
    """
    rng=np.random.default_rng(753)
    coeff=rng.normal(size=nlinks)+1j*rng.normal(size=nlinks)
    z=np.linspace(.15,.85,nlinks)
    vand=np.vander(z,N=nlinks,increasing=True)
    values=vand@coeff
    recovered=np.linalg.solve(vand,values)
    return {'max_error':float(np.max(abs(recovered-coeff))),
            'condition_number':float(np.linalg.cond(vand))}


def run(output:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    beta=10.;time=1.;omega=1.;deltas=np.array([.1,.2,.4,.7,1.,1.5,2.,3.,5.])
    epsilons=[.05,.02,.01]
    rows=[];curves=[]
    for eps in epsilons:
        nmax=int(np.ceil(18/eps));K=edge_hamiltonian(eps,omega,nmax)
        psi=evolved_state(K,beta,time,omega)
        direct=[];recon=[]
        maxid=0.
        for delta in deltas:
            d=direct_link_generators(psi,K,eps,float(delta),omega)
            x=exact_derivatives(psi,K,eps,float(delta))
            r=derivative_reconstruction(**x,epsilon=eps,delta=float(delta),omega=omega)
            maxid=max(maxid,abs(d['current_generator']-r['current_generator']),
                      abs(d['hopping_generator']-r['hopping_generator']),
                      abs(d['coherence_generator']-r['coherence_generator']))
            direct.append(d);recon.append(r)
        delta_fd=1.
        fd=finite_difference_derivatives(eps,beta,time,delta_fd,nmax,omega)
        fdr=derivative_reconstruction(**fd,epsilon=eps,delta=delta_fd,omega=omega)
        dfd=direct_link_generators(psi,K,eps,delta_fd,omega)
        row={'epsilon':eps,'beta_omega':beta*omega,'time_omega':time*omega,'nmax':nmax,
             'identity_max_abs_error':float(maxid),
             'finite_difference_current_error':float(abs(fdr['current_generator']-dfd['current_generator'])),
             'finite_difference_hopping_error':float(abs(fdr['hopping_generator']-dfd['hopping_generator'])),
             'finite_difference_coherence_error':float(abs(fdr['coherence_generator']-dfd['coherence_generator'])),
             'energy':dfd['energy'],
             'length_velocity':float(eps*np.sum(direct_link_generators(psi,K,eps,0.,omega)['current'])),
             'current_generator_delta1':dfd['current_generator'],
             'hopping_generator_delta1':dfd['hopping_generator']}
        small=1e-5; smalld=direct_link_generators(psi,K,eps,small,omega)
        smallx=exact_derivatives(psi,K,eps,small)
        smallr=derivative_reconstruction(**smallx,epsilon=eps,delta=small,omega=omega)
        row['small_delta_velocity_error']=float(abs(eps*smallr['current_generator']-row['length_velocity']))
        row['small_delta_energy_error']=float(abs(smallr['hopping_generator']-(dfd['energy']-2*omega/eps)))
        rows.append(row);curves.append((eps,direct,recon))
        print(json.dumps(row),flush=True)
    inversion={str(n):coefficient_uniqueness_demo(nlinks=n) for n in (7,10,12)}
    report={'source':'Heller-Papalini-Schuhmann arXiv:2412.17785v2 bilocal exp(-Delta L); chain algebra in this repository',
            'interpretation':'exact q-oscillator identity; established bilocal gravity dictionary for neutral DSSYK only',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'runs':rows,'finite_polynomial_inversion_demo':inversion}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (output/'scan.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[r'\begin{tabular}{rrrr}\toprule',r'$\epsilon$ & 恒等式の最大誤差 & 有限差分: 流れ & 有限差分: hopping\\\midrule']
    for r in rows:
        lines.append(f"{r['epsilon']:.3f} & {r['identity_max_abs_error']:.1e} & {r['finite_difference_current_error']:.1e} & {r['finite_difference_hopping_error']:.1e} "+chr(92)*2)
    lines.append(r'\bottomrule\end{tabular}')
    (output/'table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    eps,direct,recon=curves[1]
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    ax.plot(deltas,[d['current_generator'] for d in direct],'o-',label='Direct current generating function')
    ax.plot(deltas,[r['current_generator'] for r in recon],'x--',label='From time derivative of bilocal')
    ax.set(xlabel=r'Probe weight $\Delta$',ylabel='Current generating function J_Delta');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'current_generator.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    ax.plot(deltas,[d['hopping_generator'] for d in direct],'o-',label='Direct hopping-energy generating function')
    ax.plot(deltas,[r['hopping_generator'] for r in recon],'x--',label='From temperature derivative of bilocal')
    ax.set(xlabel=r'Probe weight $\Delta$',ylabel='Hopping-energy generating function H_Delta');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'hopping_generator.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    vals=np.array([d['coherence_generator'] for d in direct])
    ax.plot(vals.real,vals.imag,'o-')
    for delta,z in zip(deltas,vals):
        ax.annotate(f'{delta:g}',(z.real,z.imag),fontsize=7)
    ax.set(xlabel='Re C_Delta',ylabel='Im C_Delta')
    fig.tight_layout();fig.savefig(output/'coherence_curve.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/boundary_probe'))
    run(p.parse_args().output)
