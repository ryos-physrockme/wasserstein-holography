"""Length current, neighboring coherences, and local canonical momentum.

Input convention: K=2B I - sum b_{n+1}(|n><n+1|+h.c.),
b_n=B sqrt(1-exp(-epsilon*n)), Omega=epsilon*B, ell=epsilon*n.
All quantum identities are evaluated in the orthonormal chord basis.
The semiclassical momentum is a local phase variable, not an assumed global
self-adjoint phase operator on the half-line. No real-time renormalization.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
try:
    from .geodesic import edge_hamiltonian, prepare, thermal_saddle, classical_length
except ImportError:
    from geodesic import edge_hamiltonian, prepare, thermal_saddle, classical_length


def lowering(epsilon: float, nmax: int):
    """A|n>=sqrt(1-q**n)|n-1>; truncated at nmax."""
    if not np.isfinite(epsilon) or epsilon <= 0 or nmax < 2:
        raise ValueError('epsilon>0 and nmax>=2 required')
    r=np.sqrt(-np.expm1(-epsilon*np.arange(1,nmax+1)))
    return diags(r,1,shape=(nmax+1,nmax+1),format='csr')


def coherence_data(psi, K, epsilon: float, omega: float=1.) -> dict:
    """Return probabilities, adjacent complex products, currents and checks."""
    psi=np.asarray(psi,complex)
    if psi.ndim!=1 or not np.all(np.isfinite(psi)) or psi.size!=K.shape[0]:
        raise ValueError('psi must be a finite vector compatible with K')
    if epsilon<=0 or omega<=0: raise ValueError('epsilon and omega must be positive')
    B=omega/epsilon; n=np.arange(len(psi)); r=-K.diagonal(1)/B
    probability=abs(psi)**2
    links=psi[:-1].conj()*psi[1:]
    current=2*B*r*links.imag
    real_hopping=-2*B*r*links.real
    z=np.sum(r*links)
    velocity=-1j*(K@psi)
    pdot=2*np.real(psi.conj()*velocity)
    flux_divergence=np.r_[0.,current]-np.r_[current,0.]
    ell=epsilon*n
    mean=float(probability@ell)
    energy=float(np.vdot(psi,K@psi).real)
    v=float(pdot@ell)
    A=lowering(epsilon,len(psi)-1)
    X=(A+A.T)*.5;Y=(A-A.T)/(2j)
    xp=X@psi;yp=Y@psi
    norm=float(probability.sum())
    variance_sum=float(np.vdot(xp,xp).real+np.vdot(yp,yp).real-abs(z)**2)
    # The exact expectation identity below assumes normalized state; the
    # explicit norm retains the numerical normalization residual.
    rhs=norm-.5*(1+np.exp(-epsilon))*(probability@np.exp(-ell))
    rhs-=.5*(-np.expm1(-epsilon*len(psi)))*probability[-1]
    return {'probability':probability,'coherence':links,'current':current,
            'hopping_energy':real_hopping,'z':z,'mean':mean,'energy':energy,
            'velocity':v,'phase':float(np.angle(z)),
            'variance_X_plus_Y':variance_sum,
            'norm_error':abs(norm-1.),'tail_probability':float(probability[-20:].sum()),
            'continuity_error':float(np.max(abs(pdot-flux_divergence))),
            'velocity_error':float(abs(v-2*omega*z.imag)),
            'energy_error':float(abs(energy-2*B*(norm-z.real))),
            'quadrature_identity_error':float(abs(abs(z)**2+variance_sum-rhs))}


def reconstruct(probability, coherence, threshold: float=1e-14,
                positive_cosine: bool=False) -> tuple[np.ndarray,dict]:
    """Reconstruct a pure state on one connected interval of nonzero weight.

    positive_cosine uses only Im(coherence) and additionally assumes every
    phase difference lies in [-pi/2,pi/2]. The general reconstruction uses both
    real and imaginary parts. Returned state is normalized on the retained
    interval; retained mass is explicitly reported. Disconnected support is
    rejected rather than assigned arbitrary unobservable relative phases.
    """
    p=np.asarray(probability,float);c=np.asarray(coherence,complex)
    if p.ndim!=1 or c.shape!=(len(p)-1,) or np.any(p<0) or threshold<0:
        raise ValueError('invalid probability, coherence, or threshold')
    active=np.flatnonzero(p>threshold)
    if not len(active):raise ValueError('no retained probability')
    lo,hi=int(active[0]),int(active[-1])
    if np.any(p[lo:hi+1]<=threshold):raise ValueError('retained support is disconnected')
    local=c[lo:hi]
    if np.any(abs(local)==0):raise ValueError('missing coherence on retained interval')
    if positive_cosine:
        angles=np.arcsin(np.clip(local.imag/np.sqrt(p[lo:hi]*p[lo+1:hi+1]),-1.,1.))
    else:angles=np.angle(local)
    phase=np.r_[0.,np.cumsum(angles)]
    candidate=np.zeros(len(p),complex)
    retained=float(p[lo:hi+1].sum())
    candidate[lo:hi+1]=np.sqrt(p[lo:hi+1]/retained)*np.exp(1j*phase)
    return candidate,{'first_index':lo,'last_index':hi,'retained_mass':retained,
                      'min_retained_cosine':float(np.min(np.cos(np.angle(local)))) if len(local) else 1.}


def fidelity(first, second) -> float:
    return float(np.clip(abs(np.vdot(first,second))**2/
                         (np.vdot(first,first).real*np.vdot(second,second).real),0.,1.))


def classical_momentum(times,beta: float,omega: float=1.):
    """Local phase of the outward thermal orbit, from h=2Omega(1-r cos p)."""
    u=thermal_saddle(beta*omega)
    return np.arctan2(np.sin(u)*np.tanh(omega*np.sin(u)*np.asarray(times)),np.cos(u))


def thermal_run(epsilon: float,beta: float=10.,omega: float=1.,
                nmax: int|None=None,num: int=81) -> dict:
    if nmax is None:nmax=int(np.ceil(18/epsilon))
    K=edge_hamiltonian(epsilon,omega,nmax)
    seed,_=prepare(K,beta,chunk=1/omega)
    times=np.linspace(0,beta,num)
    psi=expm_multiply(-1j*K,seed,start=0,stop=beta,num=num,endpoint=True,
                      traceA=-1j*float(K.diagonal().sum()))
    data=[coherence_data(s,K,epsilon,omega) for s in psi]
    phase=np.array([d['phase'] for d in data]);mean=np.array([d['mean'] for d in data])
    phase_cl=classical_momentum(times,beta,omega)
    mean_cl=classical_length(times,beta,omega)
    v_cl=2*omega*np.sin(thermal_saddle(beta*omega))*np.tanh(omega*np.sin(thermal_saddle(beta*omega))*times)
    chosen=int(np.argmin(abs(times-1/omega)))
    reconstructed,details=reconstruct(data[chosen]['probability'],data[chosen]['coherence'])
    sine_reconstructed,_=reconstruct(data[chosen]['probability'],data[chosen]['coherence'],positive_cosine=True)
    norm=float(data[chosen]['probability'].sum())
    kept=details['retained_mass']/norm
    row={'epsilon':epsilon,'beta_omega':beta*omega,'nmax':nmax,'time_sample':float(times[chosen]),
         'mean_sample':data[chosen]['mean'],'velocity_sample':data[chosen]['velocity'],
         'phase_sample':data[chosen]['phase'],'classical_phase_sample':float(phase_cl[chosen]),
         'phase_max_abs_error':float(np.max(abs(phase-phase_cl))),
         'phase_final':float(phase[-1]),'classical_phase_final':float(phase_cl[-1]),
         'mean_max_abs_error':float(np.max(abs(mean-mean_cl))),
         'velocity_max_abs_error':float(np.max(abs(np.array([d['velocity'] for d in data])-v_cl))),
         'coherence_phase_negative_time':float(-data[chosen]['phase']),
         'max_quadrature_variance':max(d['variance_X_plus_Y'] for d in data),
         'reconstruction_full_fidelity':fidelity(psi[chosen],reconstructed),
         'reconstruction_error_on_retained_interval':abs(1-fidelity(psi[chosen],reconstructed)/kept),
         'positive_cosine_full_fidelity':fidelity(psi[chosen],sine_reconstructed),
         'omitted_probability':float(1-kept),'min_retained_cosine':details['min_retained_cosine'],
         **{key:max(d[key] for d in data) for key in ['norm_error','tail_probability','continuity_error',
                'velocity_error','energy_error','quadrature_identity_error']}}
    return {'row':row,'times':times,'phase':phase,'classical_phase':phase_cl,'length':mean,
            'classical_length':mean_cl,'velocity':np.array([d['velocity'] for d in data]),
            'sample_coherence':data[chosen]['coherence'],'sample_probability':data[chosen]['probability']}


def ambiguous_pair(epsilon: float=.2,omega: float=1.,nmax: int=90,theta: float=np.pi/6) -> dict:
    """Two pure states with the same P_n, J_n and total energy at all times.

    phi = D psi*, D_nn=(-1)**n. The real hopping matrix anticommutes with D.
    Select weights so the initial hopping expectation is exactly zero, giving
    both states <K>=2B. Initial fidelity = sin(theta)**2.
    """
    K=edge_hamiltonian(epsilon,omega,nmax); b=-K.diagonal(1)
    p1=.5;p0=.5*b[1]**2/(b[0]**2+b[1]**2);p2=.5-p0
    first=np.zeros(nmax+1,complex)
    first[:3]=np.sqrt([p0,p1,p2])*np.exp(1j*np.array([0.,theta,np.pi]))
    parity=(-1.)**np.arange(nmax+1);second=parity*first.conj()
    times=np.linspace(0,2,41)
    states=[expm_multiply(-1j*K,s,start=0,stop=2,num=41,endpoint=True,
                         traceA=-1j*float(K.diagonal().sum())) for s in (first,second)]
    a=[coherence_data(s,K,epsilon,omega) for s in states[0]]
    bdata=[coherence_data(s,K,epsilon,omega) for s in states[1]]
    reconstructed,_=reconstruct(a[0]['probability'],a[0]['coherence'])
    return {'epsilon':epsilon,'omega':omega,'theta':theta,'nmax':nmax,'final_time':2.,
            'probabilities_initial':[p0,p1,p2],'fidelity_initial':fidelity(first,second),
            'energy_first':a[0]['energy'],'energy_second':bdata[0]['energy'],
            'initial_currents':a[0]['current'][:2].tolist(),
            'initial_real_coherences_first':a[0]['coherence'][:2].real.tolist(),
            'initial_real_coherences_second':bdata[0]['coherence'][:2].real.tolist(),
            'probability_max_difference':float(max(np.max(abs(x['probability']-y['probability'])) for x,y in zip(a,bdata))),
            'current_max_difference':float(max(np.max(abs(x['current']-y['current'])) for x,y in zip(a,bdata))),
            'energy_max_difference':float(max(abs(x['energy']-y['energy']) for x,y in zip(a,bdata))),
            'fidelity_max_drift':float(max(abs(fidelity(x,y)-fidelity(first,second)) for x,y in zip(*states))),
            'full_coherence_reconstruction_fidelity':fidelity(first,reconstructed)}


def mixed_ambiguity(eta: float=.15) -> dict:
    if not 0<eta<1/3:raise ValueError('0<eta<1/3 required')
    rho=np.eye(3,dtype=complex)/3;sigma=rho.copy()
    rho[0,2]=rho[2,0]=eta;sigma[0,2]=sigma[2,0]=-eta
    return {'eta':eta,'min_eigenvalue':float(np.linalg.eigvalsh(rho).min()),
            'trace_distance':float(.5*sum(abs(np.linalg.eigvalsh(rho-sigma)))),
            'diagonal_difference':float(np.max(abs(np.diag(rho-sigma)))),
            'nearest_coherence_difference':float(np.max(abs(np.diag(rho-sigma,1))))}


def run(output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    cases=[.05,.02,.01,.005];rows=[];curves=[]
    for eps in cases:
        result=thermal_run(eps); row=result['row']
        other=thermal_run(eps,nmax=2*row['nmax'])
        row['doubled_cutoff_phase_error']=float(max(abs(result['phase']-other['phase'])))
        row['doubled_cutoff_length_error']=float(max(abs(result['length']-other['length'])))
        rows.append(row);curves.append(result)
        np.savetxt(output/f'curves_epsilon_{eps:.3f}.csv',np.column_stack([result['times'],result['length'],result['classical_length'],result['phase'],result['classical_phase'],result['velocity']]),delimiter=',',header='time,length,classical_length,phase,classical_phase,length_velocity',comments='')
        n=np.arange(len(result['sample_probability'])-1)
        np.savetxt(output/f'links_epsilon_{eps:.3f}.csv',np.column_stack([eps*(n+.5),result['sample_probability'][:-1],result['sample_coherence'].real,result['sample_coherence'].imag]),delimiter=',',header='link_midpoint,probability_left,real_coherence,imaginary_coherence',comments='')
        print(json.dumps(row),flush=True)
    report={'source':'Heller et al. arXiv:2412.17785v2, orthonormalized length recurrence; previous repo K convention',
            'meaning':'phase=arg(<A>), not an expectation of a globally defined momentum operator',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
            'thermal_runs':rows,'pure_state_ambiguity':ambiguous_pair(),'mixed_state_ambiguity':mixed_ambiguity()}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (output/'thermal_scan.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'\begin{tabular}{rrrrr}\toprule',r'$\epsilon$ & $p_{\rm coh}(1/\Omega)$ & $p_{\rm cl}(1/\Omega)$ & 最大位相偏差 & $M,2M$の位相差\\\midrule']
    for r in rows:lines.append(f"{r['epsilon']:.3f} & {r['phase_sample']:.7f} & {r['classical_phase_sample']:.7f} & {r['phase_max_abs_error']:.2e} & {r['doubled_cutoff_phase_error']:.1e} "+chr(92)*2)
    lines.append(r'\bottomrule\end{tabular}')
    (output/'table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    for r in curves:ax.plot(r['times'],r['phase'],label=fr"$\epsilon={r['row']['epsilon']}$")
    ax.plot(curves[0]['times'],curves[0]['classical_phase'],'--',label='Classical orbit (no fit)')
    ax.set(xlabel=r'$\Omega t$',ylabel=r'$p_{\rm coh}=\arg\langle A\rangle$');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'momentum.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    for r in curves:ax.plot(r['times'],(r['phase']-r['classical_phase'])/r['row']['epsilon'],label=fr"$\epsilon={r['row']['epsilon']}$")
    ax.set(xlabel=r'$\Omega t$',ylabel=r'$(p_{\rm coh}-p_{\rm cl})/\epsilon$');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'momentum_error.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    r=curves[1];ax.plot(r['length'],r['phase'],label='Forward evolution')
    ax.plot(r['length'],-r['phase'],'--',label='Reversed evolution')
    ax.set(xlabel=r'$\langle\ell\rangle$',ylabel=r'$\arg\langle A\rangle$');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'length_momentum.pdf');plt.close(fig)
    print(json.dumps({'pure_state_ambiguity':report['pure_state_ambiguity'],'mixed_state_ambiguity':report['mixed_state_ambiguity']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/phase_space'))
    run(parser.parse_args().output)
