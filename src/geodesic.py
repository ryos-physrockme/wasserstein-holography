"""Thermally prepared chord states and independently computed AdS2 lengths.

Conventions: q=exp(-epsilon), b_n=B*sqrt(1-q**n), Omega=epsilon*B.
The chord basis is the Krylov basis of |0>, NOT that of the thermal seed.
Only imaginary-time preparation is normalized; real-time probabilities are not.
The classical comparison is a leading small-epsilon calculation, with no fit.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


def logcosh(x):
    x = np.asarray(x, dtype=float)
    return np.logaddexp(x, -x) - np.log(2.0)


def thermal_saddle(beta_omega: float) -> float:
    """u in (0,pi/2], determined by beta*Omega*sin(u)=pi-2*u."""
    if not np.isfinite(beta_omega) or beta_omega < 0:
        raise ValueError("beta_omega must be finite and nonnegative")
    if beta_omega == 0:
        return np.pi / 2
    return float(brentq(lambda u: beta_omega*np.sin(u)+2*u-np.pi,
                        0.0, np.pi/2, xtol=1e-14))


def classical_length(times, beta: float, omega: float = 1.0):
    """Absolute dimensionless saddle length, including preparation cost."""
    if not np.isfinite(omega) or omega <= 0 or beta < 0:
        raise ValueError("require omega > 0 and beta >= 0")
    u = thermal_saddle(beta*omega)
    return -2*np.log(np.sin(u)) + 2*logcosh(omega*np.sin(u)*np.asarray(times))


def geodesic_growth(total_time, temperature: float, cutoff_ratio=None):
    """(L(t_L+t_R)-L(0))/L_2 for an AdS2 black hole.

    For finite cutoff A=r_c/r_h, compute the embedding-space invariant.
    This function has no dependence on the chord Hamiltonian or its evolution.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    tau = 2*np.pi*temperature*np.asarray(total_time, dtype=float)
    if cutoff_ratio is None:
        return 2*logcosh(tau/2)
    A = float(cutoff_ratio)
    if not np.isfinite(A) or A <= 1:
        raise ValueError("cutoff_ratio must exceed one")
    invariant = A*A + (A*A-1)*np.cosh(tau)
    return np.arccosh(invariant) - np.arccosh(2*A*A-1)


def edge_hamiltonian(epsilon: float, omega: float, nmax: int):
    """K=2B*I-H_K, related to H_K by a staggered basis change and shift."""
    if not (np.isfinite(epsilon) and epsilon > 0 and np.isfinite(omega) and omega > 0):
        raise ValueError("epsilon and omega must be finite and positive")
    if not isinstance(nmax, (int, np.integer)) or nmax < 2:
        raise ValueError("nmax must be an integer >= 2")
    B = omega/epsilon
    n = np.arange(1, nmax+1)
    b = B*np.sqrt(-np.expm1(-epsilon*n))
    return diags([-b, np.full(nmax+1, 2*B), -b], [-1, 0, 1], format="csr")


def prepare(K, beta: float, chunk: float = 1.0):
    """Normalized exp(-beta*K/2)|0>; record the shifted partition function.

    Intermediate normalization prevents underflow. chunk is an imaginary-time
    interval, not an integrator time step; every segment is a matrix exponential.
    """
    if beta < 0 or not np.isfinite(beta) or chunk <= 0:
        raise ValueError("invalid imaginary-time parameters")
    seed = np.zeros(K.shape[0])
    seed[0] = 1.0
    count = max(1, int(np.ceil(beta/(2*chunk))))
    h = beta/(2*count)
    trace = float(K.diagonal().sum())
    lognorm = 0.0
    for _ in range(count):
        seed = expm_multiply(-h*K, seed, traceA=-h*trace)
        norm = np.linalg.norm(seed)
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("thermal preparation lost its norm")
        seed /= norm
        lognorm += np.log(norm)
    return seed, 2*lognorm


def thermal_evolution(epsilon: float, beta: float, times, nmax: int,
                      omega: float = 1.0):
    """Finite-chain evolution; returns probabilities and reproducibility checks."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or times[0] != 0 or np.any(np.diff(times) <= 0)
            or not np.allclose(np.diff(times), times[1]-times[0])):
        raise ValueError("times must start at zero, increase, and be uniform")
    K = edge_hamiltonian(epsilon, omega, nmax)
    seed, logz = prepare(K, beta, chunk=1/omega)
    psi = expm_multiply(-1j*K, seed, start=0, stop=times[-1], num=len(times),
                        endpoint=True, traceA=-1j*float(K.diagonal().sum()))
    P = np.abs(psi)**2
    n = np.arange(nmax+1)
    mean = P@n
    ell = epsilon*mean
    velocity = -1j*(K@psi.T).T
    energies = np.real(np.sum(psi.conj()*(K@psi.T).T, axis=1))
    # Exact double-commutator identity, with its finite-chain endpoint correction.
    acceleration = -(K@(K@psi.T)).T
    ell_ddot = 2*epsilon*np.real((abs(velocity)**2 + psi.conj()*acceleration)@n)
    B = omega/epsilon
    rhs = 2*epsilon*B**2*(1-np.exp(-epsilon))*(P@np.exp(-epsilon*n))
    endpoint = 2*epsilon*B**2*(1-np.exp(-epsilon*(nmax+1)))*P[:, -1]
    tail = np.cumsum(P[:, ::-1], axis=1)[:, ::-1][:, 1:]
    delta_tail = tail-tail[0]
    w1_thermal = np.abs(delta_tail).sum(axis=1)
    mean_difference = mean-mean[0]
    # Exact discrete transport identity when the mean moves outward.
    excess_rhs = 2*np.maximum(-delta_tail, 0).sum(axis=1)
    summary = {
        "epsilon": epsilon, "q": float(np.exp(-epsilon)), "beta": beta,
        "omega": omega, "nmax": nmax, "log_Z_shifted": float(logz),
        "ell_initial": float(ell[0]), "growth_final": float(ell[-1]-ell[0]),
        "norm_error": float(np.max(abs(P.sum(axis=1)-1))),
        "tail_probability": float(np.max(P[:, -20:].sum(axis=1))),
        "energy_drift": float(np.max(abs(energies-energies[0]))),
        "initial_energy_above_edge": float(energies[0]),
        "acceleration_identity_error": float(np.max(abs(ell_ddot-rhs+endpoint))),
        "transport_identity_error": float(np.max(abs(w1_thermal-mean_difference-excess_rhs))),
        "thermal_W1_excess_scaled": float(epsilon*np.max(excess_rhs)),
    }
    return {"probability": P, "length": ell, "w1_thermal": epsilon*w1_thermal,
            "summary": summary, "initial_seed": seed}


def sector_parameters(lam: float, charge: float, p: int, energy: float = 1.0):
    """FKN v2 Eqs. (3.33),(3.34), same normalization as src/krylov.py."""
    if (lam <= 0 or energy <= 0 or abs(charge) >= .5
            or not isinstance(p, int) or p < 1):
        raise ValueError("invalid fixed-charge parameters")
    s = 1-4*charge**2
    eps = 4*lam/s
    a = energy*s**(p/2)/np.sqrt(lam)*np.exp(-lam/2*(4*charge*(charge+1)-1)/s)
    return {"epsilon": eps, "a": float(a),
            "omega": float(eps*a/np.sqrt(-np.expm1(-eps)))}


def run(output: Path):
    """Reproduce all tables and figures, including doubled-chain comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    unit_times = np.linspace(0, 1, 81)
    cases = [(.05, 10), (.02, 10), (.01, 10), (.005, 10),
             (.02, 20), (.01, 20), (.005, 20), (.005, 40)]
    rows, curves = [], []
    for eps, beta in cases:
        nmax = int(np.ceil((2*np.log(1+beta)+2*np.pi+10)/eps))
        times = unit_times*beta
        result = thermal_evolution(eps, beta, times, nmax)
        # Independent spatial cutoff, same entire time interval, only 3 outputs.
        coarse_times = np.linspace(0, beta, 3)
        doubled = thermal_evolution(eps, beta, coarse_times, 2*nmax)
        u = thermal_saddle(beta)
        v = 1-2*u/np.pi
        exact = result["length"]-result["length"][0]
        classical = classical_length(times, beta)
        classical -= classical[0]
        jt = geodesic_growth(times, 1/beta)
        row = dict(result["summary"])
        row.update({"beta_omega": beta, "v": float(v),
                    "classical_growth_final": float(classical[-1]),
                    "jt_growth_final": float(jt[-1]),
                    "relative_error_classical_final": float(exact[-1]/classical[-1]-1),
                    "relative_error_jt_final": float(exact[-1]/jt[-1]-1),
                    "max_absolute_error_classical": float(np.max(abs(exact-classical))),
                    "doubled_cutoff_length_error": float(np.max(abs(
                        result["length"][[0,40,80]]-doubled["length"]))),
                    "doubled_cutoff_probability_L1": float(np.max(np.sum(abs(
                        result["probability"][[0,40,80]]-doubled["probability"][:, :nmax+1]),axis=1))),
                   })
        rows.append(row)
        curves.append((eps, beta, exact, classical, jt))
        print(json.dumps(row), flush=True)
    fields = ["epsilon", "beta_omega", "v", "growth_final", "classical_growth_final",
              "jt_growth_final", "relative_error_classical_final", "relative_error_jt_final",
              "doubled_cutoff_length_error", "norm_error", "tail_probability"]
    with (output/"thermal_scan.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    np.savetxt(output/"curves.csv", np.column_stack([unit_times]+[c[2] for c in curves]),
               delimiter=",", header="t_over_beta,"+",".join(f"eps_{c[0]}_s_{c[1]}" for c in curves),comments="")
    # Fixed physical temperature, not a rescaled temperature separately fitted per charge.
    sector_rows=[]
    for charge in (0., .2, .4):
        pars=sector_parameters(.002, charge, 10)
        beta_physical=5.0
        s=beta_physical*pars["omega"]
        eps=pars["epsilon"]
        nmax=int(np.ceil((2*np.log(1+s)+2*np.pi+10)/eps))
        result=thermal_evolution(eps, s, unit_times*s, nmax) # Omega=1, rescaled time
        u=thermal_saddle(s)
        v=1-2*u/np.pi
        sector_rows.append({"lambda":.002,"charge":charge,"p":10,"J":1.0,
                            **pars,"beta_physical":beta_physical,"beta_omega":s,
                            "v":float(v), "growth_at_t_equals_beta":result["summary"]["growth_final"],
                            "classical_growth":float(2*logcosh(np.pi*v)),
                            "q":float(np.exp(-eps)),"nmax":nmax,
                            "norm_error":result["summary"]["norm_error"],
                            "tail_probability":result["summary"]["tail_probability"]})
    # Direct embedding-space geodesic calculation at a finite radial cutoff.
    cutoffs=[10.,100.,1000.]
    cutoff_tests=[{"cutoff_ratio":A,"max_difference":float(np.max(abs(
        geodesic_growth(unit_times,1.,A)-geodesic_growth(unit_times,1.))))} for A in cutoffs]
    report={"source":"FKN 2512.07715v2 Eqs.3.33,3.34; Heller et al.2412.17785v2 Eqs.3,4,6,8,13",
            "conventions":"q=exp(-epsilon); Omega=epsilon B; t=t_L+t_R; no real-time normalization; no fit",
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "runs":rows,"fixed_charge_runs":sector_rows,"geodesic_cutoff_tests":cutoff_tests}
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=[r"\begin{tabular}{rrrrrr}\toprule",
           r"$\epsilon$ & $\beta\Omega$ & $v$ & $\delta\ell(\beta)$ & 古典式の偏差 [\%] & JT式の偏差 [\%]\\\midrule"]
    for r in rows:
        lines.append(f"{r['epsilon']:.3f} & {r['beta_omega']:.0f} & {r['v']:.4f} & {r['growth_final']:.6f} & {100*r['relative_error_classical_final']:.3f} & {100*r['relative_error_jt_final']:.3f} "+chr(92)*2)
    lines.append(r"\bottomrule\end{tabular}")
    (output/"table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=[r"\begin{tabular}{rrrrrr}\toprule",r"$\mathcal Q$ & $\epsilon$ & $\Omega/\mathcal J$ & $\beta\Omega$ & $v$ & $\delta\ell(\beta)$\\\midrule"]
    for r in sector_rows:
        lines.append(f"{r['charge']:.1f} & {r['epsilon']:.6f} & {r['omega']:.6f} & {r['beta_omega']:.4f} & {r['v']:.4f} & {r['growth_at_t_equals_beta']:.6f} "+chr(92)*2)
    lines.append(r"\bottomrule\end{tabular}")
    (output/"charge_table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    for eps,beta,exact,cl,jt in curves:
        if beta==10:
            ax.plot(unit_times,exact,label=fr"$\epsilon={eps}$")
    ax.plot(unit_times,curves[0][3],"--",label="Thermal saddle (no fit)")
    ax.plot(unit_times,curves[0][4],":",label=r"JT at $T=1/\beta$")
    ax.set(xlabel=r"$t/\beta$, $t=t_L+t_R$",ylabel=r"$\epsilon[\langle n\rangle_t-\langle n\rangle_0]$",title=r"$\beta\Omega=10$")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/"thermal_length.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.0))
    for eps,beta,exact,cl,jt in curves:
        if eps==.005:
            ax.plot(unit_times,exact,label=fr"$\beta\Omega={beta}$, quantum")
            ax.plot(unit_times,cl,"--",label=fr"$\beta\Omega={beta}$, saddle")
    ax.plot(unit_times,curves[0][4],":",label="JT limit")
    ax.set(xlabel=r"$t/\beta$",ylabel=r"$\delta\ell(t)$",title=r"$\epsilon=0.005$")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/"low_temperature_limit.pdf");plt.close(fig)
    print(json.dumps({"fixed_charge_runs":sector_rows,"geodesic_cutoff_tests":cutoff_tests}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/geodesic"))
    run(parser.parse_args().output)
