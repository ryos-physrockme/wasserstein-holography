"""Fixed-charge q-oscillator Krylov dynamics and reproducible checks.

Input: Foerste, Kruse, Natu, arXiv:2512.07715v2, Eqs. (3.33),(3.34).
We solve d chi_n/dt = b_n chi_{n-1} - b_{n+1} chi_{n+1}, chi(0)=e_0.
The physical amplitudes are (-i)**n * chi_n. No probabilities are renormalized.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


def physical_parameters(lam: float, charge: float, p: int, energy: float = 1.0) -> dict:
    """Return q and a=b_1; charge is the normalized sector label in (-1/2,1/2)."""
    if not (np.isfinite(lam) and lam > 0 and np.isfinite(energy) and energy > 0):
        raise ValueError("lam and energy must be finite and positive")
    if not (np.isfinite(charge) and abs(charge) < 0.5):
        raise ValueError("charge must lie strictly between -1/2 and 1/2")
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError("p must be a positive integer")
    s = 1 - 4 * charge**2
    eps = 4 * lam / s
    loga = np.log(energy) + 0.5*p*np.log(s) - 0.5*np.log(lam)
    loga -= 0.5*lam*(4*charge*(charge+1)-1)/s
    return {"q": float(np.exp(-eps)), "epsilon": eps, "a": float(np.exp(loga))}


def links(q: float, a: float, nmax: int) -> np.ndarray:
    """b_1,...,b_nmax; the chain contains n=0,...,nmax."""
    if not (np.isfinite(q) and 0 < q < 1 and np.isfinite(a) and a > 0):
        raise ValueError("require 0 < q < 1 and finite a > 0")
    if not isinstance(nmax, (int, np.integer)) or nmax < 2:
        raise ValueError("nmax must be an integer >= 2")
    n = np.arange(1, nmax+1)
    return a*np.sqrt(-np.expm1(n*np.log(q))/(1-q))


def evolve(q: float, a: float, times: np.ndarray, nmax: int) -> tuple:
    """Matrix exponential on a uniform time grid starting at zero."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array")
    dt = np.diff(times)
    if times[0] != 0 or not np.all(dt > 0) or not np.allclose(dt, dt[0]):
        raise ValueError("times must be increasing, uniform, and start at zero")
    b = links(q, a, nmax)
    generator = diags([b, -b], [-1, 1], shape=(nmax+1, nmax+1), format="csr")
    initial = np.zeros(nmax+1)
    initial[0] = 1.0
    chi = expm_multiply(generator, initial, start=0, stop=times[-1],
                        num=len(times), endpoint=True, traceA=0.0)
    return chi, generator


def logcosh(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, -x)-np.log(2.0)


def matched_curve(q: float, a: float, times: np.ndarray) -> np.ndarray:
    """Matches exact t^2 and t^4 coefficients, not the full evolution."""
    return 2/(1-q)*logcosh(a*np.sqrt(1-q)*np.asarray(times))


def lower_bound(q: float, a: float, times: np.ndarray) -> np.ndarray:
    """Jensen bound for the infinite chain: C >= this curve for t >= 0."""
    eps = -np.log(q)
    return 2/eps*logcosh(a*np.sqrt(eps)*np.asarray(times))


def observables(q: float, a: float, chi: np.ndarray, generator) -> dict:
    probability = chi**2
    n = np.arange(chi.shape[1])
    velocity = (generator @ chi.T).T
    acceleration = (generator @ velocity.T).T
    mean = probability @ n
    cddot = 2*((velocity**2 + chi*acceleration) @ n)
    identity_rhs = 2*a*a*(probability @ (q**n))
    # Finite-chain correction to [A,A^dagger]=q^N at the last site.
    boundary = 2*a*a*(-np.expm1((n[-1]+1)*np.log(q)))/(1-q)*probability[:, -1]
    fisher = 4*np.sum(velocity**2, axis=1)  # continuous extension through zeros
    w1 = np.sum(np.cumsum(probability[:, :0:-1], axis=1), axis=1)
    return {"mean": mean, "norm_error": np.max(np.abs(probability.sum(axis=1)-1)),
            "tail_probability": np.max(probability[:, -20:].sum(axis=1)),
            "w1_error": np.max(np.abs(w1-mean)),
            "fisher_relative_error": np.max(np.abs(fisher/(4*a*a)-1)),
            "acceleration_relative_error": np.max(np.abs(cddot-identity_rhs+boundary))/(2*a*a)}


def pairwise_example() -> dict:
    """Mean chord number does not determine distances between arbitrary times."""
    times = np.linspace(0, 2.5, 101)
    chi, _ = evolve(0.2, 1.0, times, 180)
    p, r = chi[80]**2, chi[100]**2
    n = np.arange(len(p))
    w1 = float(np.abs(np.cumsum(p-r)[:-1]).sum())
    difference = float((r-p)@n)
    return {"q": 0.2, "a": 1.0, "t1": 2.0, "t2": 2.5,
            "W1": w1, "mean_difference": difference,
            "gap": w1-difference, "P0_t1": float(p[0]), "P0_t2": float(r[0])}


def run(output: Path) -> None:
    """Generate figures, CSV, JSON, and the numerical table used by the LaTeX note."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 4, 161)  # x=a*sqrt(1-q)*t
    qvalues = [0.2, 0.6, 0.9, 0.98, float(np.exp(-0.004))]
    records, columns, curves = [], [x], []
    for q in qvalues:
        a = 1.0
        times = x/(a*np.sqrt(1-q))
        front = 2*x[-1]/(1-q)
        nmax = int(np.ceil(front+12*np.sqrt(front+1)+64))
        chi, generator = evolve(q, a, times, nmax)
        obs = observables(q, a, chi, generator)
        chi2, _ = evolve(q, a, times, 2*nmax)
        mean2 = chi2**2 @ np.arange(2*nmax+1)
        mean = obs.pop("mean")
        normalized = (1-q)*mean/2
        candidate = matched_curve(q, a, times)
        bound = lower_bound(q, a, times)
        relative = np.abs(mean[1:]-candidate[1:])/candidate[1:]
        row = {"q": q, "nmax": nmax,
               "max_relative_deviation_matched": float(np.max(relative)),
               "normalized_C_x1": float(normalized[40]),
               "normalized_C_x4": float(normalized[-1]),
               "cutoff_max_abs_difference": float(np.max(np.abs(mean-mean2))),
               "min_gap_above_lower_bound": float(np.min(mean-bound)),
               **{k: float(v) for k, v in obs.items()}}
        records.append(row)
        columns.append(normalized)
        curves.append((q, normalized))
    np.savetxt(output/"curves.csv", np.column_stack(columns), delimiter=",",
               header="x,"+",".join(f"scaled_C_q_{q:.8f}" for q in qvalues), comments="")
    summary = {"source": "arXiv:2512.07715v2, Eqs. (3.33),(3.34)",
               "scaling": "x=a*sqrt(1-q)*t; scaled_C=(1-q)*C/2; a=b_1",
               "pairwise_transport": pairwise_example(),
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__}, "runs": records}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    for q, curve in curves:
        ax.plot(x, curve, label=fr"$q={q:.4f}$")
    ax.plot(x, logcosh(x), "--", label=r"$\log\cosh x$")
    ax.set(xlabel=r"$x=a\sqrt{1-q}\,t$", ylabel=r"$(1-q)C(t)/2$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"complexity.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    for q, curve in curves:
        ax.plot(x[1:], (curve[1:]/logcosh(x[1:])-1), label=fr"$q={q:.4f}$")
    ax.set(xlabel=r"$x=a\sqrt{1-q}\,t$", ylabel=r"$C/C_{\rm match}-1$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"relative_error.pdf")
    plt.close(fig)
    rows = []
    for row in records:
        rows.append(f"{row['q']:.6f} & {row['nmax']} & "
                    f"{100*row['max_relative_deviation_matched']:.3f} & "
                    f"{row['cutoff_max_abs_difference']:.1e} " + chr(92)*2)
    table = (r"\begin{tabular}{rrrr}\toprule"+"\n"
             +r"$q$ & $M$ & 最大相対偏差 [\%] & $M$と$2M$の最大差\\\midrule"+"\n"
             +"\n".join(rows)+"\n"+r"\bottomrule\end{tabular}"+"\n")
    (output/"table.tex").write_text(table, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    run(parser.parse_args().output)
