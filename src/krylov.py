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
from scipy.special import gammaln


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


def discrete_w1_matrix(probability: np.ndarray) -> np.ndarray:
    """Pairwise W1 distances on n=0,1,... with unit ground cost |n-m|."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("probability must have shape (nt, nmax+1)")
    cdf = np.cumsum(probability, axis=1)[:, :-1]
    nt = probability.shape[0]
    distance = np.zeros((nt, nt), dtype=float)
    for i in range(nt - 1):
        values = np.sum(np.abs(cdf[i] - cdf[i + 1:]), axis=1)
        distance[i, i + 1:] = values
        distance[i + 1:, i] = values
    return distance


def probability_current(q: float, a: float, chi: np.ndarray) -> np.ndarray:
    """Current J_k=2 b_{k+1} chi_k chi_{k+1} across the link k -> k+1."""
    chi = np.asarray(chi, dtype=float)
    if chi.ndim != 2 or chi.shape[1] < 3:
        raise ValueError("chi must have shape (nt, nmax+1)")
    b = links(q, a, chi.shape[1] - 1)
    return 2.0 * chi[:, :-1] * chi[:, 1:] * b[None, :]


def classical_mds_spectrum(distance: np.ndarray) -> np.ndarray:
    """Eigenvalues of the double-centered squared-distance matrix."""
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    return np.linalg.eigvalsh(gram)[::-1]


def distance_geometry(probability: np.ndarray, tolerance: float = 1e-10) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compare all pairwise W1 distances with the line coordinate C=<n>."""
    probability = np.asarray(probability, dtype=float)
    n = np.arange(probability.shape[1])
    mean = probability @ n
    distance = discrete_w1_matrix(probability)
    line = np.abs(mean[:, None] - mean[None, :])
    gap = distance - line
    mask = np.triu(np.ones_like(distance, dtype=bool), 1)
    cdf = np.cumsum(probability, axis=1)[:, :-1]
    maximum_cdf_reversal = 0.0
    gap_identity_error = 0.0
    for i in range(probability.shape[0] - 1):
        difference = cdf[i] - cdf[i + 1:]
        reversal = np.maximum(-difference, 0.0)
        if reversal.size:
            maximum_cdf_reversal = max(maximum_cdf_reversal, float(np.max(reversal)))
        predicted_gap = 2.0 * np.sum(reversal, axis=1)
        gap_identity_error = max(
            gap_identity_error,
            float(np.max(np.abs(gap[i, i + 1:] - predicted_gap))),
        )
    eigenvalues = classical_mds_spectrum(distance)
    positive = eigenvalues[eigenvalues > tolerance]
    negative = eigenvalues[eigenvalues < -tolerance]
    denominator = float(np.linalg.norm(distance[mask]))
    summary = {
        "max_pairwise_gap": float(np.max(gap[mask])) if np.any(mask) else 0.0,
        "relative_line_distance_deviation": (
            float(np.linalg.norm(gap[mask]) / denominator) if denominator else 0.0
        ),
        "violating_pair_fraction": float(np.mean(gap[mask] > tolerance)) if np.any(mask) else 0.0,
        "maximum_cdf_reversal": maximum_cdf_reversal,
        "gap_identity_error": gap_identity_error,
        "mds_second_to_first": (
            float(positive[1] / positive[0]) if len(positive) > 1 else 0.0
        ),
        "mds_negative_mass_fraction": (
            float(-np.sum(negative) / np.sum(positive)) if len(negative) and len(positive) else 0.0
        ),
        "mds_rank1_positive_fraction": (
            float(positive[0] / np.sum(positive)) if len(positive) else 1.0
        ),
    }
    return summary, distance, eigenvalues


def current_backflow_diagnostics(q: float, a: float, chi: np.ndarray, coordinates: np.ndarray,
                                 tolerance: float = 1e-10) -> dict:
    """Diagnose inward probability current in Krylov-index space."""
    current = probability_current(q, a, chi)
    negative = np.maximum(-current, 0.0)
    total_negative = np.sum(negative, axis=1)
    result = {
        "max_inward_current": float(np.max(negative)),
        "max_total_inward_current": float(np.max(total_negative)),
        "first_detected_backflow_coordinate": None,
        "first_detected_backflow_link": None,
        "current_tolerance": float(tolerance),
    }
    locations = np.argwhere(current < -tolerance)
    if len(locations):
        row, link = locations[0]
        result["first_detected_backflow_coordinate"] = float(coordinates[row])
        result["first_detected_backflow_link"] = int(link)
    return result


def poisson_limit_probability(a: float, times: np.ndarray, nmax: int) -> np.ndarray:
    """Exact q=1 limit: Poisson probabilities with mean mu=(a t)^2."""
    times = np.asarray(times, dtype=float)
    if np.any(times < 0) or not np.isfinite(a) or a <= 0 or nmax < 1:
        raise ValueError("require times >= 0, a > 0, and nmax >= 1")
    mu = (a * times)**2
    n = np.arange(nmax + 1)
    probability = np.empty((len(times), nmax + 1), dtype=float)
    for i, value in enumerate(mu):
        if value == 0:
            probability[i] = 0.0
            probability[i, 0] = 1.0
        else:
            probability[i] = np.exp(-value + n * np.log(value) - gammaln(n + 1))
    return probability


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
    """Generate figures, CSV, JSON, and LaTeX tables used by the research note."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)

    # Existing comparison with the coefficient-matched log-cosh curve.
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

    # Full pairwise-distance geometry on a finer time grid.
    xg = np.linspace(0, 4, 401)
    geometry_qvalues = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, float(np.exp(-0.004))]
    geometry_records = []
    mds_records = {}
    for q in geometry_qvalues:
        a = 1.0
        times = xg/(a*np.sqrt(1-q))
        front = 2*xg[-1]/(1-q)
        nmax = int(np.ceil(front+12*np.sqrt(front+1)+64))
        chi, _ = evolve(q, a, times, nmax)
        probability = chi**2
        geometry, distance, eigenvalues = distance_geometry(probability)
        backflow = current_backflow_diagnostics(q, a, chi, xg, tolerance=1e-10)
        geometry_records.append({"q": q, "nmax": nmax, **geometry, **backflow})
        mds_records[f"q={q:.8f}"] = [float(v) for v in eigenvalues[:12]]

    np.savetxt(
        output/"geometry_scan.csv",
        np.array([[r["q"], r["relative_line_distance_deviation"], r["max_pairwise_gap"],
                   r["max_inward_current"], r["mds_second_to_first"],
                   r["mds_negative_mass_fraction"]] for r in geometry_records]),
        delimiter=",",
        header="q,relative_line_distance_deviation,max_pairwise_gap,max_inward_current,mds_second_to_first,mds_negative_mass_fraction",
        comments="",
    )

    # Exact q=1 harmonic-oscillator limit.  Here P_n is Poisson with mean (a t)^2.
    poisson_times = np.linspace(0, 4, 81)
    poisson_probability = poisson_limit_probability(1.0, poisson_times, 120)
    poisson_geometry, _, poisson_eigenvalues = distance_geometry(poisson_probability, tolerance=1e-11)
    poisson_geometry["tail_probability_at_tmax"] = float(1-poisson_probability[-1].sum())
    poisson_geometry["first_mds_eigenvalues"] = [float(v) for v in poisson_eigenvalues[:5]]

    # A fixed-charge interpretation for a representative double-scaling parameter lambda=0.001.
    charge_records = []
    lam = 0.001
    for charge in [0.0, 0.2, 0.4, 0.45, 0.48, 0.49]:
        q = float(np.exp(-4*lam/(1-4*charge**2)))
        times = xg/np.sqrt(1-q)
        front = 2*xg[-1]/(1-q)
        nmax = int(np.ceil(front+12*np.sqrt(front+1)+64))
        chi, _ = evolve(q, 1.0, times, nmax)
        geometry, _, _ = distance_geometry(chi**2)
        backflow = current_backflow_diagnostics(q, 1.0, chi, xg, tolerance=1e-10)
        charge_records.append({"lambda": lam, "charge": charge, "q": q,
                               "relative_line_distance_deviation": geometry["relative_line_distance_deviation"],
                               "max_pairwise_gap": geometry["max_pairwise_gap"],
                               "max_inward_current": backflow["max_inward_current"]})

    # CDF crossing for the explicit counterexample used in the note.
    example_times = np.linspace(0, 2.5, 101)
    example_chi, _ = evolve(0.2, 1.0, example_times, 180)
    cdf_difference = np.cumsum(example_chi[80]**2-example_chi[100]**2)
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    k = np.arange(min(16, len(cdf_difference)))
    ax.axhline(0.0, linewidth=0.8)
    ax.plot(k, cdf_difference[:len(k)], marker="o")
    ax.set(xlabel=r"Krylov index cutoff $k$",
           ylabel=r"$F_{t_1}(k)-F_{t_2}(k)$")
    ax.set_xticks(k)
    fig.tight_layout()
    fig.savefig(output/"cdf_crossing.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    qplot = np.array([r["q"] for r in geometry_records])
    deviation = np.maximum(np.array([r["relative_line_distance_deviation"] for r in geometry_records]), 1e-16)
    ax.semilogy(qplot, deviation, marker="o")
    ax.set(xlabel=r"deformation parameter $q$",
           ylabel=r"relative deviation from $|C_i-C_j|$")
    fig.tight_layout()
    fig.savefig(output/"line_distance_deviation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    inward = np.maximum(np.array([r["max_inward_current"] for r in geometry_records]), 1e-16)
    ax.semilogy(qplot, inward, marker="o")
    ax.set(xlabel=r"deformation parameter $q$",
           ylabel=r"maximum inward Krylov probability current")
    fig.tight_layout()
    fig.savefig(output/"backflow_vs_q.pdf")
    plt.close(fig)

    geometry_rows = []
    for row in geometry_records:
        geometry_rows.append(
            f"{row['q']:.6f} & {row['relative_line_distance_deviation']:.2e} & "
            f"{row['max_pairwise_gap']:.2e} & {row['max_inward_current']:.2e} & "
            f"{row['mds_second_to_first']:.2e} " + chr(92)*2
        )
    geometry_table = (
        r"\begin{tabular}{rrrrr}\toprule"+"\n"
        +r"$q$ & 相対距離偏差 & 最大距離差 & 最大内向き流 & $\lambda_2^+/\lambda_1^+$\\\midrule"+"\n"
        +"\n".join(geometry_rows)+"\n"+r"\bottomrule\end{tabular}"+"\n"
    )
    (output/"geometry_table.tex").write_text(geometry_table, encoding="utf-8")

    summary = {
        "source": "arXiv:2512.07715v2, Eqs. (3.33),(3.34)",
        "scaling": "x=a*sqrt(1-q)*t; scaled_C=(1-q)*C/2; a=b_1",
        "pairwise_transport": pairwise_example(),
        "geometry_window": "0 <= x=a*sqrt(1-q)*t <= 4 on 401 uniform samples",
        "geometry_runs": geometry_records,
        "poisson_q1_limit": poisson_geometry,
        "charge_scan": charge_records,
        "mds_spectra": mds_records,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "runs": records,
    }
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    run(parser.parse_args().output)
