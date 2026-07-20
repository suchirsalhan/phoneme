"""Symmetric-Dirichlet inference and Beta order-statistic tools.

This module ports ``original/sym_dirichlet.py`` into the packaged library. The
numerical routines are **unchanged** — only documentation, comments, and the
import path have been added/cleaned. The single upstream dependency is
:func:`entropy_estimators.Entropy`, which supplies the entropy estimator used to
invert the Dirichlet concentration parameter.

Background
----------
A symmetric Dirichlet distribution ``Dir(alpha, ..., alpha)`` over a
``V``-dimensional simplex is a natural prior for categorical/rank distributions
(e.g. phoneme-type probabilities). Its *expected* Shannon entropy has a closed
form (:func:`dirichlet_entropy`), so a single observed entropy value can be used
to recover ``alpha`` by root finding (:func:`estimate_alpha_entropy`). Given
``alpha``, the marginal of each *order statistic* of the sorted probability
vector is Beta-distributed, which lets us draw a predicted rank/probability curve
with confidence bands (:func:`plot_ranks`).

References
----------
* Moscoso del Prado Martín, F. ``entropy-estimators``:
  https://github.com/fermosc24/entropy-estimators (upstream entropy estimators).
* Nemenman, Shafee & Bialek (2002), "Entropy and inference, revisited"
  (NSB estimator, one of the ``method`` options exposed via ``Entropy``).
"""

from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.integrate import quad

# Entropy estimator, vendored into this package from entropy-estimators.
# (Originally imported via ``sys.path.append('/Users/fermin/...'); from entropies
# import *``.)
from .estimators import Entropy

# ``matplotlib`` is only needed for :func:`plot_ranks`. Import lazily inside the
# function so that headless installs (e.g. CI, servers) can use the rest of the
# module without a plotting backend.

__all__ = [
    "plot_ranks",
    "beta_order_stat_pdf",
    "numerical_beta_order_stat_moments",
    "dirichlet_entropy",
    "estimate_alpha_entropy",
]


def plot_ranks(
    counts,
    alpha=None,
    loglog=True,
    title=None,
    types="order",
    error=95,
    reduced=False,
    ylim=None,
):
    """Plot predicted vs. observed probability ranks using Beta order statistics.

    The observed probabilities (``counts`` normalised and sorted ascending) are
    overlaid on the *predicted* order-statistic means of a symmetric
    ``Dir(alpha, ..., alpha)`` distribution, with shaded confidence bands.

    Parameters
    ----------
    counts : array-like
        Frequency counts of the observed items (one entry per type).
    alpha : float, optional
        Dirichlet concentration parameter. If ``None`` it is estimated from
        ``counts`` via :func:`estimate_alpha_entropy` (point estimate, no
        bootstrap).
    loglog : bool, default True
        Use log-log axes.
    title : str, optional
        Plot title.
    types : str, default "order"
        ``"order"`` labels the x-axis as the ``i``-th order statistic; any other
        value reverses the rank axis and relabels it as a rank ``r``.
    error : int | float, default 95
        Confidence level for the wide band. An ``int`` is read as a percentage
        (e.g. ``95`` -> 95%); a ``float`` as a proportion (e.g. ``0.95``). Any
        other type falls back to a 1-sigma band.
    reduced : int | False, default False
        If truthy, evaluate the (expensive) order-statistic moments at only
        ``reduced`` log-spaced ranks and interpolate, for speed on large ``V``.
    ylim : tuple, optional
        Explicit y-axis limits.

    Returns
    -------
    None
        Draws onto the current matplotlib axes; call ``plt.show()`` / ``savefig``
        afterwards.
    """
    import matplotlib.pyplot as plt

    V = len(counts)
    # Observed probabilities, sorted ascending (smallest first).
    ps = np.sort(np.array(counts) / np.sum(counts))

    if alpha is None:
        # Estimate the concentration parameter from the data itself.
        # (The BMA-constrained alternative is kept commented for provenance.)
        # entropy_val = Entropy(counts, method="CWJ")
        # alpha, _, _ = estimate_alpha_bma_order_entropy_constrained(ps, entropy_val)
        alpha, _ = estimate_alpha_entropy(counts, len(counts), n_boot=0)

    ranks = np.arange(1, V + 1)
    if reduced:
        # Evaluate on a log-spaced subset of ranks and interpolate afterwards.
        ranks2 = np.unique(
            np.round(np.exp(np.linspace(0, np.log(V), reduced)))
        ).astype(int)
        marker1 = marker2 = ""
    else:
        ranks2 = ranks
        marker1 = "."
        marker2 = "o"

    # Mean and standard error of each requested order statistic under
    # Beta(alpha, (V-1)*alpha) — the marginal of a symmetric Dirichlet coordinate.
    means, std_errors = zip(
        *[
            numerical_beta_order_stat_moments(i, V, alpha, (V - 1) * alpha)
            for i in ranks2
        ]
    )

    if reduced:
        # Interpolate the coarse grid back onto every rank, then renormalise so
        # the predicted probabilities sum to one.
        means2 = np.interp(ranks, ranks2, means)
        means = np.array(means) / np.sum(means2)
        std_errors = np.array(std_errors) / np.sum(means2)
    else:
        means = np.array(means) / np.sum(means)
        std_errors = np.array(std_errors) / np.sum(means)

    # Convert the requested confidence level into a z-multiplier.
    if type(error) is int:
        n = -sp.stats.norm.ppf((1 - error / 100) / 2)
    elif type(error) is float:
        n = -sp.stats.norm.ppf((1 - error) / 2)
    else:
        n = 1

    if types != "order":
        # Reverse the rank axis (largest probability = rank 1).
        ranks = V + 1 - ranks
        ranks2 = V + 1 - ranks2

    # Predicted means with the wide confidence band...
    plt.plot(ranks2, means, color="green", marker=marker1, label="Predicted Mean")
    plt.fill_between(
        ranks2,
        np.clip(means - n * std_errors, 0, 1),
        np.clip(means + n * std_errors, 0, 1),
        alpha=0.2,
        color="green",
    )
    # ...and the tighter 1-sigma band.
    plt.fill_between(
        ranks2,
        np.clip(means - std_errors, 0, 1),
        np.clip(means + std_errors, 0, 1),
        alpha=0.4,
        color="green",
    )

    # Observed probabilities.
    plt.plot(ranks, np.sort(ps), color="red", label="Observed", marker=marker2)

    # Axis labels depend on whether we are showing order statistics or ranks.
    if types == "order":
        plt.xlabel(r"$i$-th Order Statistic")
        plt.ylabel(r"$\mathbb{P}_{(i)}$")
    else:
        plt.xlabel(r"Rank ($r$)")
        plt.ylabel(r"$P(r)$")

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    if title:
        plt.title(title)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def beta_order_stat_pdf(x, i, n, a, b, ll=False):
    """PDF of the ``i``-th order statistic from a ``Beta(a, b)`` sample of size ``n``.

    Computed in log-space for numerical stability, then exponentiated unless
    ``ll=True``.

    Parameters
    ----------
    x : float or array-like
        Evaluation point(s) in ``[0, 1]``.
    i : int
        Order-statistic index (1-based; ``i=1`` is the minimum).
    n : int
        Total number of samples.
    a, b : float
        Shape parameters of the underlying Beta distribution.
    ll : bool, default False
        If ``True`` return the log-density instead of the density.

    Returns
    -------
    float or ndarray
        The (log-)density at ``x``.
    """
    # log of the combinatorial normaliser  n! / ((i-1)! (n-i)!)  written with
    # log-gammas, plus log(n).
    coeff = (
        sp.special.loggamma(n)
        - sp.special.loggamma(i)
        - sp.special.loggamma(n - i + 1)
        + np.log(n)
    )
    beta_pdf = sp.stats.beta.logpdf(x, a, b)
    beta_cdf = sp.stats.beta.logcdf(x, a, b)
    beta_sf = sp.stats.beta.logsf(x, a, b)  # log survival function = log(1 - CDF)
    res = coeff + beta_pdf + (i - 1) * beta_cdf + (n - i) * beta_sf
    if not ll:
        res = np.exp(res)
    return res


def numerical_beta_order_stat_moments(i, n, a, b):
    """Numerically integrate the mean and std. dev. of a Beta order statistic.

    Uses :func:`scipy.integrate.quad` against :func:`beta_order_stat_pdf` to
    obtain the first two moments of the ``i``-th order statistic of ``n`` samples
    from ``Beta(a, b)``.

    Parameters
    ----------
    i : int
        Order-statistic index (1-based).
    n : int
        Total number of samples.
    a, b : float
        Beta shape parameters.

    Returns
    -------
    tuple of float
        ``(mean, std_dev)`` of the order statistic.
    """

    def integrand_mean(x):
        return x * beta_order_stat_pdf(x, i, n, a, b)

    def integrand_sq(x):
        return x**2 * beta_order_stat_pdf(x, i, n, a, b)

    mean, _ = quad(integrand_mean, 0, 1, epsabs=1e-10)
    mean_sq, _ = quad(integrand_sq, 0, 1, epsabs=1e-10)
    std_dev = np.sqrt(mean_sq - mean**2)

    return mean, std_dev


def dirichlet_entropy(alpha, V):
    """Expected Shannon entropy of a symmetric ``Dir(alpha, ..., alpha)`` prior.

    Closed form ``psi(V*alpha + 1) - psi(alpha + 1)`` in nats, where ``psi`` is
    the digamma function.

    Parameters
    ----------
    alpha : float
        Concentration parameter.
    V : int
        Dimensionality of the simplex (number of categories).

    Returns
    -------
    float
        Expected entropy (nats).
    """
    return sp.special.psi(V * alpha + 1) - sp.special.psi(alpha + 1)


def estimate_alpha_entropy(counts, k=None, n_boot=500, method="CWJ", x0=1):
    """Estimate the symmetric-Dirichlet ``alpha`` that matches an observed entropy.

    The observed entropy of ``counts`` is estimated with
    :func:`entropy_estimators.Entropy` (default estimator ``"CWJ"``), and
    ``alpha`` is recovered by root-finding on :func:`dirichlet_entropy` (see
    :func:`_h_estimate`). An optional parametric bootstrap resamples counts from
    the empirical multinomial to attach a standard error to ``alpha**2``.

    Parameters
    ----------
    counts : list or ndarray
        Observed counts from a (presumed) Dirichlet-multinomial sample.
    k : int, optional
        True dimensionality of the Dirichlet. If ``None`` or smaller than
        ``len(counts)``, it is set to ``len(counts)``.
    n_boot : int, default 500
        Number of bootstrap resamples for the standard error. Use ``0`` for a
        deterministic point estimate with no standard error.
    method : str, default "CWJ"
        Entropy estimator name passed through to ``Entropy`` (e.g. ``"ML"``,
        ``"CWJ"``, ``"NSB"``, ...).
    x0 : float, default 1
        Initial guess for the root finder.

    Returns
    -------
    tuple of float
        ``(alpha, se)`` where ``alpha`` is the estimate and ``se`` the bootstrap
        standard error of ``alpha`` (``-1.0`` if ``n_boot == 0``). Returns
        ``(nan, nan)`` if the root finder fails to converge.

    Notes
    -----
    The optimisation is parameterised in terms of ``a`` with ``alpha = a**2`` to
    keep the estimate strictly positive; hence the squared quantities below.
    """
    # Accept lists, ndarrays, or pandas Series interchangeably. Upstream
    # ``Entropy`` expects a plain Python list (it tests ``if not counts:``, which
    # is ambiguous for arrays/Series), so we coerce to a list of floats here.
    # This does not change any numerics.
    counts = [float(c) for c in counts]

    if k is None or k < len(counts):
        k = len(counts)
    h = Entropy(counts, method=method)
    opt = _h_estimate(h, k, x0=x0)
    if opt.converged:
        if n_boot:
            # Parametric bootstrap: resample counts from the fitted multinomial
            # and re-estimate to obtain a standard error for alpha**2.
            N = np.sum(counts)
            ps = np.array(counts) / N
            samp = sp.stats.multinomial.rvs(N, ps, size=n_boot)
            hsamp = np.array(
                [
                    _h_estimate(Entropy(samp[i, :], method=method), k, x0=x0).root
                    for i in range(n_boot)
                ]
            )
            se = sp.stats.sem(hsamp**2)
        else:
            se = -1.0
        return opt.root**2, se
    else:
        return np.nan, np.nan


def _h_estimate(h, k, method=None, x0=1, x1=0):
    """Root-find the ``alpha`` whose symmetric-Dirichlet entropy equals ``h``.

    Solves ``dirichlet_entropy(a**2, k) == h`` for ``a`` (so ``alpha = a**2`` is
    guaranteed non-negative), supplying the analytic derivative for Newton-style
    convergence via :func:`scipy.optimize.root_scalar`.

    Parameters
    ----------
    h : float
        Target entropy (nats).
    k : int
        Dimensionality of the simplex.
    method : optional
        Unused; kept for signature compatibility with the original script.
    x0, x1 : float
        Initial points for ``root_scalar``.

    Returns
    -------
    scipy.optimize.RootResults
        The full root-finder result (``.root``, ``.converged``, ...).
    """
    eq = lambda a: dirichlet_entropy(a**2, k) - h
    # d/da of dirichlet_entropy(a**2, k): chain rule through the trigamma terms.
    fprime1 = lambda a: 2 * a * (
        k * sp.special.polygamma(1, k * a**2 + 1)
        - sp.special.polygamma(1, a**2 + 1)
    )

    opt = sp.optimize.root_scalar(eq, x0=x0, x1=x1, fprime=fprime1)
    return opt
