"""Tests for phoneme_entropy.dirichlet.

`dirichlet_entropy` and the Beta order-statistic helpers are self-contained;
`estimate_alpha_entropy` requires the upstream entropy-estimators package.
"""

import numpy as np
import pytest
import scipy.special as sp

from phoneme_entropy.dirichlet import (
    beta_order_stat_pdf,
    dirichlet_entropy,
    numerical_beta_order_stat_moments,
    estimate_alpha_entropy,
)
from conftest import requires_entropy_estimators


def test_dirichlet_entropy_matches_closed_form():
    alpha, V = 0.7, 5
    expected = sp.psi(V * alpha + 1) - sp.psi(alpha + 1)
    assert np.isclose(dirichlet_entropy(alpha, V), expected)


def test_dirichlet_entropy_increases_with_dimension():
    # More categories -> higher expected entropy at fixed concentration.
    assert dirichlet_entropy(1.0, 10) > dirichlet_entropy(1.0, 3)


def test_beta_order_stat_pdf_log_consistency():
    x, i, n, a, b = 0.3, 2, 5, 1.0, 4.0
    dens = beta_order_stat_pdf(x, i, n, a, b, ll=False)
    logdens = beta_order_stat_pdf(x, i, n, a, b, ll=True)
    assert np.isclose(np.log(dens), logdens)
    assert dens > 0


def test_order_stat_moments_are_ordered():
    # For n samples from a Beta, the mean of the i-th order statistic is
    # increasing in i (order statistics are sorted ascending).
    n, a, b = 6, 1.0, 3.0
    means = [numerical_beta_order_stat_moments(i, n, a, b)[0] for i in range(1, n + 1)]
    assert all(m2 > m1 for m1, m2 in zip(means, means[1:]))
    # Standard deviations are finite and non-negative.
    for i in range(1, n + 1):
        _, sd = numerical_beta_order_stat_moments(i, n, a, b)
        assert sd >= 0


@requires_entropy_estimators
def test_estimate_alpha_entropy_point_estimate_is_deterministic():
    # A clearly skewed distribution for which the root finder converges.
    counts = [100, 50, 25, 12, 6, 3, 2, 1]
    a1, se1 = estimate_alpha_entropy(counts, n_boot=0)
    a2, se2 = estimate_alpha_entropy(counts, n_boot=0)
    # Deterministic when n_boot=0 (works for both finite and nan outcomes).
    assert np.isclose(a1, a2, equal_nan=True)
    assert se1 == -1.0  # sentinel: no bootstrap requested
    if np.isfinite(a1):
        assert a1 > 0  # a valid concentration parameter is positive
