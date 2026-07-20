"""Maximum-entropy distribution fitting from a feature matrix.

Ports ``original/cost_fits.py``. Given a discrete support of ``n`` points, a
feature matrix ``F`` (shape ``(n, m)``), and a vector of ``m`` target feature
expectations, this fits the maximum-entropy distribution whose feature
expectations match the targets.

Theory
------
The maximum-entropy distribution subject to expectation constraints
``E_p[F_k] = c_k`` has the Gibbs form ``p_j proportional to exp(sum_k lambda_k F_jk)``.
The Lagrange multipliers ``lambda`` are found by minimising the smooth convex
*dual* objective

    g(lambda) = log sum_j exp((F lambda)_j)  -  lambda . c

whose gradient is ``E_p[F] - c``. We minimise ``g`` with L-BFGS-B via
:func:`scipy.optimize.minimize`, supplying the analytic gradient and using
:func:`scipy.special.logsumexp` for a numerically stable log-partition function.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = ["compute_maxent_from_matrix"]


def compute_maxent_from_matrix(F, target_expectations, tol=1e-6):
    """Fit a maximum-entropy distribution from a precomputed feature matrix.

    Parameters
    ----------
    F : array-like, shape (n, m)
        Feature matrix. Row ``j`` holds the ``m`` feature values of support
        point ``j``.
    target_expectations : array-like, shape (m,)
        Desired expectation of each feature under the fitted distribution.
    tol : float, default 1e-6
        Optimisation tolerance passed to :func:`scipy.optimize.minimize`.

    Returns
    -------
    p : ndarray, shape (n,)
        The fitted maximum-entropy probability vector (sums to 1).
    lambdas : ndarray, shape (m,)
        The optimised Lagrange multipliers (one per feature).

    Raises
    ------
    RuntimeError
        If the optimiser reports failure to converge.
    """
    F = np.asarray(F)
    n, m = F.shape
    target_expectations = np.asarray(target_expectations)

    def neg_dual(lambdas):
        # Dual objective g(lambda) = logZ - lambda . c (convex in lambda).
        log_probs = np.dot(F, lambdas)
        logZ = logsumexp(log_probs)
        return logZ - np.dot(lambdas, target_expectations)

    def grad_neg_dual(lambdas):
        # Gradient of the dual = E_p[F] - c, where p is the current Gibbs dist.
        log_probs = np.dot(F, lambdas)
        logZ = logsumexp(log_probs)
        probs = np.exp(log_probs - logZ)
        expected_features = np.dot(probs, F)
        return expected_features - target_expectations

    # Start from the uniform distribution (lambda = 0).
    lambdas0 = np.zeros(m)
    res = minimize(neg_dual, lambdas0, jac=grad_neg_dual, tol=tol)

    if not res.success:
        raise RuntimeError("Optimization failed:", res.message)

    lambdas = res.x
    # Recover the fitted distribution from the optimal multipliers.
    log_probs = np.dot(F, lambdas)
    logZ = logsumexp(log_probs)
    probs = np.exp(log_probs - logZ)

    return probs, lambdas
