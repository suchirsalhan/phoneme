"""Tests for phoneme_entropy.maxent (no upstream dependency required)."""

import numpy as np
import pytest

from phoneme_entropy.maxent import compute_maxent_from_matrix


def test_uniform_recovered_when_targets_match_uniform():
    # With a single feature and a target equal to the uniform-distribution mean,
    # the max-ent solution is the uniform distribution (lambda ~ 0).
    F = np.array([[0.0], [1.0], [2.0], [3.0]])
    uniform_mean = F.mean()  # = 1.5
    probs, lambdas = compute_maxent_from_matrix(F, [uniform_mean])

    assert probs.shape == (4,)
    assert np.isclose(probs.sum(), 1.0)
    np.testing.assert_allclose(probs, np.full(4, 0.25), atol=1e-4)
    assert np.isclose(lambdas[0], 0.0, atol=1e-4)


def test_expectations_match_targets():
    # The fitted distribution must reproduce the requested feature expectations.
    F = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    targets = np.array([1.2, 0.9])
    probs, _ = compute_maxent_from_matrix(F, targets)

    achieved = probs @ F
    np.testing.assert_allclose(achieved, targets, atol=1e-4)
    assert np.isclose(probs.sum(), 1.0)


def test_shifting_target_shifts_mass():
    # A higher target expectation should move probability mass toward larger
    # feature values (monotone response of the exponential family).
    F = np.array([[0.0], [1.0], [2.0], [3.0]])
    probs_low, _ = compute_maxent_from_matrix(F, [1.0])
    probs_high, _ = compute_maxent_from_matrix(F, [2.0])

    # F is (n, 1); (n,) @ (n, 1) -> (1,). Take element 0 for a Python scalar.
    mean_low = float((probs_low @ F)[0])
    mean_high = float((probs_high @ F)[0])
    assert mean_high > mean_low
