"""Smoke tests for the vendored entropy estimators (phoneme_entropy.estimators).

These confirm the bundled upstream code imports and runs; they are not a
re-derivation of the estimators' correctness (that lives upstream).
"""

import numpy as np
import pytest

from phoneme_entropy import Entropy  # re-exported at top level
from phoneme_entropy.estimators import (
    FreqShrink,
    JS_JensenShannon,
    JS_KullbackLeibler,
)


def test_entropy_empty_is_zero():
    assert Entropy([]) == 0.0


def test_entropy_uniform_is_log2_n_bits():
    # Uniform over 4 outcomes -> 2 bits (base-2 MLE).
    counts = [10, 10, 10, 10]
    assert np.isclose(Entropy(counts, method="MLE", base=2), 2.0)


@pytest.mark.parametrize("method", ["MLE", "JSE", "CAE", "CWJ", "NBRS", "NSB"])
def test_all_methods_run_and_are_nonnegative(method):
    counts = [40, 25, 15, 12, 8, 3, 1, 1]
    # NSB returns an mpmath.mpf; coerce to float for the numeric checks.
    h = float(Entropy(counts, method=method))
    assert np.isfinite(h)
    assert h >= 0.0


def test_freqshrink_is_a_distribution():
    p = FreqShrink([5, 3, 2, 0, 1])
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


def test_divergences_of_identical_counts_are_zero():
    counts = [5, 3, 2, 1]
    assert np.isclose(JS_KullbackLeibler(counts, counts), 0.0)
    assert np.isclose(JS_JensenShannon(counts, counts), 0.0)
