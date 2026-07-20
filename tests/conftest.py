"""Shared pytest fixtures and import guards.

The segmentation and dirichlet modules depend on the upstream `entropy_estimators`
package. If it is not installed, the tests that need it are skipped with a clear
message rather than erroring at collection time.
"""

import importlib.util

import pytest

HAS_ENTROPY_ESTIMATORS = importlib.util.find_spec("entropy_estimators") is not None

requires_entropy_estimators = pytest.mark.skipif(
    not HAS_ENTROPY_ESTIMATORS,
    reason=(
        "entropy-estimators is not installed; "
        "install with pip install 'git+https://github.com/fermosc24/entropy-estimators.git'"
    ),
)


@pytest.fixture
def toy_words():
    """A tiny phoneme-string lexicon with frequencies."""
    words = ["k a t", "k a p", "d o g", "d o t", "k a"]
    frequencies = [10, 5, 8, 3, 2]
    return words, frequencies
