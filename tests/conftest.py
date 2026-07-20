"""Shared pytest fixtures.

The entropy estimators are vendored into ``phoneme_entropy.estimators``, so no
optional-dependency guards are needed — everything is importable directly.
"""

import pytest


@pytest.fixture
def toy_words():
    """A tiny phoneme-string lexicon with frequencies."""
    words = ["k a t", "k a p", "d o g", "d o t", "k a"]
    frequencies = [10, 5, 8, 3, 2]
    return words, frequencies
