"""Tests for phoneme_entropy.segmentation (requires entropy-estimators for
prefix entropy; segment_informativity is self-contained)."""

import math

import numpy as np
import pandas as pd
import pytest

from phoneme_entropy.segmentation import phoneme_prefix_entropy, segment_informativity


def test_segment_informativity_columns_and_inventory(toy_words):
    words, freqs = toy_words
    df = segment_informativity(words, freqs)

    assert list(df.columns) == [
        "Phoneme",
        "Surprisal",
        "Weight",
        "Occurrences",
        "Continuations",
    ]
    # Inventory is the set of distinct phonemes across all words.
    inventory = {p for w in words for p in w.split()}
    assert set(df["Phoneme"]) == inventory


def test_segment_informativity_empty_returns_empty_frame():
    df = segment_informativity([], [])
    assert len(df) == 0
    assert "Surprisal" in df.columns


def test_segment_informativity_validates_inputs():
    with pytest.raises(ValueError):
        segment_informativity(["a b"], [1, 2])  # length mismatch
    with pytest.raises(ValueError):
        segment_informativity(["a b"], [-1])  # negative frequency
    with pytest.raises(ValueError):
        segment_informativity(["a b"], alpha=0)  # alpha must be > 0
    with pytest.raises(ValueError):
        segment_informativity(["a b"], base=1)  # base must differ from 1


def test_segment_informativity_word_final_surprisal_is_log_v():
    # A single-phoneme word has only a word-final occurrence, whose smoothed
    # surprisal is log_base(V) with V = inventory size.
    words = ["a", "b c"]
    df = segment_informativity(words, base=2)
    V = 3  # {a, b, c}
    row_a = df.loc[df["Phoneme"] == "a"].iloc[0]
    assert row_a["Continuations"] == 0
    assert math.isclose(row_a["Surprisal"], math.log(V, 2), rel_tol=1e-9)


def test_phoneme_prefix_entropy_columns(toy_words):
    words, freqs = toy_words
    df = phoneme_prefix_entropy(words, freqs, smoothing="ML")
    assert list(df.columns) == ["Phoneme", "Entropy", "Weight", "Prefixes"]
    assert len(df) > 0
    assert (df["Prefixes"] >= 1).all()
