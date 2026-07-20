"""Segmental informativity: lexical information gain and next-phoneme surprisal.

This module ports ``original/phoneme_info.py``. Both public functions operate on
words encoded as **space-separated phoneme strings** (e.g. ``"k a t"``) and
return tidy :class:`pandas.DataFrame` summaries keyed by phoneme.

* :func:`phoneme_prefix_entropy` measures the **lexical information gain** at each
  phoneme: the mean entropy of the set of words still compatible with the prefix
  that *ends* in that phoneme — i.e. how much lexical uncertainty remains (and is
  progressively resolved) at that point. It uses
  :func:`entropy_estimators.Entropy` so the entropy estimator (ML, CWJ, NSB, ...)
  is configurable. (The function keeps its historical name for API stability.)

* :func:`segment_informativity` measures, for each phoneme, the frequency-weighted
  mean *surprisal* of the phoneme that follows it, under additive (Laplace/Lidstone)
  smoothing and with no end-of-word symbol.

The numerics are unchanged from the original research script; only documentation,
comments, and the import path were added.
"""

from __future__ import annotations

import builtins
from collections import defaultdict
from math import log

import pandas as pd

# Entropy estimator, vendored into this package from entropy-estimators
# (was ``from entropies import *`` behind a sys.path hack).
from .estimators import Entropy

__all__ = ["phoneme_prefix_entropy", "segment_informativity"]


def phoneme_prefix_entropy(
    words,
    frequencies=None,
    weighted=True,
    smoothing="CWJ",
):
    """Lexical information gain: mean entropy of the words still compatible
    after each phoneme.

    For every prefix of every word, we collect the (frequency-weighted) set of
    words sharing that prefix and estimate its entropy with
    :func:`entropy_estimators.Entropy`. Each prefix is then attributed to its
    *final* phoneme, and the per-phoneme entropies are averaged (optionally
    frequency-weighted).

    Parameters
    ----------
    words : iterable of str
        Words encoded as space-separated phoneme strings.
    frequencies : iterable of numbers, optional
        Per-word frequencies. If ``None``, every word is weighted equally (1.0).
    weighted : bool, default True
        If ``True``, average per-phoneme entropies weighted by the total prefix
        weight; otherwise take a plain mean over prefixes.
    smoothing : str, default "CWJ"
        Entropy estimator passed to ``Entropy`` (e.g. ``"ML"``, ``"CWJ"``,
        ``"NSB"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``Phoneme``, ``Entropy``, ``Weight``, ``Prefixes``.

    Raises
    ------
    ValueError
        If ``words`` and ``frequencies`` differ in length.
    """
    words = list(words)

    if frequencies is None:
        frequencies = [1.0] * len(words)
    else:
        frequencies = list(frequencies)

    if len(words) != len(frequencies):
        raise ValueError("words and frequencies must have the same length")

    # Map every prefix (as a tuple of phonemes) to the indices of words that
    # start with it. A word of length L contributes L prefixes.
    prefix_to_word_indices = defaultdict(list)

    for word_index, word in enumerate(words):
        phonemes = word.split()
        for position in range(len(phonemes)):
            prefix = tuple(phonemes[: position + 1])
            prefix_to_word_indices[prefix].append(word_index)

    # For each prefix, entropy of the compatible-word frequency distribution,
    # bucketed by the prefix's final phoneme.
    phoneme_values = defaultdict(list)

    for prefix, indices in prefix_to_word_indices.items():
        counts = [frequencies[i] for i in indices if frequencies[i] > 0]

        if not counts:
            continue

        H = Entropy(counts, method=smoothing)
        W = builtins.sum(counts)

        # prefix[-1] is the phoneme that this prefix ends on.
        phoneme_values[prefix[-1]].append((H, W))

    rows = []

    for phoneme in sorted(phoneme_values):
        values = phoneme_values[phoneme]

        if weighted:
            # Frequency-weighted mean entropy across all prefixes ending here.
            total_weight = builtins.sum(w for _, w in values)
            mean_entropy = builtins.sum(h * w for h, w in values) / total_weight
        else:
            # Unweighted mean over prefixes.
            total_weight = len(values)
            mean_entropy = builtins.sum(h for h, _ in values) / len(values)

        rows.append(
            {
                "Phoneme": phoneme,
                "Entropy": mean_entropy,
                "Weight": total_weight,
                "Prefixes": len(values),
            }
        )

    return pd.DataFrame(rows)


def segment_informativity(
    words,
    frequencies=None,
    base=2,
    alpha=0.5,
):
    """Frequency-weighted mean surprisal of the next phoneme after each phoneme.

    For non-final phoneme occurrences, the next-phoneme probability is estimated
    from all words sharing the prefix up to and including the current phoneme,
    with additive (Lidstone) smoothing of strength ``alpha`` over the phoneme
    inventory of size ``V``.

    For word-final occurrences there is no observed continuation, so the smoothed
    distribution is uniform over the inventory and the surprisal is
    ``log_base(V)``.

    Parameters
    ----------
    words : iterable of str
        Words encoded as space-separated phoneme strings.
    frequencies : iterable of numbers, optional
        Word frequencies. If ``None``, every word receives frequency 1.
    base : float, default 2
        Logarithm base (2 -> bits).
    alpha : float, default 0.5
        Additive smoothing parameter (must be > 0).

    Returns
    -------
    pandas.DataFrame
        Columns: ``Phoneme``, ``Surprisal``, ``Weight``, ``Occurrences``,
        ``Continuations``.

    Raises
    ------
    ValueError
        If lengths mismatch, any frequency is negative, ``alpha <= 0``, or
        ``base`` is non-positive or equal to 1.
    """
    words = list(words)

    if frequencies is None:
        frequencies = [1.0] * len(words)
    else:
        frequencies = list(frequencies)

    if len(words) != len(frequencies):
        raise ValueError("words and frequencies must have the same length")

    if any(freq < 0 for freq in frequencies):
        raise ValueError("frequencies must be non-negative")

    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")

    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from 1")

    tokenised_words = [word.split() for word in words]

    # Phoneme inventory and its size V (used by the additive-smoothing denominator).
    inventory = {
        phoneme for phonemes in tokenised_words for phoneme in phonemes
    }

    V = len(inventory)

    if V == 0:
        return pd.DataFrame(
            columns=[
                "Phoneme",
                "Surprisal",
                "Weight",
                "Occurrences",
                "Continuations",
            ]
        )

    # Weighted next-phoneme counts for each prefix. Word-final positions have no
    # continuation and therefore contribute nothing to these counts.
    next_counts = defaultdict(lambda: defaultdict(float))

    for phonemes, frequency in zip(tokenised_words, frequencies):
        if frequency <= 0:
            continue

        for position in range(len(phonemes) - 1):
            prefix = tuple(phonemes[: position + 1])
            next_phoneme = phonemes[position + 1]
            next_counts[prefix][next_phoneme] += frequency

    # Per-phoneme accumulators.
    surprisal_sum = defaultdict(float)
    weight_sum = defaultdict(float)
    occurrence_count = defaultdict(int)
    continuation_count = defaultdict(int)

    for phonemes, frequency in zip(tokenised_words, frequencies):
        if frequency <= 0:
            continue

        for position, current_phoneme in enumerate(phonemes):
            occurrence_count[current_phoneme] += 1
            weight_sum[current_phoneme] += frequency

            prefix = tuple(phonemes[: position + 1])
            counts = next_counts[prefix]
            total = builtins.sum(counts.values())

            if position < len(phonemes) - 1:
                # Observed continuation: Lidstone-smoothed conditional probability.
                observed_next = phonemes[position + 1]
                probability = (counts.get(observed_next, 0.0) + alpha) / (
                    total + alpha * V
                )
                continuation_count[current_phoneme] += 1
            else:
                # Word-final: no continuation counts, so additive smoothing
                # yields a uniform distribution -> probability alpha / (alpha*V).
                probability = alpha / (alpha * V)

            surprisal = -log(probability, base)
            surprisal_sum[current_phoneme] += frequency * surprisal

    rows = []

    for phoneme in sorted(inventory):
        rows.append(
            {
                "Phoneme": phoneme,
                "Surprisal": (
                    surprisal_sum[phoneme] / weight_sum[phoneme]
                    if weight_sum[phoneme] > 0
                    else float("nan")
                ),
                "Weight": weight_sum[phoneme],
                "Occurrences": occurrence_count[phoneme],
                "Continuations": continuation_count[phoneme],
            }
        )

    return pd.DataFrame(rows)
