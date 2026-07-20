from collections import defaultdict
import builtins
import pandas as pd
import sys
sys.path.append('/Users/fermin/CODE/PYTHON3/') # just to add to my searchpath
from entropies import *

def phoneme_prefix_entropy(
    words,
    frequencies=None,
    weighted=True,
    smoothing ="CWJ"
):
    """
    Mean entropy of the compatible words after each phoneme.
    """

    words = list(words)

    if frequencies is None:
        frequencies = [1.0] * len(words)
    else:
        frequencies = list(frequencies)

    if len(words) != len(frequencies):
        raise ValueError("words and frequencies must have the same length")

    prefix_to_word_indices = defaultdict(list)

    for word_index, word in enumerate(words):
        phonemes = word.split()
        for position in range(len(phonemes)):
            prefix = tuple(phonemes[:position + 1])
            prefix_to_word_indices[prefix].append(word_index)

    phoneme_values = defaultdict(list)

    for prefix, indices in prefix_to_word_indices.items():

        counts = [
            frequencies[i]
            for i in indices
            if frequencies[i] > 0
        ]

        if not counts:
            continue

        H = Entropy(counts,method=smoothing)
        W = builtins.sum(counts)

        phoneme_values[prefix[-1]].append((H, W))

    rows = []

    for phoneme in sorted(phoneme_values):

        values = phoneme_values[phoneme]

        if weighted:
            total_weight = builtins.sum(w for _, w in values)
            mean_entropy = (
                builtins.sum(h * w for h, w in values)
                / total_weight
            )
        else:
            total_weight = len(values)
            mean_entropy = (
                builtins.sum(h for h, _ in values)
                / len(values)
            )

        rows.append({
            "Phoneme": phoneme,
            "Entropy": mean_entropy,
            "Weight": total_weight,
            "Prefixes": len(values),
        })

    return pd.DataFrame(rows)

from math import log

def segment_informativity(
    words,
    frequencies=None,
    base=2,
    alpha=0.5,
):
    """
    Compute the frequency-weighted mean surprisal of the next phoneme after
    each phoneme, using additive smoothing and no end-of-word symbol.

    For non-final occurrences, the next-phoneme probability is estimated from
    all words sharing the prefix up to and including the current phoneme.

    For word-final occurrences, where no continuation is observed, the
    smoothed distribution is uniform over the phoneme inventory. Their
    surprisal is therefore log_base(V), where V is the inventory size.

    Parameters
    ----------
    words : iterable of str
        Words encoded as space-separated phoneme strings.

    frequencies : iterable of numbers or None, default=None
        Word frequencies. If None, every word receives frequency 1.

    base : float, default=2
        Logarithm base.

    alpha : float, default=0.5
        Additive smoothing parameter.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - Phoneme
        - Surprisal
        - Weight
        - Occurrences
        - Continuations
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

    inventory = {
        phoneme
        for phonemes in tokenised_words
        for phoneme in phonemes
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

    # Weighted next-phoneme counts for each prefix.
    # Word-final prefixes contribute no continuation count.
    next_counts = defaultdict(lambda: defaultdict(float))

    for phonemes, frequency in zip(tokenised_words, frequencies):
        if frequency <= 0:
            continue

        for position in range(len(phonemes) - 1):
            prefix = tuple(phonemes[:position + 1])
            next_phoneme = phonemes[position + 1]

            next_counts[prefix][next_phoneme] += frequency

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

            prefix = tuple(phonemes[:position + 1])
            counts = next_counts[prefix]
            total = builtins.sum(counts.values())

            if position < len(phonemes) - 1:
                observed_next = phonemes[position + 1]

                probability = (
                    counts.get(observed_next, 0.0) + alpha
                ) / (
                    total + alpha * V
                )

                continuation_count[current_phoneme] += 1

            else:
                # No continuation counts and no observed next phoneme:
                # additive smoothing gives a uniform distribution.
                probability = alpha / (alpha * V)

            surprisal = -log(probability, base)

            surprisal_sum[current_phoneme] += frequency * surprisal

    rows = []

    for phoneme in sorted(inventory):
        rows.append({
            "Phoneme": phoneme,
            "Surprisal": (
                surprisal_sum[phoneme] / weight_sum[phoneme]
                if weight_sum[phoneme] > 0
                else float("nan")
            ),
            "Weight": weight_sum[phoneme],
            "Occurrences": occurrence_count[phoneme],
            "Continuations": continuation_count[phoneme],
        })

    return pd.DataFrame(rows)

