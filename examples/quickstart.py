#!/usr/bin/env python3
"""End-to-end quickstart for phoneme-entropy.

Reproduces the workflow from the original tutorial notebook using the packaged
API and the bundled CELEX sample. Run it after installing the package:

    pip install -e ".[dev]"
    python examples/quickstart.py

It prints intermediate results and (if matplotlib is available) draws the
predicted-vs-observed rank plot and the max-ent fit.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from phoneme_entropy import (
    compute_maxent_from_matrix,
    estimate_alpha_entropy,
    phoneme_prefix_entropy,
    segment_informativity,
)
from phoneme_entropy.data import load_sample


def main() -> None:
    # 1. Load the bundled CELEX English sample. "Word" holds space-separated
    #    phonemes; "Frequency" holds corpus frequencies.
    sample = load_sample()
    print(f"Loaded {len(sample)} words. Columns: {list(sample.columns)}")

    # 2. Phoneme unigram frequencies.
    phonfreqs = defaultdict(int)
    for word in sample["Word"]:
        for phoneme in word.split():
            phonfreqs[phoneme] += 1
    phonfreqs = pd.DataFrame(phonfreqs.items(), columns=["Phoneme", "Frequency"])
    print(f"Inventory size (V): {len(phonfreqs)}")

    # 3. Symmetric-Dirichlet concentration implied by the phoneme distribution.
    #    n_boot=0 -> deterministic point estimate (no bootstrap standard error).
    alpha, se = estimate_alpha_entropy(phonfreqs["Frequency"], n_boot=0)
    print(f"Estimated Dirichlet alpha: {alpha:.4f}")

    # 4. Per-phoneme surprisal (next-phoneme informativity) and prefix entropy.
    surprisal = segment_informativity(sample["Word"])
    prefix_entropy = phoneme_prefix_entropy(sample["Word"])

    # 5. Merge, form frequency-weighted target expectations, fit max-ent.
    merged = phonfreqs.merge(surprisal, on="Phoneme").merge(prefix_entropy, on="Phoneme")
    p = merged["Frequency"] / merged["Frequency"].sum()
    targets = np.array(
        [np.sum(p * merged["Surprisal"]), np.sum(p * merged["Entropy"])]
    )
    F = merged[["Surprisal", "Entropy"]].to_numpy()
    probs, lambdas = compute_maxent_from_matrix(F, targets)
    print(f"Max-ent Lagrange multipliers (lambda): {lambdas}")

    # 6. Optional plots (only if matplotlib is installed).
    try:
        import matplotlib.pyplot as plt

        from phoneme_entropy.dirichlet import plot_ranks

        plot_ranks(phonfreqs["Frequency"], types="Reversed", loglog=False)
        plt.figure()
        plt.scatter(np.log(p), np.log(probs))
        plt.xlabel("(log) observed probability")
        plt.ylabel("(log) predicted probability")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping plots. Install with '.[plot]'.")


if __name__ == "__main__":
    main()
