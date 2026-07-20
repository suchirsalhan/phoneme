"""phoneme_entropy — phoneme-level entropy, informativity, and maximum-entropy tools.

This package is a small, reproducible research library that *extends*
`entropy-estimators <https://github.com/fermosc24/entropy-estimators>`_ (F. Moscoso
del Prado Martín) with utilities aimed at phonological / lexical analysis:

* **Symmetric-Dirichlet inference** (:mod:`phoneme_entropy.dirichlet`)
  Estimate the concentration parameter ``alpha`` of a symmetric Dirichlet prior
  directly from an entropy estimate, together with Beta order-statistic machinery
  used to draw predicted rank/probability curves.

* **Segmental informativity** (:mod:`phoneme_entropy.segmentation`)
  Frequency-weighted lexical information gain (via :func:`phoneme_prefix_entropy`)
  and next-phoneme surprisal, computed over words encoded as space-separated
  phoneme strings.

* **Maximum-entropy fitting** (:mod:`phoneme_entropy.maxent`)
  Fit a maximum-entropy distribution to a set of target feature expectations via
  the convex dual, using a precomputed feature matrix.

* **Data access** (:mod:`phoneme_entropy.data`)
  Loaders for the CELEX-derived sample bundled with the package and (once
  released) the private Hugging Face dataset used in the paper.

The heavy entropy-estimation numerics (MLE, James-Stein, Chao-Shen,
Chao-Wang-Jost, NSB, ...) are provided by :mod:`phoneme_entropy.estimators`,
which is vendored verbatim from
`entropy-estimators <https://github.com/fermosc24/entropy-estimators>`_
(MIT, (c) 2025 fermosc24) so this package is a standalone distribution with no
external VCS dependency. See ``THIRD_PARTY_LICENSES`` for attribution.

Reproducibility notes
---------------------
* Every public function is deterministic given its inputs, except the optional
  bootstrap in :func:`~phoneme_entropy.dirichlet.estimate_alpha_entropy`, which
  draws multinomial resamples. Set ``n_boot=0`` (the default in most call sites)
  for a fully deterministic point estimate, or seed ``numpy`` beforehand for
  reproducible standard errors.
* Function signatures and numerical behaviour are kept identical to the original
  research scripts (preserved verbatim under ``original/``); this package only
  adds documentation, packaging, and a clean import path.
"""

from importlib import metadata as _metadata

# Public API ---------------------------------------------------------------
# Re-exported here so users can simply ``from phoneme_entropy import <name>``.
from .dirichlet import (
    beta_order_stat_pdf,
    dirichlet_entropy,
    estimate_alpha_entropy,
    numerical_beta_order_stat_moments,
    plot_ranks,
)
from .estimators import (
    Entropy,
    FreqShrink,
    JS_JensenShannon,
    JS_KullbackLeibler,
    dict_to_ndarray,
    sample_frequencies,
)
from .maxent import compute_maxent_from_matrix
from .segmentation import (
    lexical_information_gain,
    phoneme_prefix_entropy,
    segment_informativity,
)

try:  # Populated from package metadata when installed (editable or wheel).
    __version__ = _metadata.version("phoneme-entropy")
except _metadata.PackageNotFoundError:  # Running from a source checkout.
    __version__ = "0.0.0.dev0"

__all__ = [
    # estimators (vendored from entropy-estimators)
    "Entropy",
    "FreqShrink",
    "JS_KullbackLeibler",
    "JS_JensenShannon",
    "dict_to_ndarray",
    "sample_frequencies",
    # dirichlet
    "estimate_alpha_entropy",
    "dirichlet_entropy",
    "beta_order_stat_pdf",
    "numerical_beta_order_stat_moments",
    "plot_ranks",
    # segmentation
    "phoneme_prefix_entropy",
    "lexical_information_gain",
    "segment_informativity",
    # maxent
    "compute_maxent_from_matrix",
    "__version__",
]
