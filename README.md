<div align="center">

# 🔤 phoneme-entropy

**Information-theoretic tools for phonology: entropy estimation, segmental informativity, and maximum-entropy modelling.**

[![PyPI version](https://img.shields.io/pypi/v/phoneme-entropy?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/phoneme-entropy/)
[![Python](https://img.shields.io/pypi/pyversions/phoneme-entropy?logo=python&logoColor=white)](https://pypi.org/project/phoneme-entropy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/suchirsalhan/phoneme/actions/workflows/ci.yml/badge.svg)](https://github.com/suchirsalhan/phoneme/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suchirsalhan/phoneme/blob/main/examples/colab_quickstart.ipynb)

</div>

`phoneme-entropy` is a small, reproducible Python library for information-theoretic
analysis of phoneme sequences. It computes per-phoneme **surprisal** and **lexical
information gain**, estimates a **symmetric-Dirichlet** concentration parameter from
an entropy estimate, and fits **maximum-entropy** distributions to feature
expectations. The underlying entropy/divergence estimators are vendored from
[`entropy-estimators`](https://github.com/fermosc24/entropy-estimators)
(F. Moscoso del Prado Martín; MIT), so the package is **self-contained** — it has no
version-control dependencies and installs directly from PyPI.

A small CELEX-derived English sample ships with the package, so every example below
runs with no data download.

---

## Installation

```bash
pip install phoneme-entropy
```

Requires Python ≥ 3.9. Runtime dependencies: `numpy`, `scipy`, `pandas`, `mpmath`.

Optional extras:

```bash
pip install "phoneme-entropy[plot]"   # matplotlib + seaborn (for plot_ranks and the notebooks)
pip install "phoneme-entropy[data]"   # datasets + huggingface_hub (for the gated dataset loader)
```

## Quickstart

```python
from collections import defaultdict
import numpy as np, pandas as pd

from phoneme_entropy import (
    estimate_alpha_entropy,
    segment_informativity,
    lexical_information_gain,
    compute_maxent_from_matrix,
)
from phoneme_entropy.data import load_sample

# Bundled CELEX sample: "Word" holds space-separated phonemes; "Frequency" is corpus frequency.
sample = load_sample()

# Per-phoneme next-phoneme surprisal and lexical information gain.
surprisal = segment_informativity(sample["Word"])      # cols: Phoneme, Surprisal, Weight, Occurrences, Continuations
info_gain = lexical_information_gain(sample["Word"])    # cols: Phoneme, Entropy, Weight, Prefixes

# Phoneme unigram frequencies -> symmetric-Dirichlet concentration implied by their entropy.
phon = defaultdict(int)
for w in sample["Word"]:
    for p in w.split():
        phon[p] += 1
alpha, se = estimate_alpha_entropy(list(phon.values()), n_boot=0)

# Maximum-entropy fit to frequency-weighted (surprisal, information-gain) targets.
freqs = pd.DataFrame(phon.items(), columns=["Phoneme", "Frequency"])
merged = freqs.merge(surprisal, on="Phoneme").merge(info_gain, on="Phoneme")
p = merged["Frequency"] / merged["Frequency"].sum()
targets = np.array([np.sum(p * merged["Surprisal"]), np.sum(p * merged["Entropy"])])
probs, lambdas = compute_maxent_from_matrix(merged[["Surprisal", "Entropy"]].to_numpy(), targets)
```

Runnable versions: [`examples/quickstart.py`](examples/quickstart.py),
[`examples/usage_example.ipynb`](examples/usage_example.ipynb), and a self-contained
[Colab notebook](https://colab.research.google.com/github/suchirsalhan/phoneme/blob/main/examples/colab_quickstart.ipynb).

## What it computes

**Entropy & divergence estimators** (vendored). Call `Entropy(counts, method=...)`
with any of:

| `method` | Estimator |
| --- | --- |
| `"MLE"` (default) | Maximum likelihood (plug-in) |
| `"JSE"` | James–Stein shrinkage |
| `"CAE"` | Chao–Shen (coverage-adjusted) |
| `"CWJ"` | Chao–Wang–Jost |
| `"NBRS"` | Nemenman–Bialek–de Ruyter van Steveninck |
| `"NSB"` | Nemenman–Shafee–Bialek |

Entropy is returned in bits by default (`base=2`). Divergences between two count
vectors are available as `JS_KullbackLeibler` and `JS_JensenShannon` (both on
James–Stein–smoothed probabilities).

**Segmental measures** over words encoded as space-separated phoneme strings:

- `segment_informativity(words, frequencies=None, base=2, alpha=0.5)` — the
  frequency-weighted mean **surprisal** of the phoneme following each phoneme, under
  additive (Lidstone) smoothing with no end-of-word symbol. Returns a `DataFrame`
  with columns `Phoneme, Surprisal, Weight, Occurrences, Continuations`.
- `lexical_information_gain(words, frequencies=None, weighted=True, smoothing="CWJ")` —
  the mean entropy of the set of words still compatible after each phoneme (i.e. the
  lexical uncertainty resolved as the word unfolds). Returns columns
  `Phoneme, Entropy, Weight, Prefixes`. *(Also available under its historical name
  `phoneme_prefix_entropy`, kept as an alias for backwards compatibility.)*

**Symmetric-Dirichlet inference**:

- `estimate_alpha_entropy(counts, k=None, n_boot=500, method="CWJ", x0=1)` — recover
  the concentration parameter `α` whose expected symmetric-Dirichlet entropy matches
  the estimated entropy of `counts`. Returns `(alpha, se)`; `n_boot=0` gives a
  deterministic point estimate with no bootstrap standard error.
- `dirichlet_entropy(alpha, V)`, `plot_ranks(...)`, and the Beta order-statistic
  helpers `beta_order_stat_pdf` / `numerical_beta_order_stat_moments`.

**Maximum entropy**:

- `compute_maxent_from_matrix(F, target_expectations, tol=1e-6)` — fit the
  maximum-entropy distribution over the rows of feature matrix `F` whose feature
  expectations match `target_expectations`, via the convex dual. Returns
  `(probabilities, lagrange_multipliers)`.

## Data

`phoneme_entropy.data.load_sample()` returns the bundled CELEX-derived English sample
as a `pandas.DataFrame`.

The full sample is also mirrored as a **gated, permissions-only** Hugging Face dataset,
[`suchirsalhan/celex-en-phoneme-sample`](https://huggingface.co/datasets/suchirsalhan/celex-en-phoneme-sample).
CELEX is licensed by the Linguistic Data Consortium
([LDC96L14](https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html)); access is
granted manually to holders of a valid CELEX/LDC licence:

```python
from phoneme_entropy.data import load_hf_dataset
ds = load_hf_dataset(split="train")   # requires the [data] extra + an approved access request
```

## Reproducibility

- All public functions are **deterministic** given their inputs, with one exception:
  the optional parametric bootstrap in `estimate_alpha_entropy` (`n_boot > 0`) draws
  multinomial resamples — seed NumPy for reproducible standard errors, or use
  `n_boot=0` for a deterministic point estimate.
- The vendored estimator module (`phoneme_entropy/estimators.py`) is a **verbatim**
  copy of the upstream `entropy-estimators` source, with only an attribution header
  added; see [`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES).
- Dependency lower bounds are pinned in [`pyproject.toml`](pyproject.toml); continuous
  integration runs the test suite on Python 3.9–3.13.

## Repository layout

```
src/phoneme_entropy/
  estimators.py     # vendored entropy/divergence estimators (upstream, verbatim)
  segmentation.py   # segment_informativity, lexical_information_gain
  dirichlet.py      # estimate_alpha_entropy, plot_ranks, Beta order statistics
  maxent.py         # compute_maxent_from_matrix
  data.py           # load_sample, load_hf_dataset
  _data/            # bundled CELEX sample
tests/              # pytest suite
examples/           # quickstart.py, usage_example.ipynb, colab_quickstart.ipynb
```

## Development & testing

```bash
git clone https://github.com/suchirsalhan/phoneme
cd phoneme
pip install -e ".[dev]"
pytest
```

## Citing & license

Released under the [MIT License](LICENSE). The bundled estimators are MIT-licensed by
their original author — see [`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES). If you use
this library, please cite it (metadata in [`CITATION.cff`](CITATION.cff)) together with
the upstream [`entropy-estimators`](https://github.com/fermosc24/entropy-estimators)
project.
