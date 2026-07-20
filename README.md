# phoneme-entropy

**Phoneme-level entropy, informativity, and maximum-entropy tools — a reproducible
extension of [`entropy-estimators`](https://github.com/fermosc24/entropy-estimators).**

This library packages the analysis code behind our camera-ready submission. It
builds *on top of* Fermín Moscoso del Prado Martín's
[`entropy-estimators`](https://github.com/fermosc24/entropy-estimators) (which
provides the underlying entropy estimators: MLE, James–Stein shrinkage,
Chao–Shen, Chao–Wang–Jost, NSB, …) and adds phonology-oriented tooling:

| Module | Public functions | What it does |
| --- | --- | --- |
| `phoneme_entropy.dirichlet` | `estimate_alpha_entropy`, `dirichlet_entropy`, `beta_order_stat_pdf`, `numerical_beta_order_stat_moments`, `plot_ranks` | Estimate a symmetric-Dirichlet concentration `alpha` from an entropy estimate; predicted rank/probability curves via Beta order statistics. |
| `phoneme_entropy.segmentation` | `phoneme_prefix_entropy`, `segment_informativity` | Frequency-weighted phoneme prefix entropy and next-phoneme surprisal over words encoded as space-separated phonemes. |
| `phoneme_entropy.maxent` | `compute_maxent_from_matrix` | Fit a maximum-entropy distribution to target feature expectations via the convex dual. |
| `phoneme_entropy.data` | `load_sample`, `load_hf_dataset` | Load the bundled CELEX sample, or (once released) the private Hugging Face dataset. |

> The original research scripts are preserved **verbatim** under [`original/`](original/)
> for provenance. The packaged modules under `src/phoneme_entropy/` are the same
> numerics with documentation, comments, tests, and a clean import path added.

---

## Installation

Requires Python ≥ 3.9. The package depends on `entropy-estimators`, currently
distributed via GitHub, so installation pulls it directly:

```bash
# From a clone of this repo (editable install for development)
pip install -e ".[dev]"

# Or a plain install
pip install .
```

Optional extras:

```bash
pip install ".[plot]"   # matplotlib + seaborn, for plot_ranks / the notebook
pip install ".[data]"   # datasets + huggingface_hub, for the private HF dataset
```

If `pip` cannot resolve `entropy-estimators` from the direct git reference in
your environment, install it explicitly first:

```bash
pip install "git+https://github.com/fermosc24/entropy-estimators.git"
```

---

## Quick start

```python
import numpy as np
import pandas as pd
from collections import defaultdict

from phoneme_entropy import (
    estimate_alpha_entropy,
    segment_informativity,
    phoneme_prefix_entropy,
    compute_maxent_from_matrix,
)
from phoneme_entropy.data import load_sample

# 1. Load the bundled CELEX English sample (Word column = space-separated phonemes)
sample = load_sample()

# 2. Phoneme unigram frequencies
phonfreqs = defaultdict(int)
for word in sample["Word"]:
    for p in word.split():
        phonfreqs[p] += 1
phonfreqs = pd.DataFrame(phonfreqs.items(), columns=["Phoneme", "Frequency"])

# 3. Symmetric-Dirichlet alpha implied by the phoneme frequency distribution
alpha, se = estimate_alpha_entropy(phonfreqs["Frequency"], n_boot=0)

# 4. Segmental informativity (surprisal) and prefix entropy, per phoneme
surprisal = segment_informativity(sample["Word"])
prefix_H = phoneme_prefix_entropy(sample["Word"])

# 5. Maximum-entropy fit to (surprisal, entropy) expectations
merged = phonfreqs.merge(surprisal, on="Phoneme").merge(prefix_H, on="Phoneme")
p = merged["Frequency"] / merged["Frequency"].sum()
targets = np.array([np.sum(p * merged["Surprisal"]), np.sum(p * merged["Entropy"])])
F = merged[["Surprisal", "Entropy"]].to_numpy()
probs, lambdas = compute_maxent_from_matrix(F, targets)
```

A full, narrated walkthrough (the original tutorial, updated for the packaged
API) lives in [`examples/usage_example.ipynb`](examples/usage_example.ipynb),
with a script version in [`examples/quickstart.py`](examples/quickstart.py).

---

## Data

* **Bundled sample.** `phoneme_entropy.data.load_sample()` returns a small
  CELEX-derived English lexicon shipped inside the package
  (`src/phoneme_entropy/data/celex_en_sample.csv`) — enough to run every example
  and test.
* **Full dataset (private).** The complete dataset used in the paper is hosted in
  a private Hugging Face dataset repo. Once released, configure its id and load
  it via:

  ```python
  from phoneme_entropy.data import load_hf_dataset
  ds = load_hf_dataset(dataset_id="<org>/<dataset>", split="train", token="hf_...")
  ```

  or set `PHONEME_ENTROPY_HF_DATASET` in the environment. See
  `phoneme_entropy/data.py` for details.

---

## Reproducibility

* All public functions are **deterministic** given their inputs, except the
  optional bootstrap in `estimate_alpha_entropy` (multinomial resampling). Pass
  `n_boot=0` for a deterministic point estimate, or seed `numpy` for reproducible
  standard errors.
* Numerical behaviour matches the original scripts exactly (see `original/`).
* Pinned, tested dependency ranges are declared in `pyproject.toml`.

Run the tests:

```bash
pip install -e ".[dev]"
pytest
```

---

## Citation

If you use this library, please cite both this work and the upstream
`entropy-estimators` project. See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). Builds on `entropy-estimators` (MIT,
© 2025 Fermín Moscoso del Prado Martín).
