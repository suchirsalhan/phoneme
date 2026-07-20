<div align="center">

# 🔤 phoneme-entropy

**Phoneme-level entropy, informativity, and maximum-entropy tools for Python.**

*A standalone, reproducible toolkit for information-theoretic phonology — built on top of
[`entropy-estimators`](https://github.com/fermosc24/entropy-estimators).*

[![PyPI version](https://img.shields.io/pypi/v/phoneme-entropy?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/phoneme-entropy/)
[![Python](https://img.shields.io/pypi/pyversions/phoneme-entropy?logo=python&logoColor=white)](https://pypi.org/project/phoneme-entropy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/suchirsalhan/phoneme/actions/workflows/ci.yml/badge.svg)](https://github.com/suchirsalhan/phoneme/actions/workflows/ci.yml)
[![Dataset on HF](https://img.shields.io/badge/🤗%20Dataset-gated-yellow)](https://huggingface.co/datasets/suchirsalhan/celex-en-phoneme-sample)

### ▶️ Try it in your browser — no install required

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the quickstart in Google Colab" height="28">](https://colab.research.google.com/github/suchirsalhan/phoneme/blob/main/examples/colab_quickstart.ipynb)

</div>

---

## ✨ What you get

- 📊 **Entropy & divergence estimators** — MLE, James–Stein, Chao–Shen, Chao–Wang–Jost, NBRS, and NSB, bundled and ready to call.
- 🔡 **Segmental informativity** — per-phoneme next-phoneme surprisal and prefix entropy over words encoded as phoneme strings.
- 🎲 **Symmetric-Dirichlet inference** — recover a concentration parameter `α` from an entropy estimate, with predicted rank/probability curves.
- 🧮 **Maximum-entropy fitting** — fit a distribution to target feature expectations via the convex dual.
- 📦 **Batteries included** — ships a CELEX sample so every example runs out of the box; reproducible, tested, and `pip`-installable.

## 🚀 Install

```bash
pip install phoneme-entropy
```

<details>
<summary>Optional extras</summary>

```bash
pip install "phoneme-entropy[plot]"   # matplotlib + seaborn for plot_ranks
pip install "phoneme-entropy[data]"   # Hugging Face datasets loader
```
</details>

## ⚡ Quickstart

```python
from collections import defaultdict
import numpy as np, pandas as pd
from phoneme_entropy import (
    Entropy, estimate_alpha_entropy,
    segment_informativity, phoneme_prefix_entropy,
    compute_maxent_from_matrix,
)
from phoneme_entropy.data import load_sample

sample = load_sample()                       # bundled CELEX sample (Word = space-separated phonemes)

# Per-phoneme next-phoneme surprisal and prefix entropy
surprisal = segment_informativity(sample["Word"])
prefix_H  = phoneme_prefix_entropy(sample["Word"])

# Symmetric-Dirichlet concentration implied by an entropy estimate
phon = defaultdict(int)
for w in sample["Word"]:
    for p in w.split(): phon[p] += 1
alpha, se = estimate_alpha_entropy(list(phon.values()), n_boot=0)
```

## 📓 Notebooks & examples

| Notebook | Open | What it does |
| --- | :---: | --- |
| **Colab quickstart** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="20">](https://colab.research.google.com/github/suchirsalhan/phoneme/blob/main/examples/colab_quickstart.ipynb) | Self-contained: `pip install`s from PyPI and runs the full pipeline on the bundled sample. |
| **Tutorial** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" height="20">](https://colab.research.google.com/github/suchirsalhan/phoneme/blob/main/examples/usage_example.ipynb) | The same walkthrough for local or Colab use. |
| [`quickstart.py`](examples/quickstart.py) | — | Script version of the walkthrough. |

## 🗂️ Data

A small CELEX-derived sample ships with the package — just call `load_sample()`.

The full sample is mirrored as a [![🤗 gated dataset](https://img.shields.io/badge/🤗%20suchirsalhan%2Fcelex--en--phoneme--sample-gated-yellow)](https://huggingface.co/datasets/suchirsalhan/celex-en-phoneme-sample)
(**permissions-only**) — access is granted to holders of a valid CELEX/LDC license
([LDC96L14](https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html)):

```python
from phoneme_entropy.data import load_hf_dataset
ds = load_hf_dataset(split="train")   # after your access request is approved
```

## 🧰 API at a glance

| Import | Purpose |
| --- | --- |
| `Entropy`, `FreqShrink`, `JS_KullbackLeibler`, `JS_JensenShannon` | Entropy / divergence estimators (MLE, JSE, Chao-Shen, CWJ, NBRS, NSB). |
| `estimate_alpha_entropy`, `dirichlet_entropy`, `plot_ranks` | Symmetric-Dirichlet inference and rank/probability curves. |
| `segment_informativity`, `phoneme_prefix_entropy` | Segmental informativity over phoneme strings. |
| `compute_maxent_from_matrix` | Maximum-entropy fit to feature expectations. |
| `phoneme_entropy.data.load_sample`, `load_hf_dataset` | Data loaders. |

## 📖 Citation & license

Released under the [MIT License](LICENSE). Bundles third-party code — see
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES). If you use this library, please cite it
(see [`CITATION.cff`](CITATION.cff)) along with the upstream
[`entropy-estimators`](https://github.com/fermosc24/entropy-estimators) project.

<div align="center">
<sub>Built with ❤️ for reproducible information-theoretic phonology.</sub>
</div>
