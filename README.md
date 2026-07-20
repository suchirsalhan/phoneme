# phoneme-entropy

Phoneme-level entropy, informativity, and maximum-entropy tools. A standalone
library that bundles the entropy estimators from
[`entropy-estimators`](https://github.com/fermosc24/entropy-estimators) (MIT,
© 2025 fermosc24) and adds phonology-oriented analysis utilities.

## Install

```bash
pip install phoneme-entropy
# or, from source:
pip install .
```

Optional extras: `pip install "phoneme-entropy[plot]"` (matplotlib/seaborn for
`plot_ranks`), `pip install "phoneme-entropy[data]"` (Hugging Face `datasets`).

## Usage

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

See [`examples/quickstart.py`](examples/quickstart.py) and
[`examples/usage_example.ipynb`](examples/usage_example.ipynb) for a full
walkthrough.

### Data

A small CELEX-derived sample ships with the package (`load_sample()`). The full
sample is mirrored as a **gated, permissions-only** Hugging Face dataset
[`suchirsalhan/celex-en-phoneme-sample`](https://huggingface.co/datasets/suchirsalhan/celex-en-phoneme-sample)
— access is granted to holders of a valid CELEX/LDC license
([LDC96L14](https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html)):

```python
from phoneme_entropy.data import load_hf_dataset
ds = load_hf_dataset(split="train")   # after your access request is approved
```

## API

| Import | Purpose |
| --- | --- |
| `Entropy`, `FreqShrink`, `JS_KullbackLeibler`, `JS_JensenShannon` | Entropy / divergence estimators (MLE, JSE, Chao-Shen, CWJ, NBRS, NSB), vendored. |
| `estimate_alpha_entropy`, `dirichlet_entropy`, `plot_ranks` | Symmetric-Dirichlet inference and rank/probability curves. |
| `segment_informativity`, `phoneme_prefix_entropy` | Segmental informativity over phoneme strings. |
| `compute_maxent_from_matrix` | Maximum-entropy fit to feature expectations. |
| `phoneme_entropy.data.load_sample`, `load_hf_dataset` | Data loaders. |

## License

MIT — see [`LICENSE`](LICENSE). Bundles third-party code; see
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES). Please cite via
[`CITATION.cff`](CITATION.cff).
