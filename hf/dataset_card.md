---
license: other
license_name: celex-ldc96l14
license_link: https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html
pretty_name: CELEX English Phoneme Sample (gated)
language:
  - en
tags:
  - phonology
  - phonemes
  - celex
  - entropy
  - psycholinguistics
size_categories:
  - n<1K
extra_gated_prompt: >-
  This dataset is derived from the CELEX Lexical Database of English
  (LDC96L14), distributed under a restricted license by the Linguistic Data
  Consortium (LDC). See the CELEX license and documentation:
  https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html

  Access is granted only to users who already hold a valid CELEX / LDC license
  for LDC96L14. By requesting access you affirm the statements below. Requests
  are reviewed manually.
extra_gated_fields:
  Full name: text
  Institution / affiliation: text
  Intended use: text
  I hold a valid CELEX / LDC (LDC96L14) license: checkbox
  I will not redistribute this data or derivatives outside my licensed institution: checkbox
  I will cite CELEX (Baayen, Piepenbrock & Gulikers, 1995) in any resulting work: checkbox
extra_gated_button_content: Request access
---

# CELEX English Phoneme Sample (gated)

A small sample derived from the **CELEX Lexical Database of English**
([LDC96L14](https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html)),
used by the [`phoneme-entropy`](https://github.com/suchirsalhan/phoneme)
library for phoneme-level entropy and informativity analysis.

> **Gated / restricted.** CELEX is licensed by the Linguistic Data Consortium.
> This mirror is **permissions-only**: access is granted manually to users who
> already hold a valid CELEX/LDC license for LDC96L14.

## Contents

A single CSV, `celex_en_sample.csv`, with columns:

| Column | Description |
| --- | --- |
| `Word` | Word transcribed as **space-separated phonemes** (DISC/CELEX phoneme symbols). |
| `Frequency` | CELEX corpus frequency of the word. |

## Usage

```python
from phoneme_entropy.data import load_hf_dataset
ds = load_hf_dataset("suchirsalhan/celex-en-phoneme-sample", split="train", token="hf_...")
```

You must be logged in (`huggingface-cli login`) or pass a token, and your
access request must have been approved.

## License & citation

Derived from CELEX (LDC96L14), © Linguistic Data Consortium — see the
[CELEX license](https://catalog.ldc.upenn.edu/docs/LDC96L14/celex.readme.html).
Please cite:

> Baayen, R. H., Piepenbrock, R., & Gulikers, L. (1995). *The CELEX Lexical
> Database (CD-ROM)*. Linguistic Data Consortium, University of Pennsylvania.
