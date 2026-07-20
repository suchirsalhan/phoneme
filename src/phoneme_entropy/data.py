"""Dataset loaders for the bundled sample and the gated Hugging Face dataset.

Two data sources are supported:

1. **Bundled CELEX sample** — ``celex_en_sample.csv`` ships inside the package
   for quick smoke tests, examples, and the tutorial notebook. Load it with
   :func:`load_sample`.

2. **Gated Hugging Face dataset** — the full CELEX-derived sample lives in a
   gated (permissions-only) HF dataset (id set via :data:`HF_DATASET_ID` or the
   ``PHONEME_ENTROPY_HF_DATASET`` environment variable). Load it with
   :func:`load_hf_dataset`, which requires the optional ``datasets`` dependency
   and an approved HF access request (CELEX/LDC96L14 license holders only).

The CSV is expected to contain at least a ``Word`` column (space-separated
phonemes) and a ``Frequency`` column, matching the API of
:mod:`phoneme_entropy.segmentation`.
"""

from __future__ import annotations

import os
from importlib import resources

import pandas as pd

__all__ = ["load_sample", "load_hf_dataset", "HF_DATASET_ID", "sample_path"]

# Gated (permissions-only) Hugging Face dataset holding the full CELEX-derived
# sample. Access is granted manually to users who hold a valid CELEX/LDC license
# (LDC96L14). Override at runtime via the PHONEME_ENTROPY_HF_DATASET env var.
HF_DATASET_ID = os.environ.get(
    "PHONEME_ENTROPY_HF_DATASET", "suchirsalhan/celex-en-phoneme-sample"
)

_SAMPLE_FILENAME = "celex_en_sample.csv"


def sample_path() -> str:
    """Return the absolute path to the bundled CELEX sample CSV.

    Uses :mod:`importlib.resources` so it works whether the package is installed
    as a wheel, in editable mode, or run from a source checkout.
    """
    return str(resources.files("phoneme_entropy").joinpath("_data", _SAMPLE_FILENAME))


def load_sample(sep: str = ",") -> pd.DataFrame:
    """Load the bundled CELEX English sample as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    sep : str, default ","
        Field separator forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The sample lexicon. The first (unnamed index) column present in the raw
        file is dropped, mirroring the original tutorial's
        ``pd.read_csv(...).iloc[:, 1:]``.
    """
    with resources.as_file(
        resources.files("phoneme_entropy").joinpath("_data", _SAMPLE_FILENAME)
    ) as path:
        df = pd.read_csv(path, sep=sep)
    # The raw CSV carries a leading unnamed index column; drop it for parity with
    # the original notebook.
    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]
    return df


def load_hf_dataset(
    dataset_id: str | None = None,
    split: str | None = None,
    token: str | bool | None = None,
    **kwargs,
):
    """Load the gated Hugging Face dataset (full CELEX-derived sample).

    Requires the optional ``datasets`` dependency (``pip install
    "phoneme-entropy[data]"``) and, for this gated repo, an approved access
    request plus a Hugging Face token — pass ``token=...`` or run
    ``huggingface-cli login`` beforehand.

    Parameters
    ----------
    dataset_id : str, optional
        HF dataset repo id. Defaults to :data:`HF_DATASET_ID` (which itself
        honours the ``PHONEME_ENTROPY_HF_DATASET`` environment variable).
    split : str, optional
        Which split to load (e.g. ``"train"``). If ``None``, the full
        ``DatasetDict`` is returned.
    token : str | bool, optional
        HF access token for private datasets. If ``None``, the token cached by
        ``huggingface-cli login`` is used.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`datasets.load_dataset`.

    Returns
    -------
    datasets.Dataset or datasets.DatasetDict
        The loaded dataset.

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "load_hf_dataset requires the 'datasets' package. Install the optional "
            "data extra with:  pip install 'phoneme-entropy[data]'"
        ) from exc

    dataset_id = dataset_id or HF_DATASET_ID
    if dataset_id.startswith("REPLACE_ME"):
        raise ValueError(
            "The Hugging Face dataset id has not been configured yet. Set "
            "phoneme_entropy.data.HF_DATASET_ID, pass dataset_id=..., or export "
            "PHONEME_ENTROPY_HF_DATASET once the private dataset is released."
        )

    return load_dataset(dataset_id, split=split, token=token, **kwargs)
