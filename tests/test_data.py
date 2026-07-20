"""Tests for phoneme_entropy.data (bundled sample loader)."""

import pandas as pd

from phoneme_entropy.data import load_sample, sample_path


def test_sample_path_exists():
    path = sample_path()
    assert path.endswith("celex_en_sample.csv")


def test_load_sample_has_expected_columns():
    df = load_sample()
    assert isinstance(df, pd.DataFrame)
    assert "Word" in df.columns
    assert "Frequency" in df.columns
    assert len(df) > 0
    # The leading unnamed index column should have been dropped.
    assert not df.columns[0].startswith("Unnamed")
