"""
Smoke tests for the CBRFC + S2F baseline modules.

Both modules expose a `baseline_predictions(test_samples)` hook that train.py
calls during evaluation. Until the historical archive integrations are wired
in (see module docstrings), both must return None cleanly -- not crash, not
log noise, not pollute the persistence comparison. These tests pin that
contract so the train.py integration stays safe.
"""

import numpy as np
import pandas as pd
import pytest

from data import get_cbrfc, get_s2f
from windowing import WindowedSample


def _sample(site_id='USGS:09163500'):
    """A single WindowedSample-shaped object sufficient for baseline_predictions."""
    return WindowedSample(
        encoder_X=np.zeros((60, 9), dtype='float32'),
        decoder_X=np.zeros((14, 4), dtype='float32'),
        target_Y=np.zeros((14, 2), dtype='float32'),
        persistence_anchor=np.zeros(2, dtype='float32'),
        station_idx=1,
        basin_idx=1,
        site_id=site_id,
        anchor_date=pd.Timestamp('2024-01-01'),
    )


def test_cbrfc_baseline_returns_none_when_no_lid_mapped():
    # The default _AHPS_LID_TABLE is empty -> every site has no LID -> every
    # fetch returns empty -> the predictions stack is None (skip cleanly).
    pred = get_cbrfc.baseline_predictions([_sample()])
    assert pred is None


def test_cbrfc_baseline_returns_none_for_empty_test_set():
    assert get_cbrfc.baseline_predictions([]) is None


def test_cbrfc_fetch_current_skips_when_no_lid():
    df = get_cbrfc.fetch_current('USGS:UNKNOWN')
    assert df.empty
    assert list(df.columns) == ['Date', 'cbrfc_flow']


def test_cbrfc_fetch_historical_returns_empty_until_archive_wired():
    # Anchor in the past -> archive lookup is stubbed -> empty.
    df = get_cbrfc.fetch('USGS:09163500', '2024-01-01')
    assert df.empty


def test_s2f_baseline_returns_none_until_archive_wired():
    assert get_s2f.baseline_predictions([_sample()]) is None
    assert get_s2f.baseline_predictions([]) is None


def test_s2f_fetch_returns_empty_until_archive_wired():
    df = get_s2f.fetch('USGS:09163500', '2024-01-01')
    assert df.empty
    assert list(df.columns) == ['Date', 's2f_volume_kaf']
