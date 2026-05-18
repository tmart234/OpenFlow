"""
Smoke tests for the CBRFC + S2F baseline modules.

Both modules expose a `baseline_predictions(test_samples)` hook that train.py
calls during evaluation. The CBRFC module fetches forecasts (live AHPS for
today, NWPS archive for historical anchors) -- when no LID is mapped for a
site, or the archive returns no forecast for an anchor date, the module must
return None / empty cleanly. The S2F module remains a stub; same contract.
These tests pin that contract so the train.py integration stays safe.
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
    # The seeded cbrfc_lid_map.json carries no site mappings by default --
    # the comment-only file means every fetch returns empty and the stacked
    # prediction is None (skip cleanly).
    pred = get_cbrfc.baseline_predictions([_sample('USGS:UNMAPPED')])
    assert pred is None


def test_cbrfc_baseline_returns_none_for_empty_test_set():
    assert get_cbrfc.baseline_predictions([]) is None


def test_cbrfc_fetch_current_skips_when_no_lid():
    df = get_cbrfc.fetch_current('USGS:UNMAPPED')
    assert df.empty
    assert list(df.columns) == ['Date', 'cbrfc_flow']


def test_cbrfc_fetch_historical_skips_when_no_lid():
    # Anchor in the past, no LID mapping -> empty (no NWPS request issued).
    df = get_cbrfc.fetch('USGS:UNMAPPED', '2024-01-01')
    assert df.empty


def test_s2f_baseline_returns_none_until_archive_wired():
    assert get_s2f.baseline_predictions([_sample()]) is None
    assert get_s2f.baseline_predictions([]) is None


def test_s2f_fetch_returns_empty_until_archive_wired():
    df = get_s2f.fetch('USGS:09163500', '2024-01-01')
    assert df.empty
    assert list(df.columns) == ['Date', 's2f_volume_kaf']
