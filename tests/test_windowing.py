import numpy as np
import pandas as pd
import pytest

import windowing


def _make_station_frame(site_id, station_idx, basin_idx, n_days=120, start='2022-01-01'):
    """A synthetic station frame in the post-normalization shape."""
    dates = pd.date_range(start, periods=n_days, freq='D')
    rng = np.random.default_rng(seed=station_idx)
    return pd.DataFrame({
        'Date': dates,
        'site_id': site_id,
        'station_idx': station_idx,
        'basin_idx': basin_idx,
        'huc8': f'1401000{station_idx}',
        'Min Flow': rng.standard_normal(n_days),
        'Max Flow': rng.standard_normal(n_days),
        'TMIN': rng.standard_normal(n_days),
        'TMAX': rng.standard_normal(n_days),
        'SWE': rng.standard_normal(n_days),
        'doy_sin': np.sin(2 * np.pi * np.arange(n_days) / 365),
        'doy_cos': np.cos(2 * np.pi * np.arange(n_days) / 365),
    })


def _make_multi_station_frame():
    return pd.concat(
        [_make_station_frame('USGS:A', 1, 1),
         _make_station_frame('USGS:B', 2, 2, start='2022-01-01')],
        ignore_index=True,
    )


def test_decoder_features_carry_no_flow_information():
    # This is THE invariant: forecast-window input never includes flow.
    assert 'Min Flow' not in windowing.DECODER_FEATURES
    assert 'Max Flow' not in windowing.DECODER_FEATURES
    # SWE is also a current-conditions feature -- no skillful 14-day forecast.
    assert 'SWE' not in windowing.DECODER_FEATURES


def test_build_windows_shapes_are_correct():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    # 120 - 74 + 1 = 47 windows for a fully-contiguous single station
    assert len(samples) == 47
    s = samples[0]
    assert s.encoder_X.shape == (60, len(windowing.ENCODER_FEATURES))
    assert s.decoder_X.shape == (14, len(windowing.DECODER_FEATURES))
    assert s.target_Y.shape == (14, len(windowing.TARGET_FEATURES))
    assert s.site_id == 'USGS:A'
    assert s.station_idx == 1


def test_windows_never_span_two_stations():
    df = _make_multi_station_frame()
    samples = windowing.build_windows(df, encoder_days=30, decoder_days=7)
    seen_pairs = {(s.site_id, s.station_idx) for s in samples}
    # Both stations contribute, but no sample shows up with a "merged" identity.
    assert seen_pairs == {('USGS:A', 1), ('USGS:B', 2)}
    # And every sample has consistent encoder rows -- impossible to verify
    # cross-station bleed directly without a marker, so we instead assert that
    # the total number of windows is exactly the sum of per-station windows
    # (no extras from spanning the boundary).
    n_per = len(windowing.build_windows(
        _make_station_frame('USGS:A', 1, 1), encoder_days=30, decoder_days=7))
    assert len(samples) == 2 * n_per


def test_windows_skip_over_date_gaps():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    # Punch a hole: drop a single day in the middle.
    df = df.drop(index=70).reset_index(drop=True)
    # Any window covering day 70 is no longer contiguous and must be rejected.
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    for s in samples:
        end = s.anchor_date + pd.Timedelta(days=73)
        assert not (s.anchor_date <= pd.Timestamp('2022-03-12') <= end)


def test_windows_skip_when_any_feature_is_nan():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    df.loc[10, 'TMIN'] = np.nan
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    # Any window touching day 10 must be excluded.
    for s in samples:
        end = s.anchor_date + pd.Timedelta(days=73)
        assert not (s.anchor_date <= pd.Timestamp('2022-01-11') <= end)


def test_chronological_split_is_temporally_disjoint_with_embargo():
    df = _make_multi_station_frame()
    samples = windowing.build_windows(df, encoder_days=30, decoder_days=7)
    splits = windowing.chronological_split(
        samples, val_frac=0.2, test_frac=0.2, embargo_days=7)

    # Per-station, train.max < val.min and val.max < test.min, by at least
    # the embargo width.
    for site in {'USGS:A', 'USGS:B'}:
        train_dates = [s.anchor_date for s in splits.train if s.site_id == site]
        val_dates = [s.anchor_date for s in splits.val if s.site_id == site]
        test_dates = [s.anchor_date for s in splits.test if s.site_id == site]
        if train_dates and val_dates:
            assert max(train_dates) < min(val_dates)
            assert (min(val_dates) - max(train_dates)).days >= 7
        if val_dates and test_dates:
            assert max(val_dates) < min(test_dates)
            assert (min(test_dates) - max(val_dates)).days >= 7


def test_stack_produces_keras_input_dict_and_target_tensor():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    inputs, targets = windowing.stack(samples)
    assert set(inputs) == {'encoder_input', 'decoder_input',
                           'station_input', 'basin_input'}
    n = len(samples)
    assert inputs['encoder_input'].shape == (n, 60, len(windowing.ENCODER_FEATURES))
    assert inputs['decoder_input'].shape == (n, 14, len(windowing.DECODER_FEATURES))
    assert inputs['station_input'].shape == (n,)
    assert inputs['basin_input'].shape == (n,)
    assert targets.shape == (n, 14, len(windowing.TARGET_FEATURES))
    assert inputs['encoder_input'].dtype == np.float32
    assert inputs['station_input'].dtype == np.int32


def test_build_windows_rejects_dataframe_missing_required_columns():
    df = _make_station_frame('USGS:A', 1, 1).drop(columns=['SWE'])
    with pytest.raises(ValueError):
        windowing.build_windows(df)
