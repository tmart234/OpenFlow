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
        'precipitation': rng.standard_normal(n_days),
        'SWE': rng.standard_normal(n_days),
        'soil_moisture': rng.standard_normal(n_days),
        'sm_observed': rng.integers(0, 2, n_days),
        'drought_index': rng.standard_normal(n_days),
        'reservoir_storage': rng.standard_normal(n_days),
        'reservoir_release': rng.standard_normal(n_days),
        'reservoir_observed': rng.integers(0, 2, n_days),
        'upstream_flow': rng.standard_normal(n_days),
        'upstream_observed': np.ones(n_days, dtype=int),
        'upstream_known_days': 7,
        **{c: rng.standard_normal(n_days) for c in windowing.BASIN_PRECIP_FEATURES},
        'basin_precip_observed': np.ones(n_days, dtype=int),
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
    # Same for soil moisture: SMAP is encoder-only, never in the decoder window.
    assert 'soil_moisture' not in windowing.DECODER_FEATURES
    assert 'soil_moisture' in windowing.ENCODER_FEATURES
    # The sm_observed indicator is also encoder-only.
    assert 'sm_observed' in windowing.ENCODER_FEATURES
    assert 'sm_observed' not in windowing.DECODER_FEATURES
    # USDM drought + USBR reservoir are encoder-only too -- none has a
    # skillful 14-day forecast available.
    assert 'drought_index' in windowing.ENCODER_FEATURES
    assert 'drought_index' not in windowing.DECODER_FEATURES
    assert 'reservoir_storage' in windowing.ENCODER_FEATURES
    assert 'reservoir_release' in windowing.ENCODER_FEATURES
    assert 'reservoir_observed' in windowing.ENCODER_FEATURES
    for c in ('reservoir_storage', 'reservoir_release', 'reservoir_observed'):
        assert c not in windowing.DECODER_FEATURES
    # Precipitation is the ONE auxiliary that IS available at forecast time
    # (Open-Meteo / GFS QPF ships a 14-day daily precipitation_sum), so it
    # appears in BOTH windows -- observed in the encoder, QPF in the decoder.
    assert 'precipitation' in windowing.ENCODER_FEATURES
    assert 'precipitation' in windowing.DECODER_FEATURES
    # Upstream flow is forecast-available (already observed upstream for the
    # first lag days, then the RFC forecast), so it is in both windows; the
    # observed indicator is encoder-only, the known-day mask decoder-only.
    assert 'upstream_flow' in windowing.ENCODER_FEATURES
    assert 'upstream_flow' in windowing.DECODER_FEATURES
    assert 'upstream_observed' in windowing.ENCODER_FEATURES
    assert 'upstream_observed' not in windowing.DECODER_FEATURES
    assert 'upstream_known' in windowing.DECODER_FEATURES
    assert 'upstream_known' not in windowing.ENCODER_FEATURES


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
                           'persistence_input',
                           'station_input', 'basin_input'}
    n = len(samples)
    assert inputs['encoder_input'].shape == (n, 60, len(windowing.ENCODER_FEATURES))
    assert inputs['decoder_input'].shape == (n, 14, len(windowing.DECODER_FEATURES))
    assert inputs['persistence_input'].shape == (n, len(windowing.TARGET_FEATURES))
    assert inputs['station_input'].shape == (n,)
    assert inputs['basin_input'].shape == (n,)
    assert targets.shape == (n, 14, len(windowing.TARGET_FEATURES))
    assert inputs['encoder_input'].dtype == np.float32
    assert inputs['persistence_input'].dtype == np.float32
    assert inputs['station_input'].dtype == np.int32


def test_persistence_anchor_is_the_last_encoder_day_target_values():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    # The persistence anchor is the model's free baseline -- it MUST match the
    # last encoder day's target-feature values exactly, or the residual
    # learning becomes biased.
    target_indices = [windowing.ENCODER_FEATURES.index(c) for c in windowing.TARGET_FEATURES]
    for s in samples[:5]:
        expected = s.encoder_X[-1, target_indices]
        np.testing.assert_array_equal(s.persistence_anchor, expected)


def test_build_windows_rejects_dataframe_missing_required_columns():
    df = _make_station_frame('USGS:A', 1, 1).drop(columns=['SWE'])
    with pytest.raises(ValueError):
        windowing.build_windows(df)


def test_build_windows_rejects_dataframe_missing_soil_moisture():
    df = _make_station_frame('USGS:A', 1, 1).drop(columns=['soil_moisture'])
    with pytest.raises(ValueError):
        windowing.build_windows(df)


def _decoder_col(name):
    return windowing.DECODER_FEATURES.index(name)


def test_upstream_decoder_window_uses_known_days_then_holds():
    future = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    observed = np.ones(5)
    values, known = windowing.upstream_decoder_window(future, observed, known_days=3,
                                                      last_value=0.5)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(known, [1, 1, 1, 0, 0])


def test_upstream_decoder_window_zero_known_days_holds_last_encoder_value():
    values, known = windowing.upstream_decoder_window(
        np.array([9.0, 9.0]), np.ones(2), known_days=0, last_value=0.25)
    np.testing.assert_array_equal(values, [0.25, 0.25])
    np.testing.assert_array_equal(known, [0, 0])


def test_upstream_decoder_window_unobserved_day_is_held_not_used():
    values, known = windowing.upstream_decoder_window(
        np.array([1.0, 7.0, 3.0]), np.array([1, 0, 1]), known_days=3, last_value=0.0)
    np.testing.assert_array_equal(values, [1.0, 1.0, 3.0])
    np.testing.assert_array_equal(known, [1, 0, 1])


def test_upstream_decoder_window_placeholder_when_last_encoder_unobserved():
    values, known = windowing.upstream_decoder_window(
        np.array([5.0]), np.array([0]), known_days=1, last_value=3.0, last_observed=0)
    np.testing.assert_array_equal(values, [0.0])
    np.testing.assert_array_equal(known, [0])


def test_windows_apply_upstream_known_horizon_per_station():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    df['upstream_known_days'] = 3
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    s = samples[0]
    up = df['upstream_flow'].to_numpy()
    dec_up = s.decoder_X[:, _decoder_col('upstream_flow')]
    dec_known = s.decoder_X[:, _decoder_col('upstream_known')]
    # First 3 decoder days carry the real routed values, then the 3rd is held.
    np.testing.assert_allclose(dec_up[:3], up[60:63])
    np.testing.assert_allclose(dec_up[3:], up[62])
    np.testing.assert_array_equal(dec_known, [1, 1, 1] + [0] * 11)


def test_windows_without_upstream_column_default_to_no_known_days():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120).drop(columns=['upstream_known_days'])
    samples = windowing.build_windows(df, encoder_days=60, decoder_days=14)
    s = samples[0]
    assert (s.decoder_X[:, _decoder_col('upstream_known')] == 0).all()
    # Held at the last encoder day's value for the whole horizon.
    last_enc = s.encoder_X[-1, windowing.ENCODER_FEATURES.index('upstream_flow')]
    assert (s.decoder_X[:, _decoder_col('upstream_flow')] == last_enc).all()


def test_build_windows_rejects_dataframe_missing_upstream_flow():
    df = _make_station_frame('USGS:A', 1, 1).drop(columns=['upstream_flow'])
    with pytest.raises(ValueError):
        windowing.build_windows(df)


def test_basin_precip_features_in_both_windows_with_decoder_mask():
    for c in windowing.BASIN_PRECIP_FEATURES:
        assert c in windowing.ENCODER_FEATURES and c in windowing.DECODER_FEATURES
    assert 'basin_precip_observed' in windowing.ENCODER_FEATURES
    assert 'basin_precip_observed' not in windowing.DECODER_FEATURES
    assert 'basin_precip_known' in windowing.DECODER_FEATURES
    assert 'basin_precip_known' in windowing.WINDOW_BUILT_DECODER_FEATURES


def test_basin_precip_decoder_window_masks_beyond_qpf_horizon():
    vals = np.arange(10 * 6, dtype=float).reshape(10, 6) + 1
    obs = np.ones(10)
    obs[2] = 0
    out, known = windowing.basin_precip_decoder_window(vals, obs, known_days=7)
    np.testing.assert_array_equal(known, [1, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(out[0], vals[0])
    assert (out[2] == 0).all() and (out[7:] == 0).all()


def test_windows_apply_basin_precip_mask():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    s = windowing.build_windows(df, encoder_days=60, decoder_days=14)[0]
    cols = [windowing.DECODER_FEATURES.index(c) for c in windowing.BASIN_PRECIP_FEATURES]
    known = s.decoder_X[:, windowing.DECODER_FEATURES.index('basin_precip_known')]
    np.testing.assert_array_equal(known, [1] * windowing.BASIN_PRECIP_KNOWN_DAYS
                                  + [0] * (14 - windowing.BASIN_PRECIP_KNOWN_DAYS))
    rows = df[windowing.BASIN_PRECIP_FEATURES].to_numpy()[60:60 + windowing.BASIN_PRECIP_KNOWN_DAYS]
    np.testing.assert_allclose(s.decoder_X[:windowing.BASIN_PRECIP_KNOWN_DAYS][:, cols], rows)
    assert (s.decoder_X[windowing.BASIN_PRECIP_KNOWN_DAYS:][:, cols] == 0).all()


def test_basin_precip_known_days_matches_qpf_horizon():
    from data import get_basin_precip
    assert windowing.BASIN_PRECIP_KNOWN_DAYS == get_basin_precip.QPF_DAYS
    assert windowing.BASIN_PRECIP_FEATURES == get_basin_precip.FEATURE_COLUMNS


def test_upstream_change_is_relative_to_last_encoder_day_on_known_days():
    out = windowing.upstream_change(np.array([2.0, 3.0, 3.0]), np.array([1, 1, 0]), last_value=1.5)
    np.testing.assert_allclose(out, [0.5, 1.5, 0.0])
    assert (windowing.upstream_change(np.ones(3), np.ones(3), 1.0, last_observed=0) == 0).all()


def test_windows_carry_upstream_change():
    df = _make_station_frame('USGS:A', 1, 1, n_days=120)
    df['upstream_known_days'] = 3
    s = windowing.build_windows(df, encoder_days=60, decoder_days=14)[0]
    up = df['upstream_flow'].to_numpy()
    chg = s.decoder_X[:, windowing.DECODER_FEATURES.index('upstream_change')]
    np.testing.assert_allclose(chg[:3], up[60:63] - up[59])
    assert (chg[3:] == 0).all()
