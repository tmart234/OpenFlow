"""
End-to-end smoke of train.main() on a synthetic two-station dataset (one
station with upstream gauges, one without): windowing -> model fit (1 epoch)
-> test evaluation -> hindcast charts -> persisted artifacts. Network-free;
combine_data.main is replaced by a synthetic frame pushed through the real
normalize_data. Needs tensorflow (runs in the CI model-smoke job).
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('tensorflow')

import combine_data  # noqa: E402
import normalize_data  # noqa: E402
import train  # noqa: E402


def _station(site_id, huc8, upstream, n_days=420, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    season = 1 + 0.8 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365)
    up = 400 * season + rng.normal(0, 20, n_days)
    flow = 200 + (0.9 * np.roll(up, 2) if upstream else 300 * season) + rng.normal(0, 10, n_days)
    df = pd.DataFrame({
        'Date': dates, 'site_id': site_id, 'huc8': huc8,
        'Min Flow': flow * 0.9, 'Max Flow': flow * 1.1,
        'TMIN': 20 + 10 * season, 'TMAX': 40 + 15 * season,
        'precipitation': rng.exponential(1.0, n_days),
        'SWE': np.maximum(0, 10 - 10 * season), 'soil_moisture': 0.2, 'sm_observed': 1,
        'drought_index': 0.0, 'reservoir_storage': 0.0, 'reservoir_release': 0.0,
        'reservoir_observed': 0,
        'upstream_flow': up if upstream else 0.0,
        'upstream_observed': 1 if upstream else 0,
        'upstream_known_days': 7 if upstream else 0,
        **{c: rng.exponential(2.0, n_days) for c in
           ('rain_local', 'rain_mid', 'rain_far', 'snow_local', 'snow_mid', 'snow_far')},
        'basin_precip_observed': 1,
    })
    return df


def test_train_main_end_to_end_with_upstream_station(tmp_path, monkeypatch):
    base = str(tmp_path)
    frame = pd.concat([_station('USGS:08276500', '13020101', True, seed=1),
                       _station('USGS:09114500', '14020002', False, seed=2)],
                      ignore_index=True)

    monkeypatch.setattr(combine_data, 'get_base_path', lambda: base)
    monkeypatch.setattr(combine_data, 'main',
                        lambda: normalize_data.normalize_data(frame, artifacts_dir=base))
    monkeypatch.setattr(train, 'EPOCHS', 1)
    for mod in ('get_cbrfc', 'get_s2f'):   # keep the external baselines offline
        try:
            m = __import__(f'data.{mod}', fromlist=[mod])
            monkeypatch.setattr(m, 'baseline_predictions', lambda samples: None)
        except Exception:
            pass

    train.main()

    assert os.path.exists(os.path.join(base, 'lstm_model.h5'))
    with open(os.path.join(base, 'training_config.json')) as f:
        config = json.load(f)
    assert 'upstream_flow' in config['encoder_features']
    assert 'upstream_known' in config['decoder_features']
    assert config['window_built_decoder_features'] == ['upstream_known', 'upstream_change',
                                                        'basin_precip_known']
    assert config['event_weight'] == train.EVENT_WEIGHT and config['linear_skip'] is True
    assert config['upstream_known_days'] == {'USGS:08276500': 7, 'USGS:09114500': 0}
    assert config['upstream_registry'] == 'upstream_gauges.json'
    assert config['basin_precip_known_days'] == 7
    assert 'rain_local' in config['encoder_features'] and 'basin_precip_known' in config['decoder_features']
    with open(os.path.join(base, 'basin_zones.json')) as f:
        assert 'USGS:08276500' in json.load(f)
    with open(os.path.join(base, 'upstream_gauges.json')) as f:
        assert 'USGS:08276500' in json.load(f)
    with open(os.path.join(base, 'scalers.json')) as f:
        assert json.load(f)['upstream_flow']['transform'] == 'log1p'
    charts = sorted(os.listdir(os.path.join(base, 'charts')))
    assert charts == ['USGS_08276500_hindcast.png', 'USGS_09114500_hindcast.png']


def test_event_weights_emphasise_changing_windows():
    inputs = {'persistence_input': np.zeros((3, 2), dtype='float32')}
    targets = np.zeros((3, 14, 2), dtype='float32')
    targets[2] += 1.0                      # one window that changes a lot
    w = train.event_weights(inputs, targets, event_weight=2.0)
    assert w.mean() == pytest.approx(1.0)
    assert w[2] == pytest.approx(3 * w[0])
