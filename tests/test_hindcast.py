import numpy as np
import pandas as pd
import pytest

import hindcast
import windowing

SCALERS = {'Min Flow': {'mean': 6.0, 'scale': 1.5, 'transform': 'log1p'},
           'Max Flow': {'mean': 6.5, 'scale': 1.5, 'transform': 'log1p'}}


def _sample(site, peak, anchor='2024-03-01'):
    enc = np.zeros((windowing.ENCODER_DAYS, len(windowing.ENCODER_FEATURES)))
    enc[:, windowing.ENCODER_FEATURES.index('upstream_observed')] = 1
    tgt = np.full((windowing.DECODER_DAYS, len(windowing.TARGET_FEATURES)), 0.1)
    tgt[5] = peak
    return windowing.WindowedSample(
        encoder_X=enc, decoder_X=np.zeros((windowing.DECODER_DAYS, len(windowing.DECODER_FEATURES))),
        target_Y=tgt, persistence_anchor=enc[-1, :2], station_idx=1, basin_idx=1,
        site_id=site, anchor_date=pd.Timestamp(anchor))


def test_invert_round_trips_log1p_scaling():
    raw = np.array([10.0, 1000.0])
    z = (np.log1p(raw) - 6.0) / 1.5
    np.testing.assert_allclose(hindcast.invert(z, SCALERS['Min Flow']), raw)
    assert hindcast.invert(np.array([1.0]), {'mean': 2, 'scale': 3})[0] == 5.0


def test_pick_peak_windows_chooses_highest_observed_per_station():
    samples = [_sample('A', 1.0), _sample('A', 2.5), _sample('B', 0.3), _sample('A', 2.0)]
    assert hindcast.pick_peak_windows(samples) == {'A': 1, 'B': 2}


def test_hindcast_series_dates_and_units():
    s = _sample('A', 1.0)
    pred = np.full((windowing.DECODER_DAYS, 2), 0.5)
    out = hindcast.hindcast_series(s, pred, SCALERS, history_days=30)
    assert len(out['history_dates']) == 30
    assert out['history_dates'][-1] == pd.Timestamp('2024-03-01') + pd.Timedelta(days=windowing.ENCODER_DAYS - 1)
    assert out['future_dates'][0] == out['history_dates'][-1] + pd.Timedelta(days=1)
    assert out['observed_max'][5] == pytest.approx(np.expm1(1.0 * 1.5 + 6.5))
    assert out['predicted_min'][0] == pytest.approx(np.expm1(0.5 * 1.5 + 6.0))


def test_write_hindcast_charts_one_png_per_station(tmp_path):
    pytest.importorskip('matplotlib')
    samples = [_sample('USGS:08276500', 2.0), _sample('USGS:09114500', 1.0)]
    preds = np.zeros((2, windowing.DECODER_DAYS, 2))
    paths = hindcast.write_hindcast_charts(samples, preds, SCALERS, str(tmp_path),
                                           names={'USGS:08276500': 'Lower Taos Box'})
    assert sorted(p.split('/')[-1] for p in paths) == [
        'USGS_08276500_hindcast.png', 'USGS_09114500_hindcast.png']
