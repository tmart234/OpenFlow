"""Offline tests for Phase 2 archived weather forecasts (GEFS v12 + Open-Meteo Previous Runs) and decoder injection."""

import numpy as np
import pandas as pd
import pytest

import windowing
from data import get_gefs_reforecast
from test_windowing import _make_station_frame


def test_parse_open_meteo_previous_runs_produces_issue_date_lead_rows():
    # Two valid days (2025-05-05 and 2025-05-06), 24 hours each, leads 1 and 2.
    times = list(pd.date_range('2025-05-05 00:00', periods=48, freq='1h').strftime('%Y-%m-%dT%H:%M'))
    payload = {
        'hourly': {
            'time': times,
            'precipitation_previous_day1': [0.5] * 24 + [0.0] * 24,
            'precipitation_previous_day2': [0.25] * 24 + [0.1] * 24,
            'temperature_2m_previous_day1': [10.0] * 24 + [12.0] * 24,
            'temperature_2m_previous_day2': [8.0] * 24 + [14.0] * 24,
            'shortwave_radiation_previous_day1': [250.0] * 48,
            'shortwave_radiation_previous_day2': [200.0] * 48,
        }
    }
    df = get_gefs_reforecast.parse_open_meteo_previous_runs('USGS:A', payload, leads=(1, 2))
    assert list(df.columns) == get_gefs_reforecast.REFORECAST_COLUMNS
    # Valid 2025-05-05 at lead 1 -> issue_date 2025-05-04.
    r1 = df[(df['issue_date'] == pd.Timestamp('2025-05-04')) & (df['lead_day'] == 1)].iloc[0]
    assert r1['fc_precipitation'] == pytest.approx(12.0)
    assert r1['fc_TMAX'] == pytest.approx(100.0)  # tenths of deg C
    assert r1['fc_srad'] == pytest.approx(250.0 * 0.0864, rel=1e-3)
    # Cross-lead revision spread between day1 (12mm) and day2 (6mm) is nonzero.
    assert r1['precip_ens_std'] > 0.0
    assert r1['precip_ens_p90'] >= r1['fc_precipitation'] - 1.0


def test_gefs_v12_url_idx_parser_and_member_aggregation():
    url = get_gefs_reforecast.gefs_v12_grib_url('2015-05-10', member='p02', var_file='apcp_sfc')
    assert url.endswith('/2015/2015051000/p02/Days:1-10/apcp_sfc_2015051000_p02.grib2')

    idx_text = (
        "1:0:d=2015051000:APCP:surface:0-3 hour acc fcst:\n"
        "2:12000:d=2015051000:APCP:surface:3-6 hour acc fcst:\n"
        "3:25000:d=2015051000:TMAX:2 m above ground:0-3 hour max fcst:\n"
    )
    entries = get_gefs_reforecast.parse_grib_idx(idx_text)
    assert len(entries) == 3
    assert entries[0]['byte_start'] == 0 and entries[0]['byte_end'] == 11999
    assert entries[1]['var'] == 'APCP' and entries[2]['byte_end'] is None

    members = {
        'c00': pd.DataFrame({'lead_day': [1, 2], 'TMIN': [10.0, 20.0], 'TMAX': [100.0, 120.0],
                             'precipitation': [2.0, 0.0], 'srad': [20.0, 22.0]}),
        'p01': pd.DataFrame({'lead_day': [1, 2], 'TMIN': [20.0, 20.0], 'TMAX': [140.0, 120.0],
                             'precipitation': [10.0, 0.0], 'srad': [18.0, 22.0]}),
    }
    agg = get_gefs_reforecast.aggregate_gefs_members('USGS:A', '2015-05-10', members)
    d1 = agg[agg['lead_day'] == 1].iloc[0]
    assert d1['fc_precipitation'] == pytest.approx(6.0)
    assert d1['precip_ens_std'] == pytest.approx(4.0)
    assert d1['precip_ens_p90'] > 9.0
    assert d1['temp_ens_std'] == pytest.approx(20.0)


def test_reforecast_cache_round_trip_and_windowing_injection(tmp_path):
    df_station = _make_station_frame('USGS:A', 1, 1, n_days=90, start='2024-05-01')
    issue_date = pd.Timestamp('2024-05-01') + pd.Timedelta(days=windowing.ENCODER_DAYS - 1)
    fc = pd.DataFrame([{
        'site_id': 'USGS:A', 'issue_date': issue_date, 'lead_day': 1,
        'fc_TMIN': 25.0, 'fc_TMAX': 150.0, 'fc_precipitation': 10.0, 'fc_srad': 24.0,
        'precip_ens_std': 3.0, 'precip_ens_p90': 14.0, 'temp_ens_std': 15.0,
    }, {
        'site_id': 'USGS:A', 'issue_date': issue_date, 'lead_day': 2,
        'fc_TMIN': 30.0, 'fc_TMAX': 160.0, 'fc_precipitation': 0.0, 'fc_srad': 26.0,
        'precip_ens_std': 0.5, 'precip_ens_p90': 1.0, 'temp_ens_std': 10.0,
    }])
    get_gefs_reforecast.save_reforecast_table(fc, 'USGS:A', str(tmp_path))
    loaded = get_gefs_reforecast.load_reforecast_table(str(tmp_path))
    assert len(loaded) == 2

    scalers = {
        'TMIN': {'mean': 0.0, 'scale': 10.0, 'transform': 'identity'},
        'TMAX': {'mean': 0.0, 'scale': 10.0, 'transform': 'identity'},
        'precipitation': {'mean': 0.0, 'scale': 1.0, 'transform': 'log1p'},
        'srad': {'mean': 0.0, 'scale': 10.0, 'transform': 'identity'},
        'Min Flow': {'mean': 0.0, 'scale': 1.0, 'transform': 'log1p'},
        'Max Flow': {'mean': 0.0, 'scale': 1.0, 'transform': 'log1p'},
        'upstream_flow': {'mean': 0.0, 'scale': 1.0, 'transform': 'log1p'},
    }
    samples = windowing.build_windows(df_station, scalers=scalers, forecast_table=loaded)
    s0 = samples[0]  # issue_date matches anchor_date + ENCODER_DAYS - 1
    D = windowing.DECODER_FEATURES
    assert s0.decoder_X[0, D.index('TMIN')] == pytest.approx(2.5)
    assert s0.decoder_X[0, D.index('TMAX')] == pytest.approx(15.0)
    assert s0.decoder_X[0, D.index('precipitation')] == pytest.approx(np.log1p(10.0))
    assert s0.decoder_X[0, D.index('forcing_is_forecast')] == 1.0
    assert s0.decoder_X[1, D.index('forcing_is_forecast')] == 1.0
    assert s0.decoder_X[2, D.index('forcing_is_forecast')] == 0.0
    assert s0.decoder_X[0, D.index('ens_spread_known')] == 1.0
    assert s0.decoder_X[2, D.index('ens_spread_known')] == 0.0
    assert s0.decoder_X[0, D.index('lead_day_norm')] == pytest.approx(1.0 / 14.0)
    assert s0.decoder_X[-1, D.index('lead_day_norm')] == pytest.approx(1.0)
