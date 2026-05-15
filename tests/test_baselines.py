import numpy as np
import pandas as pd

import baselines


def _stations_with_steady_flow():
    # Two stations, each constant -- persistence MAE must be 0.
    dates = pd.date_range('2022-01-01', periods=30, freq='D')
    return pd.concat([
        pd.DataFrame({'Date': dates, 'site_id': 'A', 'Min Flow': 10.0, 'Max Flow': 12.0}),
        pd.DataFrame({'Date': dates, 'site_id': 'B', 'Min Flow': 50.0, 'Max Flow': 60.0}),
    ], ignore_index=True)


def test_persistence_is_zero_for_constant_flow():
    df = _stations_with_steady_flow()
    report = baselines.persistence_baseline(df, horizon=14)
    for col in ['Min Flow', 'Max Flow']:
        assert report[col]['overall'] == 0.0
        assert all(p == 0.0 for p in report[col]['per_horizon'])


def test_persistence_error_grows_with_horizon_for_a_linear_trend():
    # Flow = day index, so |flow[t+k] - flow[t]| = k. Per-horizon MAE = k.
    dates = pd.date_range('2022-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'site_id': 'A',
        'Min Flow': np.arange(30, dtype=float),
        'Max Flow': np.arange(30, dtype=float),
    })
    report = baselines.persistence_baseline(df, horizon=5)
    assert np.allclose(report['Min Flow']['per_horizon'], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_climatology_is_zero_when_flow_equals_day_of_year_mean():
    # Make each year's flow identical so per-DOY mean == actual.
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    flow = pd.to_datetime(dates).dayofyear.astype(float)
    df = pd.DataFrame({'Date': dates, 'site_id': 'A', 'Min Flow': flow, 'Max Flow': flow})
    report = baselines.climatology_baseline(df)
    assert report['Min Flow'] == 0.0
    assert report['Max Flow'] == 0.0


def test_evaluate_baselines_returns_structured_report():
    df = _stations_with_steady_flow()
    report = baselines.evaluate_baselines(df, horizon=3)
    assert set(report) == {'persistence', 'climatology', 'horizon', 'sites'}
    assert report['horizon'] == 3
    assert report['sites'] == ['A', 'B']
    assert report['persistence']['Min Flow']['overall'] == 0.0


def test_format_report_contains_expected_labels():
    df = _stations_with_steady_flow()
    text = baselines.format_report(baselines.evaluate_baselines(df, horizon=3))
    assert 'persistence' in text
    assert 'climatology' in text
    assert 'Min Flow' in text and 'Max Flow' in text
