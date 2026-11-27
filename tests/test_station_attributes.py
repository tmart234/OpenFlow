"""Offline tests for Phase 5 static catchment attributes and specific discharge."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import evaluate
import normalize_data
import windowing
from data import build_station_attributes as bsa


def _sample_streamcat_row(area=1000.0, elev_ws=3000.0, elev_cat=1800.0, dam_stor=25000.0):
    return {
        'comid': 2885178,
        'wsareasqkm': area,
        'elevws': elev_ws,
        'elevcat': elev_cat,
        'precip9120ws': 650.0,
        'tmean9120ws': 2.8,
        'precip_minus_evtws': 20.5,
        'runoffws': 220.0,
        'bfiws': 68.0,
        'permws': 11.2,
        'clayws': 14.0,
        'sandws': 51.0,
        'rckdepws': 115.0,
        'pctconif2019ws': 55.0,
        'pctdecid2019ws': 3.0,
        'pctmxfst2019ws': 1.0,
        'pctshrb2019ws': 12.0,
        'wetindexws': 640.0,
        'damdensws': 0.02,
        'damnidstorws': dam_stor,
    }


def test_extract_station_record_populates_all_static_features():
    rec = bsa.extract_station_record(
        'USGS:06719505', 2885178, _sample_streamcat_row(), ungauged_frac=0.61)
    assert rec['area_km2'] == pytest.approx(1000.0)
    assert rec['regulated'] is True
    feats = rec['features']
    assert list(feats.keys()) == bsa.STATIC_FEATURE_NAMES
    assert feats['elev_relief_m'] == pytest.approx(1200.0)
    assert feats['forest_pct'] == pytest.approx(59.0)
    assert feats['ungauged_frac'] == pytest.approx(0.61)
    assert all(np.isfinite(v) for v in feats.values())


def test_static_scalers_and_unseen_station_vector():
    attrs = {
        'USGS:A': bsa.extract_station_record('USGS:A', 1, _sample_streamcat_row(area=100.0, elev_ws=2500.0), ungauged_frac=1.0),
        'USGS:B': bsa.extract_station_record('USGS:B', 2, _sample_streamcat_row(area=10000.0, elev_ws=3500.0), ungauged_frac=0.2),
        'USGS:UNSEEN': bsa.extract_station_record('USGS:UNSEEN', 3, _sample_streamcat_row(area=1000.0, elev_ws=3000.0), ungauged_frac=0.6),
    }
    # Fit strictly on training sites A & B.
    sc = bsa.fit_static_scalers(attrs, ['USGS:A', 'USGS:B'])
    vec_a = bsa.station_static_vector('USGS:A', attrs, sc)
    vec_b = bsa.station_static_vector('USGS:B', attrs, sc)
    vec_unseen = bsa.station_static_vector('USGS:UNSEEN', attrs, sc)
    vec_missing = bsa.station_static_vector('USGS:NOWHERE', attrs, sc)

    assert vec_a.shape == (len(bsa.STATIC_FEATURE_NAMES),)
    assert vec_a.dtype == np.float32
    # Mean of A and B is ~0 for every feature; UNSEEN (midpoint elev=3000) sits near 0.
    elev_i = bsa.STATIC_FEATURE_NAMES.index('elev_mean_m')
    assert vec_a[elev_i] == pytest.approx(-1.0, abs=1e-5)
    assert vec_b[elev_i] == pytest.approx(1.0, abs=1e-5)
    assert vec_unseen[elev_i] == pytest.approx(0.0, abs=1e-5)
    # Unknown site absent from attrs falls back cleanly to zeros.
    np.testing.assert_array_equal(vec_missing, np.zeros(len(bsa.STATIC_FEATURE_NAMES), dtype='float32'))


def test_checked_in_station_attributes_covers_all_core_sites():
    attrs = bsa.load_station_attributes()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, '.github', 'site_ids.txt')) as f:
        core_sites = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
    for sid in core_sites:
        assert sid in attrs, f"Core station {sid} missing from station_attributes.json"
        rec = attrs[sid]
        assert rec['area_km2'] > 0
        for feat in bsa.STATIC_FEATURE_NAMES:
            assert np.isfinite(rec['features'][feat]), f"{sid}.{feat} not finite"


def test_specific_discharge_round_trip_and_mass_balance():
    q_cfs = np.array([10.0, 250.0, 4200.0])
    area = 1027.2
    mm = normalize_data.cfs_to_specific(q_cfs, area)
    back = normalize_data.specific_to_cfs(mm, area)
    np.testing.assert_allclose(back, q_cfs, rtol=1e-9)

    # Check mass-balance invariance: dividing target and upstream by the SAME
    # target station area keeps Q0 + (U_dec - U_0) identical in cfs after inversion.
    df = pd.DataFrame({
        'Date': pd.date_range('2024-05-01', periods=4),
        'site_id': ['USGS:A'] * 4,
        'huc8': ['10190004'] * 4,
        'TMIN': [10.0] * 4,
        'TMAX': [100.0] * 4,
        'Min Flow': [100.0, 120.0, 200.0, 300.0],
        'Max Flow': [120.0, 140.0, 240.0, 340.0],
        'upstream_flow': [60.0, 80.0, 160.0, 260.0],
        'upstream_observed': [1, 1, 1, 1],
    })
    conv = normalize_data.convert_frame_to_specific_discharge(df, {'USGS:A': area})
    dq_target_cfs = normalize_data.specific_to_cfs(conv['Max Flow'].iloc[2] - conv['Max Flow'].iloc[0], area)
    dq_up_cfs = normalize_data.specific_to_cfs(conv['upstream_flow'].iloc[2] - conv['upstream_flow'].iloc[0], area)
    assert dq_target_cfs == pytest.approx(120.0, rel=1e-6)
    assert dq_up_cfs == pytest.approx(100.0, rel=1e-6)


def test_evaluate_forecasts_inverts_specific_discharge_back_to_cfs(tmp_path):
    """Specific-discharge normalized samples evaluate back to exact original cfs."""
    n = windowing.ENCODER_DAYS + windowing.DECODER_DAYS + 5
    dates = pd.date_range('2024-01-01', periods=n)
    area = 2500.0
    max_cfs = np.linspace(500.0, 1500.0, n)
    raw = pd.DataFrame({
        'Date': dates,
        'site_id': 'USGS:TEST',
        'huc8': '14010001',
        'TMIN': np.full(n, 0.0),
        'TMAX': np.full(n, 100.0),
        'Min Flow': max_cfs * 0.9,
        'Max Flow': max_cfs,
        'precipitation': np.zeros(n),
        'SWE': np.zeros(n),
        'soil_moisture': np.zeros(n),
        'sm_observed': np.zeros(n, dtype=int),
        'drought_index': np.zeros(n),
        'reservoir_storage': np.zeros(n),
        'reservoir_release': np.zeros(n),
        'reservoir_observed': np.zeros(n, dtype=int),
        'upstream_flow': np.zeros(n),
        'upstream_observed': np.zeros(n, dtype=int),
        'upstream_known_days': np.zeros(n, dtype=int),
        **{c: np.zeros(n) for c in windowing.BASIN_PRECIP_FEATURES},
        'basin_precip_observed': np.zeros(n, dtype=int),
    })
    areas = {'USGS:TEST': area}
    norm = normalize_data.normalize_data(raw, artifacts_dir=str(tmp_path), station_areas_km2=areas)
    scalers = json.loads((tmp_path / 'scalers.json').read_text())
    samples = windowing.build_windows(norm, scalers=scalers)
    assert len(samples) > 0
    _, y = windowing.stack(samples)
    fc = evaluate.forecasts(samples, y, scalers, station_areas_km2=areas)
    # Perfect prediction == obs == original raw Max Flow in cfs!
    expected_first = max_cfs[windowing.ENCODER_DAYS:windowing.ENCODER_DAYS + windowing.DECODER_DAYS]
    np.testing.assert_allclose(fc['obs'][0], expected_first, rtol=1e-4)
    np.testing.assert_allclose(fc['model'][0], expected_first, rtol=1e-4)
