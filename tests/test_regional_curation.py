"""Offline tests for Phase 5 regional gauge curation and leave-basin-out CV."""

import numpy as np
import pandas as pd
import pytest

import combine_data
import windowing
from data import curate_regional_sites as crs


def test_parse_rdb_skips_comments_and_type_row():
    text = (
        "# USGS comment header\n"
        "# another comment\n"
        "agency_cd\tsite_no\tstation_nm\tdrain_area_va\n"
        "5s\t15s\t50s\t8s\n"
        "USGS\t06036805\tFirehole River at Old Faithful\t39.5\n"
        "USGS\t09070500\tColorado River near Dotsero\t4394\n"
    )
    rows = crs.parse_rdb(text)
    assert len(rows) == 2
    assert rows[0]['site_no'] == '06036805'
    assert rows[1]['drain_area_va'] == '4394'


def test_filter_candidates_enforces_hydrologic_vetting_and_diversity():
    as_of = pd.Timestamp('2026-11-27')
    cands = [
        # Good WY gauge #1 (longest record in HUC A)
        {'site_id': 'USGS:1001', 'state': 'WY', 'name': 'Snake River near Moran, WY',
         'huc8': '17040101', 'area_km2': 2090.0, 'begin_date': '1990-01-01',
         'end_date': '2026-11-25', 'count_nu': 13400},
        # Good WY gauge #2 (same HUC A, allowed since max_per_huc8=2)
        {'site_id': 'USGS:1002', 'state': 'WY', 'name': 'Snake River above Reservoir, WY',
         'huc8': '17040101', 'area_km2': 8900.0, 'begin_date': '1995-01-01',
         'end_date': '2026-11-25', 'count_nu': 11500},
        # 3rd gauge in same HUC A -> rejected by HUC8 diversity cap
        {'site_id': 'USGS:1003', 'state': 'WY', 'name': 'Hoback River near Bondurant, WY',
         'huc8': '17040101', 'area_km2': 1400.0, 'begin_date': '2000-01-01',
         'end_date': '2026-11-25', 'count_nu': 9700},
        # Too small (< 25 km^2)
        {'site_id': 'USGS:1004', 'state': 'WY', 'name': 'Tiny Creek, WY',
         'huc8': '17040102', 'area_km2': 12.0, 'begin_date': '1990-01-01',
         'end_date': '2026-11-25', 'count_nu': 13400},
        # Too short (< 15 yrs)
        {'site_id': 'USGS:1005', 'state': 'NM', 'name': 'Short Record River, NM',
         'huc8': '13020101', 'area_km2': 500.0, 'begin_date': '2018-01-01',
         'end_date': '2026-11-25', 'count_nu': 3200},
        # Too many missing days (< 90% complete)
        {'site_id': 'USGS:1006', 'state': 'NM', 'name': 'Gappy River, NM',
         'huc8': '13020102', 'area_km2': 500.0, 'begin_date': '2000-01-01',
         'end_date': '2026-11-25', 'count_nu': 7000},
        # Canal/ditch excluded by name
        {'site_id': 'USGS:1007', 'state': 'NM', 'name': 'Irrigation Canal near Taos, NM',
         'huc8': '13020103', 'area_km2': 500.0, 'begin_date': '1995-01-01',
         'end_date': '2026-11-25', 'count_nu': 11500},
        # Good NM gauge
        {'site_id': 'USGS:1008', 'state': 'NM', 'name': 'Rio Chama near La Puente, NM',
         'huc8': '13020102', 'area_km2': 1240.0, 'begin_date': '1992-01-01',
         'end_date': '2026-11-26', 'count_nu': 12600},
        # Core gauge excluded from regional additions
        {'site_id': 'USGS:08276500', 'state': 'NM', 'name': 'Rio Grande below Taos Junction',
         'huc8': '13020101', 'area_km2': 16756.0, 'begin_date': '1990-01-01',
         'end_date': '2026-11-26', 'count_nu': 13400},
    ]
    selected = crs.filter_candidates(
        cands, min_years=15.0, min_completeness=0.90, min_area_km2=25.0,
        max_area_km2=50000.0, max_per_huc8=2, max_per_state=5, max_total=10,
        exclude_site_ids={'USGS:08276500'}, as_of=as_of)
    ids = [r['site_id'] for r in selected]
    assert set(ids) == {'USGS:1001', 'USGS:1002', 'USGS:1008'}
    assert 'USGS:1003' not in ids
    assert 'USGS:1004' not in ids
    assert 'USGS:1005' not in ids
    assert 'USGS:1006' not in ids
    assert 'USGS:1007' not in ids
    assert 'USGS:08276500' not in ids


def test_combine_data_get_site_ids_respects_env_and_comments(tmp_path, monkeypatch):
    p = tmp_path / 'custom_sites.txt'
    p.write_text("# header comment\n\nUSGS:09163500\n# mid comment\nDWR:ARKCANCO\n")
    monkeypatch.setenv('OPENFLOW_SITE_IDS_FILE', str(p))
    assert combine_data.get_site_ids() == ['USGS:09163500', 'DWR:ARKCANCO']


def _make_sample(site_id, station_idx, basin_idx, day_offset, static_dim=4):
    return windowing.WindowedSample(
        encoder_X=np.zeros((10, len(windowing.ENCODER_FEATURES)), dtype='float32'),
        decoder_X=np.zeros((4, len(windowing.DECODER_FEATURES)), dtype='float32'),
        target_Y=np.zeros((4, len(windowing.TARGET_FEATURES)), dtype='float32'),
        persistence_anchor=np.zeros((len(windowing.TARGET_FEATURES),), dtype='float32'),
        station_idx=station_idx,
        basin_idx=basin_idx,
        site_id=site_id,
        anchor_date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=day_offset),
        static_X=np.full((static_dim,), float(station_idx), dtype='float32'),
    )


def test_leave_basin_out_folds_zeroes_unseen_indices_and_separates_basins():
    # 4 sites across 3 basins (sites A1 & A2 share basin 1).
    samples = []
    specs = [('USGS:A1', 1, 1), ('USGS:A2', 2, 1), ('USGS:B1', 3, 2), ('USGS:C1', 4, 3)]
    for sid, st_i, b_i in specs:
        for d in range(0, 120, 3):
            samples.append(_make_sample(sid, st_i, b_i, d))

    folds = windowing.leave_basin_out_folds(samples, n_folds=3, val_frac=0.2, embargo_days=4)
    assert len(folds) == 3

    all_tested_sites = set()
    for fold in folds:
        tr_sites = {s.site_id for s in fold.train}
        va_sites = {s.site_id for s in fold.val}
        te_sites = {s.site_id for s in fold.test}
        assert tr_sites and va_sites and te_sites
        # Zero site overlap between train/val and held-out test.
        assert not (te_sites & tr_sites)
        assert not (te_sites & va_sites)
        # Basin integrity: A1 and A2 share basin 1 so they must always be held out together.
        assert ('USGS:A1' in te_sites) == ('USGS:A2' in te_sites)
        # Every held-out test sample is masked to unseen station_idx=0 and unseen basin_idx=0.
        assert all(s.station_idx == 0 for s in fold.test)
        assert all(s.basin_idx == 0 for s in fold.test)
        # static_input stacks cleanly.
        te_x, _ = windowing.stack(fold.test)
        assert 'static_input' in te_x
        assert te_x['static_input'].shape == (len(fold.test), 4)
        all_tested_sites |= te_sites

    assert all_tested_sites == {'USGS:A1', 'USGS:A2', 'USGS:B1', 'USGS:C1'}
