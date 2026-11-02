import numpy as np
import pytest

import combine_data
from data import basin_zones

SQUARE = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
HOLE = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]]


def test_liquid_fraction_ramp():
    np.testing.assert_allclose(basin_zones.liquid_fraction([-5, -1, 1, 3, 10]),
                               [0, 0, 0.5, 1, 1])


def test_points_in_rings_handles_holes_and_multiple_polygons():
    lon = np.array([0.5, 1.5, 3.5, 5.0, 10.5])
    lat = np.array([0.5, 1.5, 3.5, 5.0, 10.5])
    other = [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0], [10.0, 10.0]]
    inside = basin_zones.points_in_rings(lon, lat, [SQUARE, HOLE, other])
    np.testing.assert_array_equal(inside, [True, False, True, False, True])


def test_grid_masks_and_bbox():
    cfg = {'zones': {'local': {'rings': [SQUARE]}, 'mid': {'rings': []}, 'far': {'rings': []}},
           'ungauged': {'rings': [HOLE]}}
    lon2d, lat2d = np.meshgrid(np.arange(0.25, 4, 0.5), np.arange(0.25, 4, 0.5))
    m = basin_zones.grid_masks(cfg, lon2d, lat2d)
    assert m['local'].all() and not m['mid'].any()
    assert m['ungauged'].sum() == 4
    assert basin_zones.bbox(cfg, pad=0.0) == (0.0, 4.0, 0.0, 4.0)


def test_point_weights_normalise_within_each_area():
    cfg = {'points': [{'zone': 'local', 'ungauged': True, 'w': 3.0},
                      {'zone': 'local', 'ungauged': False, 'w': 1.0},
                      {'zone': 'far', 'ungauged': False, 'w': 2.0}]}
    w = basin_zones.point_weights(cfg)
    np.testing.assert_allclose(w['local'], [0.75, 0.25, 0])
    np.testing.assert_allclose(w['far'], [0, 0, 1])
    np.testing.assert_allclose(w['mid'], [0, 0, 0])
    np.testing.assert_allclose(w['ungauged'], [1, 0, 0])


def test_mm_per_hour_to_cfs():
    # 1 mm/h over 3.6 km2 = 1 m3/s = 35.31 cfs
    assert basin_zones.mm_per_hour_to_cfs(1.0, 3.6) == pytest.approx(35.3147, rel=1e-4)


def test_committed_zones_cover_every_training_station():
    zones = basin_zones.load_zones()
    for site in combine_data.get_site_ids():
        cfg = zones.get(site)
        assert cfg, f"run data/build_basin_zones.py for {site}"
        areas = [cfg['zones'][z]['area_km2'] for z in basin_zones.ZONES]
        assert sum(areas) == pytest.approx(cfg['contributing_area_km2'], rel=0.01)
        assert cfg['points'] and all(p.get('nws') for p in cfg['points'])
        assert {p['zone'] for p in cfg['points']} <= set(basin_zones.ZONES)
        # Every non-empty zone has a polygon and at least one sample point.
        for z in basin_zones.ZONES:
            if cfg['zones'][z]['area_km2'] > 0:
                assert cfg['zones'][z]['rings']
                assert any(p['zone'] == z for p in cfg['points'])


def test_taos_box_zones_exclude_the_closed_basin_and_mark_gauged_area():
    cfg = basin_zones.load_zones()['USGS:08276500']
    # USGS: 6,790 mi2 contributing (~17,600 km2) of 9,730 mi2 total.
    assert 15000 < cfg['contributing_area_km2'] < 19000
    huc8s = {h for z in cfg['zones'].values() for h in z['huc8']}
    assert '13010003' not in huc8s and '13010004' not in huc8s   # Closed Basin
    assert set(cfg['ungauged']['gauges']) == {'08223000', '08246500'}
    assert 0 < cfg['ungauged']['area_km2'] < cfg['contributing_area_km2']
    assert any(not p['ungauged'] for p in cfg['points'])
