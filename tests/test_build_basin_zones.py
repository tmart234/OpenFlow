import pytest

pytest.importorskip('shapely')

from shapely.geometry import box  # noqa: E402

from data import build_basin_zones as bbz  # noqa: E402


def test_assign_zones_by_nested_distance_sets():
    zones = bbz.assign_zones({'a', 'b', 'c'}, local_ids={'a'}, mid_ids={'a', 'b'})
    assert zones == {'a': 'local', 'b': 'mid', 'c': 'far'}


def test_rings_of_polygon_and_multipolygon():
    poly = box(0, 0, 1, 1)
    assert len(bbz.rings_of(poly)) == 1
    multi = box(0, 0, 1, 1).union(box(3, 3, 4, 4))
    rings = bbz.rings_of(multi)
    assert len(rings) == 2 and all(r[0] == r[-1] for r in rings)


def test_sample_points_cover_zones_and_flag_ungauged():
    zones = {'local': box(0, 0, 1, 1), 'mid': box(1, 0, 2, 1), 'far': box(2, 0, 2.05, 0.05)}
    ungauged = box(0, 0, 1.5, 1)
    pts = bbz.sample_points(zones, ungauged, total_area_km2=2 * 111 * 111, n_points=20)
    assert {p['zone'] for p in pts} == {'local', 'mid', 'far'}   # tiny far zone still sampled
    assert any(p['ungauged'] for p in pts) and any(not p['ungauged'] for p in pts)
    for p in pts:
        assert zones[p['zone']].buffer(1e-9).contains(__import__('shapely.geometry', fromlist=['Point']).Point(p['lon'], p['lat']))
        assert p['w'] > 0
    assert 10 <= len(pts) <= 40


def test_usgs_id_for_usgs_sites_needs_no_network():
    assert bbz.usgs_id_for('USGS:08276500') == '08276500'
    with pytest.raises(ValueError):
        bbz.usgs_id_for('NOAA:1')


@pytest.mark.network
def test_live_nldi_finds_taos_upstream_huc12s():
    ids = bbz.upstream_huc12('08276500')
    assert len(ids) > 100
    assert not any(h.startswith('13010003') for h in ids)   # Closed Basin excluded
