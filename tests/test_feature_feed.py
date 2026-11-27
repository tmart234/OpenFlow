"""Offline tests for Phase 4 Daily Mobile Feature Feed (`openFlowML/feature_feed.py`)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import feature_feed
import ml_inference
import normalize_data
import windowing


class _DummyModel:
    """Deterministic lightweight stand-in for a Keras model (no TF import needed)."""

    def __init__(self, with_band=True):
        self.with_band = with_band
        self.last_inputs = None

    def predict(self, inputs, verbose=0):
        self.last_inputs = {k: np.asarray(v) for k, v in inputs.items()}
        pers = np.asarray(inputs['persistence_input'], dtype='float32')  # (1, 2)
        dec = np.asarray(inputs['decoder_input'], dtype='float32')       # (1, H, F)
        h = dec.shape[1]
        static_bias = 0.0
        if 'static_input' in inputs:
            static_bias = 0.05 * float(np.mean(inputs['static_input']))
        delta = 0.02 * np.mean(dec, axis=-1, keepdims=True) + static_bias
        point = np.repeat(pers[:, None, :], h, axis=1) + delta
        if not self.with_band:
            return point.astype('float32')
        # 5 ordered quantiles around point
        offsets = np.array([-0.20, -0.08, 0.0, 0.09, 0.25], dtype='float32')
        band = point[..., None] + offsets
        return {
            'persistence_plus_delta': point.astype('float32'),
            'flow_quantiles': band.astype('float32'),
        }


def _make_artifacts(with_static=False, flow_units='cfs'):
    enc_feats = list(windowing.ENCODER_FEATURES[:10])
    dec_feats = list(windowing.DECODER_FEATURES[:8])
    tc = {
        'encoder_days': 6,
        'decoder_days': 4,
        'encoder_features': enc_feats,
        'decoder_features': dec_feats,
        'target_features': ['Min Flow', 'Max Flow'],
        'num_stations': 2,
        'num_basins': 2,
        'outputs': ['persistence_plus_delta', 'flow_quantiles'],
        'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
        'flow_units': flow_units,
    }
    if with_static:
        tc['static_features'] = ['log_area_km2', 'elev_mean_m', 'ungauged_frac']
        tc['station_areas_km2'] = {'USGS:08276500': 1000.0}
    scalers = {
        'Min Flow': {'mean': 5.0, 'scale': 1.0, 'transform': 'log1p'},
        'Max Flow': {'mean': 5.2, 'scale': 1.0, 'transform': 'log1p'},
    }
    return {
        'training_config': tc,
        'scalers': scalers,
        'station_index': {'USGS:08276500': 1},
        'basin_index': {'13020101': 1},
        'registry': {
            'USGS:08276500': {
                'name': 'Rio Grande below Taos Junction Bridge',
                'short_name': 'Lower Taos Box',
                'aliases': ['taos-box', 'TJBN5'],
            },
        },
        'version': 'model-test-1',
        'model': _DummyModel(with_band=True),
    }


def _make_prepared(art, with_static=False, flow_units='cfs'):
    rng = np.random.default_rng(7)
    tc = art['training_config']
    inputs = {
        'encoder_input': rng.standard_normal((1, tc['encoder_days'], len(tc['encoder_features']))).astype('float32'),
        'decoder_input': rng.standard_normal((1, tc['decoder_days'], len(tc['decoder_features']))).astype('float32'),
        'persistence_input': np.array([[0.15, 0.22]], dtype='float32'),
        'station_input': np.array([1], dtype='int32'),
        'basin_input': np.array([1], dtype='int32'),
    }
    if with_static:
        inputs['static_input'] = np.array([[0.4, -0.3, 0.7]], dtype='float32')
    dates = pd.date_range('2026-11-28', periods=tc['decoder_days'], name='Date')
    return {
        'inputs': inputs,
        'anchor': pd.Timestamp('2026-11-27'),
        'dates': dates,
        'info': {
            'anchor': '2026-11-27',
            'today': '2026-11-27',
            'station_idx': 1,
            'basin_idx': 1,
            'trained_station': True,
            'flow_units': flow_units,
            'area_km2': 1000.0 if flow_units == 'mm_day' else None,
        },
    }


@pytest.mark.parametrize('with_static,flow_units', [(False, 'cfs'), (True, 'mm_day')])
def test_feed_round_trip_matches_direct_predict(with_static, flow_units):
    art = _make_artifacts(with_static=with_static, flow_units=flow_units)
    prepared = _make_prepared(art, with_static=with_static, flow_units=flow_units)

    payload = feature_feed.build_station_feed('USGS:08276500', art, prepared=prepared)
    # Survives JSON round-trip cleanly.
    round_tripped = json.loads(json.dumps(payload))
    feature_feed.verify_feed_payload(round_tripped, training_config=art['training_config'])

    direct_df = ml_inference.predict(art, prepared)
    feed_df = ml_inference.predict_from_feed(art, round_tripped)

    assert list(direct_df.columns) == list(feed_df.columns)
    assert list(direct_df['Date']) == list(feed_df['Date'])
    for col in direct_df.columns:
        if col == 'Date':
            continue
        np.testing.assert_allclose(feed_df[col].to_numpy(), direct_df[col].to_numpy(), rtol=1e-4, atol=1e-3)


def test_verify_detects_tampering_and_schema_drift():
    art = _make_artifacts(with_static=False)
    prepared = _make_prepared(art, with_static=False)
    payload = feature_feed.build_station_feed('USGS:08276500', art, prepared=prepared)

    # Tamper a single float -> checksum must fail.
    bad_hash = json.loads(json.dumps(payload))
    bad_hash['inputs']['persistence_input'][0][0] += 0.1
    with pytest.raises(ValueError, match='inputs_sha256'):
        feature_feed.verify_feed_payload(bad_hash)

    # Change model schema -> schema_sha256 must fail.
    drifted_tc = dict(art['training_config'], decoder_days=14)
    with pytest.raises(ValueError, match='schema_sha256'):
        feature_feed.verify_feed_payload(payload, training_config=drifted_tc)


def test_run_feed_writes_catalog_and_isolates_station_errors(tmp_path, monkeypatch):
    art = _make_artifacts(with_static=False)
    prepared = _make_prepared(art, with_static=False)

    def fake_prepare(site_id, a, today=None, tz=None):
        if site_id == 'USGS:08276500':
            return prepared
        raise ml_inference.InferenceError("simulated gauge outage")

    monkeypatch.setattr(ml_inference, 'prepare_inputs', fake_prepare)
    catalog = feature_feed.run_feed(
        ['taos-box', 'USGS:99999999'], str(tmp_path), str(tmp_path / 'feed'), art=art)

    assert len(catalog['stations']) == 1
    assert catalog['stations'][0]['slug'] == 'taos-box'
    assert 'USGS:99999999' in catalog['errors']
    cat_path = tmp_path / 'feed' / 'catalog.json'
    station_path = tmp_path / 'feed' / 'inputs' / 'taos-box.json'
    assert cat_path.exists()
    assert station_path.exists()
    assert catalog['stations'][0]['bytes'] == os.path.getsize(station_path)
    loaded = json.loads(station_path.read_text())
    feature_feed.verify_feed_payload(loaded, training_config=art['training_config'])


def test_run_feed_resolves_repo_alias_absent_from_legacy_model_registry(tmp_path, monkeypatch):
    """Legacy releases (e.g. model-219) lack clear-creek in their bundled upstream_gauges.json."""
    art = _make_artifacts(with_static=False)
    # Deliberately omit clear-creek from art['registry'] (mimicking model-219).
    assert 'USGS:06719505' not in art['registry']
    prepared = _make_prepared(art, with_static=False)
    seen = []

    def fake_prepare(site_id, a, today=None, tz=None):
        seen.append(site_id)
        return prepared

    monkeypatch.setattr(ml_inference, 'prepare_inputs', fake_prepare)
    catalog = feature_feed.run_feed(['clear-creek'], str(tmp_path), str(tmp_path / 'feed'), art=art)
    assert seen == ['USGS:06719505']
    assert catalog['errors'] == {}
    assert catalog['stations'][0]['slug'] == 'clear-creek'

