"""
Parity tests for the Phase 6 mobile export.

Trains a tiny seq2seq model, exports it to CoreML + TFLite, and verifies that:
  - both formats load in their respective runtimes
  - per-sample predictions match the Keras .h5 within tolerance
  - manifest.json sha256s match files on disk
  - the schema block in manifest.json reflects the training config

This is heavy (requires tensorflow + coremltools), so it's NOT in the default
`tests.yml` selection -- it runs in the opt-in export.yml workflow. Locally,
invoke with: pytest tests/test_export_parity.py
"""

import hashlib
import json
import os

import numpy as np
import pytest

pytest.importorskip('tensorflow')
pytest.importorskip('coremltools')

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import model as model_mod
import windowing
import export_mobile


def _build_tiny_model():
    return model_mod.build_encoder_decoder(
        num_stations=3,
        num_basins=2,
        encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES),
        encoder_days=windowing.ENCODER_DAYS,
        decoder_days=windowing.DECODER_DAYS,
        lstm_units=8,
        embedding_dim=4,
        station_dropout=0.0,   # inference-time parity: no stochasticity at all
        dropout=0.0,
        recurrent_dropout=0.0,
    )


def _write_training_config(base_path, num_stations=3, num_basins=2):
    config = {
        'encoder_days': windowing.ENCODER_DAYS,
        'decoder_days': windowing.DECODER_DAYS,
        'encoder_features': windowing.ENCODER_FEATURES,
        'decoder_features': windowing.DECODER_FEATURES,
        'target_features': windowing.TARGET_FEATURES,
        'num_stations': num_stations,
        'num_basins': num_basins,
    }
    with open(os.path.join(base_path, 'training_config.json'), 'w') as f:
        json.dump(config, f)
    return config


def _seed_index_files(base_path):
    """The exporter doesn't read these, but write them for manifest realism."""
    for name in ('scalers.json', 'station_index.json', 'basin_index.json'):
        with open(os.path.join(base_path, name), 'w') as f:
            json.dump({}, f)


def _sample_inputs(n=1):
    rng = np.random.default_rng(0)
    return {
        'encoder_input': rng.standard_normal(
            (n, windowing.ENCODER_DAYS, len(windowing.ENCODER_FEATURES))).astype('float32'),
        'decoder_input': rng.standard_normal(
            (n, windowing.DECODER_DAYS, len(windowing.DECODER_FEATURES))).astype('float32'),
        'persistence_input': rng.standard_normal(
            (n, len(windowing.TARGET_FEATURES))).astype('float32'),
        'station_input': np.array([1] * n, dtype='int32'),
        'basin_input': np.array([1] * n, dtype='int32'),
    }


def test_export_round_trips_with_parity(tmp_path):
    base_path = str(tmp_path)
    net = _build_tiny_model()
    h5_path = os.path.join(base_path, 'lstm_model.h5')
    net.save(h5_path)
    config = _write_training_config(base_path)
    _seed_index_files(base_path)

    result = export_mobile.export_all(base_path)

    # Both formats made it to disk.
    assert result['mlpackage_zip'] and os.path.exists(result['mlpackage_zip'])
    assert result['tflite'] and os.path.exists(result['tflite'])
    _assert_mobile_parity(result, base_path, net)


def _assert_mobile_parity(result, base_path, net):
    """
    Both exports answer to the names the mobile apps use and match Keras:
      - Android runs the TFLite SignatureRunner and reads the output key
        'persistence_plus_delta' (OpenFlowAndroid ml/ModelBundle.kt)
      - iOS reads featureValue(for: "persistence_plus_delta")
        (OpenFlowiOS/ModelBundle.swift)
    """
    import sys
    import tensorflow as tf
    keras_inputs = _sample_inputs(n=4)
    keras_pred = net.predict(keras_inputs, verbose=0)

    if not isinstance(keras_pred, dict):
        keras_pred = {export_mobile.OUTPUT_NAME: keras_pred}
    n = keras_inputs['encoder_input'].shape[0]

    runner = tf.lite.Interpreter(model_path=result['tflite']).get_signature_runner()
    for i in range(n):
        out = runner(**{name: keras_inputs[name][i:i+1] for name in keras_inputs})
        assert sorted(out) == sorted(keras_pred)
        for name, want in keras_pred.items():
            assert np.max(np.abs(out[name] - want[i:i+1])) < 1e-3, name

    import coremltools as ct
    ct_model = ct.models.MLModel(os.path.join(base_path, export_mobile.MLPACKAGE_NAME))
    spec = ct_model.get_spec()
    assert sorted(o.name for o in spec.description.output) == sorted(keras_pred)
    assert sorted(i.name for i in spec.description.input) == sorted(keras_inputs)
    if sys.platform != 'darwin':
        return   # CoreML can only run predictions on macOS
    for i in range(n):
        ct_out = ct_model.predict({name: keras_inputs[name][i:i+1] for name in keras_inputs})
        for name, want in keras_pred.items():
            assert np.max(np.abs(np.asarray(ct_out[name]) - want[i:i+1])) < 1e-3, name


def test_ensemble_export_round_trips_with_parity(tmp_path):
    """What CI ships: an averaged ensemble of members (model.average_ensemble)."""
    base_path = str(tmp_path)
    net = model_mod.average_ensemble([_build_tiny_model(), _build_tiny_model()])
    net.save(os.path.join(base_path, 'lstm_model.h5'))
    _write_training_config(base_path)
    _seed_index_files(base_path)

    result = export_mobile.export_all(base_path)

    assert result['mlpackage_zip'] and os.path.exists(result['mlpackage_zip'])
    assert result['tflite'] and os.path.exists(result['tflite'])
    _assert_mobile_parity(result, base_path, net)


def test_manifest_records_correct_sha256s(tmp_path):
    base_path = str(tmp_path)
    net = _build_tiny_model()
    h5_path = os.path.join(base_path, 'lstm_model.h5')
    net.save(h5_path)
    _write_training_config(base_path)
    _seed_index_files(base_path)

    export_mobile.export_all(base_path)

    with open(os.path.join(base_path, 'manifest.json')) as f:
        manifest = json.load(f)

    # Every file the manifest lists must exist and hash to the recorded sha.
    for entry in manifest['files']:
        path = os.path.join(base_path, entry['name'])
        assert os.path.exists(path)
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        assert entry['sha256'] == h.hexdigest()
        assert entry['bytes'] == os.path.getsize(path)


def test_manifest_schema_matches_training_config(tmp_path):
    base_path = str(tmp_path)
    net = _build_tiny_model()
    net.save(os.path.join(base_path, 'lstm_model.h5'))
    config = _write_training_config(base_path)
    _seed_index_files(base_path)

    export_mobile.export_all(base_path)

    with open(os.path.join(base_path, 'manifest.json')) as f:
        manifest = json.load(f)

    for key in ('encoder_days', 'decoder_days', 'encoder_features',
                'decoder_features', 'target_features',
                'num_stations', 'num_basins'):
        assert manifest['schema'][key] == config[key]


def test_band_model_exports_both_outputs(tmp_path):
    """What train.py ships: a calibrated two-output ensemble with the river-specific gain."""
    base_path = str(tmp_path)
    def member():
        return model_mod.build_encoder_decoder(
            num_stations=3, num_basins=2, encoder_features=len(windowing.ENCODER_FEATURES),
            decoder_features=len(windowing.DECODER_FEATURES),
            target_features=len(windowing.TARGET_FEATURES), encoder_days=windowing.ENCODER_DAYS,
            decoder_days=windowing.DECODER_DAYS, lstm_units=8, embedding_dim=4,
            station_dropout=0.0, dropout=0.0, recurrent_dropout=0.0,
            quantiles=model_mod.DEFAULT_QUANTILES,
            upstream_change_index=windowing.DECODER_FEATURES.index('upstream_change'))
    net = model_mod.average_ensemble([member(), member()])
    model_mod.set_band_scale(net, np.full((windowing.DECODER_DAYS, 2, 2), 1.5, 'float32'))
    net.save(os.path.join(base_path, 'lstm_model.h5'))
    _write_training_config(base_path)
    _seed_index_files(base_path)

    result = export_mobile.export_all(base_path)

    assert result['mlpackage_zip'] and result['tflite']
    _assert_mobile_parity(result, base_path, net)
