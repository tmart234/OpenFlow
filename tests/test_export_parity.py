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

    # Keras reference predictions.
    keras_inputs = _sample_inputs(n=4)
    keras_pred = net.predict(keras_inputs, verbose=0)

    # TFLite parity (one sample at a time -- fixed batch=1 in input specs).
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=result['tflite'])
    interp.allocate_tensors()
    in_details = {d['name'].split(':')[0]: d for d in interp.get_input_details()}
    out_details = interp.get_output_details()
    for i in range(keras_pred.shape[0]):
        for name in ('encoder_input', 'decoder_input', 'persistence_input',
                     'station_input', 'basin_input'):
            # Names may carry serving prefixes ("serving_default_encoder_input")
            # depending on TF version; match by suffix.
            matched = next((v for k, v in in_details.items() if k.endswith(name)), None)
            assert matched is not None, f"TFLite input {name} not found in {list(in_details)}"
            interp.set_tensor(matched['index'], keras_inputs[name][i:i+1])
        interp.invoke()
        tflite_pred = interp.get_tensor(out_details[0]['index'])
        assert np.max(np.abs(tflite_pred - keras_pred[i:i+1])) < 1e-3

    # CoreML parity.
    import coremltools as ct
    ct_model = ct.models.MLModel(os.path.join(base_path, export_mobile.MLPACKAGE_NAME))
    for i in range(keras_pred.shape[0]):
        ct_inputs = {name: keras_inputs[name][i:i+1] for name in keras_inputs}
        ct_out = ct_model.predict(ct_inputs)
        # CoreML names the output after the Add layer; pick the single output.
        coreml_pred = next(iter(ct_out.values()))
        assert np.max(np.abs(np.asarray(coreml_pred) - keras_pred[i:i+1])) < 1e-3


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
