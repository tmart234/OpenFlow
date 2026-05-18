"""
Phase 6: convert the trained Keras model into mobile-friendly formats.

Produces, alongside the canonical lstm_model.h5:
  - lstm_model.mlpackage(.zip)  -- CoreML mlprogram for iOS (iOS15+)
  - lstm_model.tflite           -- TFLite for Android
  - manifest.json               -- sha256s + schema + tool versions so the
                                   mobile app can verify what it downloaded

Can be invoked standalone after training:

    python -m export_mobile --base-path .

train.py calls this as a best-effort post-save step; a failed export must NOT
lose the trained .h5, so the workflow runs this with continue-on-error and
the release step is gated on its success.

The single highest-risk piece is TFLite conversion of the dual
Embedding(mask_zero=True) + LSTM-with-initial_state graph. We try three paths
in order and log which one wins:
  1. Strict TFLITE_BUILTINS.
  2. TFLITE_BUILTINS + SELECT_TF_OPS (Flex delegate; needs select-tf-ops AAR).
  3. Rebuild the model with mask_zero=False and copy weights; mask is unused
     at inference because unseen-station fallback maps to index 0 explicitly.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

H5_NAME = 'lstm_model.h5'
MLPACKAGE_NAME = 'lstm_model.mlpackage'
MLPACKAGE_ZIP_NAME = 'lstm_model.mlpackage.zip'
TFLITE_NAME = 'lstm_model.tflite'
TFLITE_INT8_NAME = 'lstm_model_int8.tflite'
MANIFEST_NAME = 'manifest.json'
TRAINING_CONFIG_NAME = 'training_config.json'

# Maximum per-output absolute error (scaled space) the int8-quantized TFLite
# is allowed to introduce vs the float32 Keras model. Beyond this we ship
# only the float32 TFLite -- the size win isn't worth a noticeable accuracy
# regression. Tuned to roughly the noise floor of the float-32 conversion
# parity tests in tests/test_export_parity.py.
TFLITE_INT8_MAX_ABS_DIFF = 0.05


def _load_keras_model(h5_path):
    import tensorflow as tf
    import model as model_mod
    return tf.keras.models.load_model(
        h5_path, custom_objects={'StationDropout': model_mod.StationDropout})


def _input_specs_from_config(config):
    """TensorSpecs for the five model inputs, in the order build_encoder_decoder declares them."""
    import tensorflow as tf
    return [
        tf.TensorSpec(shape=(1, config['encoder_days'], len(config['encoder_features'])),
                     dtype=tf.float32, name='encoder_input'),
        tf.TensorSpec(shape=(1, config['decoder_days'], len(config['decoder_features'])),
                     dtype=tf.float32, name='decoder_input'),
        tf.TensorSpec(shape=(1, len(config['target_features'])),
                     dtype=tf.float32, name='persistence_input'),
        tf.TensorSpec(shape=(1,), dtype=tf.int32, name='station_input'),
        tf.TensorSpec(shape=(1,), dtype=tf.int32, name='basin_input'),
    ]


def _make_serving_fn(net):
    """Wrap the dict-input Keras model in a positional tf.function for converter consumption."""
    import tensorflow as tf

    @tf.function
    def serving_fn(encoder_input, decoder_input, persistence_input, station_input, basin_input):
        return net({
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'persistence_input': persistence_input,
            'station_input': station_input,
            'basin_input': basin_input,
        }, training=False)

    return serving_fn


def export_coreml(h5_path, out_dir):
    """
    Convert Keras .h5 -> CoreML .mlpackage (mlprogram), then zip the directory
    for release upload. Returns the path to the .zip on success, None on
    failure (so callers can degrade gracefully).
    """
    try:
        import coremltools as ct
    except ImportError:
        logger.error("coremltools not installed; skipping CoreML export")
        return None

    net = _load_keras_model(h5_path)
    try:
        ml_model = ct.convert(
            net,
            source='tensorflow',
            convert_to='mlprogram',
            minimum_deployment_target=ct.target.iOS15,
        )
    except Exception as e:
        logger.error("CoreML conversion failed: %s", e)
        return None

    mlpackage_path = os.path.join(out_dir, MLPACKAGE_NAME)
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path, ignore_errors=True)
    ml_model.save(mlpackage_path)

    # .mlpackage is a directory; release uploads need a single file.
    zip_base = os.path.join(out_dir, MLPACKAGE_NAME)
    zip_path = shutil.make_archive(zip_base, 'zip', root_dir=out_dir, base_dir=MLPACKAGE_NAME)
    logger.info("Wrote CoreML mlpackage: %s (zipped: %s)", mlpackage_path, zip_path)
    return zip_path


def _rebuild_without_mask_zero(net, config):
    """Build a fresh model with mask_zero=False and copy weights from `net`."""
    import model as model_mod
    rebuilt = model_mod.build_encoder_decoder(
        num_stations=config['num_stations'],
        num_basins=config['num_basins'],
        encoder_features=len(config['encoder_features']),
        decoder_features=len(config['decoder_features']),
        target_features=len(config['target_features']),
        encoder_days=config['encoder_days'],
        decoder_days=config['decoder_days'],
    )
    # Swap the two Embedding layers for mask_zero=False replicas, then copy
    # all weights layer-by-layer. The Keras builder is deterministic in layer
    # naming, so set_weights by name aligns the two graphs cleanly.
    import tensorflow as tf
    for layer in rebuilt.layers:
        if isinstance(layer, tf.keras.layers.Embedding) and layer.mask_zero:
            layer.mask_zero = False
    rebuilt.set_weights(net.get_weights())
    return rebuilt


def _random_sample_inputs(config, n=8, seed=0):
    """Synthetic (Keras-shaped) sample inputs for parity comparisons; CPU-only, no fixtures needed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return {
        'encoder_input': rng.standard_normal(
            (n, config['encoder_days'], len(config['encoder_features']))).astype('float32'),
        'decoder_input': rng.standard_normal(
            (n, config['decoder_days'], len(config['decoder_features']))).astype('float32'),
        'persistence_input': rng.standard_normal(
            (n, len(config['target_features']))).astype('float32'),
        # Index 0 always exists ("unseen station/basin"), so the synthetic
        # range stays inside the trained embedding vocab.
        'station_input': rng.integers(0, max(2, config['num_stations']), size=(n,)).astype('int32'),
        'basin_input': rng.integers(0, max(2, config['num_basins']), size=(n,)).astype('int32'),
    }


def export_tflite_int8(h5_path, out_path, config):
    """
    Dynamic-range quantization (weights -> int8, activations stay float32).
    Returns (out_path, max_abs_diff_vs_keras) on success, or (None, None)
    on failure. The caller is responsible for deciding whether the parity
    is good enough to ship.

    Why dynamic-range and not full-integer quantization: full-integer needs a
    representative dataset to calibrate activation ranges, which we don't
    have at export time outside of training. Dynamic-range is a one-line
    converter flag, runs every weight tensor through int8, and matches the
    float32 model to within a few percent on most architectures.
    """
    import numpy as np
    import tensorflow as tf

    net = _load_keras_model(h5_path)
    input_specs = _input_specs_from_config(config)
    serving_fn = _make_serving_fn(net)
    concrete = serving_fn.get_concrete_function(*input_specs)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Same Flex-fallback policy as the float32 path -- mask_zero=True
    # embeddings sometimes need it.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    try:
        tflite_bytes = converter.convert()
    except Exception as e:
        logger.warning("TFLite int8 conversion failed: %s", e)
        return None, None
    with open(out_path, 'wb') as f:
        f.write(tflite_bytes)
    logger.info("Wrote TFLite int8: %s (%d bytes)", out_path, len(tflite_bytes))

    # Parity check: int8 must stay within TFLITE_INT8_MAX_ABS_DIFF of the
    # float32 Keras prediction on synthetic inputs. If it doesn't, the caller
    # drops the int8 file and ships only float32.
    keras_inputs = _random_sample_inputs(config, n=8)
    keras_pred = net.predict(keras_inputs, verbose=0)

    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    in_details = {d['name'].split(':')[0]: d for d in interp.get_input_details()}
    out_details = interp.get_output_details()
    max_diff = 0.0
    for i in range(keras_pred.shape[0]):
        for name in ('encoder_input', 'decoder_input', 'persistence_input',
                     'station_input', 'basin_input'):
            matched = next((v for k, v in in_details.items() if k.endswith(name)), None)
            if matched is None:
                logger.warning("Int8 TFLite missing input %s; skipping parity", name)
                return out_path, None
            interp.set_tensor(matched['index'], keras_inputs[name][i:i+1])
        interp.invoke()
        int8_pred = interp.get_tensor(out_details[0]['index'])
        diff = float(np.max(np.abs(int8_pred - keras_pred[i:i+1])))
        max_diff = max(max_diff, diff)
    logger.info("TFLite int8 parity: max abs diff vs Keras = %.6f", max_diff)
    return out_path, max_diff


def export_tflite(h5_path, out_path, config):
    """
    Convert Keras .h5 -> TFLite. Returns (out_path, mode) on success or
    (None, None) on failure. `mode` is one of 'builtins', 'select_tf_ops',
    'rebuilt_no_mask' so we can record which path won in the manifest.
    """
    import tensorflow as tf

    net = _load_keras_model(h5_path)
    input_specs = _input_specs_from_config(config)

    def _try_convert(target_net, supported_ops, label):
        serving_fn = _make_serving_fn(target_net)
        concrete = serving_fn.get_concrete_function(*input_specs)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], target_net)
        converter.target_spec.supported_ops = supported_ops
        try:
            tflite_bytes = converter.convert()
        except Exception as e:
            logger.warning("TFLite conversion (%s) failed: %s", label, e)
            return None
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)
        logger.info("Wrote TFLite (%s): %s (%d bytes)", label, out_path, len(tflite_bytes))
        return out_path

    # Tier 1: strict builtins.
    result = _try_convert(net, [tf.lite.OpsSet.TFLITE_BUILTINS], 'builtins')
    if result:
        return result, 'builtins'
    # Tier 2: builtins + Flex delegate.
    result = _try_convert(net,
                          [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS],
                          'select_tf_ops')
    if result:
        return result, 'select_tf_ops'
    # Tier 3: rebuild without mask_zero, retry with strict builtins.
    rebuilt = _rebuild_without_mask_zero(net, config)
    result = _try_convert(rebuilt, [tf.lite.OpsSet.TFLITE_BUILTINS], 'rebuilt_no_mask')
    if result:
        return result, 'rebuilt_no_mask'
    return None, None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(base_path, config, artifact_files, tflite_mode,
                   tflite_int8_max_abs_diff=None):
    """
    Emit manifest.json next to the artifacts. The mobile app reads this to
    verify integrity, learn the input/output schema, and pick which artifact
    to use without parsing training_config.json.
    """
    import tensorflow as tf
    try:
        import coremltools as ct
        coremltools_version = ct.__version__
    except ImportError:
        coremltools_version = None

    files = []
    for name in artifact_files:
        path = os.path.join(base_path, name)
        if not os.path.exists(path):
            continue
        files.append({
            'name': name,
            'sha256': _sha256(path),
            'bytes': os.path.getsize(path),
        })

    manifest = {
        'model_version': os.environ.get('OPENFLOW_MODEL_VERSION',
                                        datetime.now(timezone.utc).strftime('model-%Y.%m.%d')),
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'tf_version': tf.__version__,
        'coremltools_version': coremltools_version,
        'tflite_mode': tflite_mode,
        'tflite_int8_max_abs_diff': tflite_int8_max_abs_diff,
        'schema': {
            'encoder_days': config['encoder_days'],
            'decoder_days': config['decoder_days'],
            'encoder_features': config['encoder_features'],
            'decoder_features': config['decoder_features'],
            'target_features': config['target_features'],
            'num_stations': config['num_stations'],
            'num_basins': config['num_basins'],
        },
        'files': files,
    }
    out_path = os.path.join(base_path, MANIFEST_NAME)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s (%d files)", out_path, len(files))
    return out_path


def export_all(base_path):
    """End-to-end: read training_config + .h5, emit mlpackage.zip + tflite + manifest."""
    h5_path = os.path.join(base_path, H5_NAME)
    config_path = os.path.join(base_path, TRAINING_CONFIG_NAME)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing {h5_path} -- run train.py first")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path} -- run train.py first")
    with open(config_path) as f:
        config = json.load(f)

    mlpackage_zip = export_coreml(h5_path, base_path)
    tflite_path = os.path.join(base_path, TFLITE_NAME)
    tflite_result, tflite_mode = export_tflite(h5_path, tflite_path, config)

    # Try the int8-quantized TFLite as an additional, optional artifact.
    # If conversion or the parity check fails we silently drop it -- the
    # float32 .tflite is the canonical Android artifact.
    tflite_int8_path = os.path.join(base_path, TFLITE_INT8_NAME)
    int8_path, int8_max_diff = export_tflite_int8(h5_path, tflite_int8_path, config)
    int8_shipped = False
    if int8_path is not None and int8_max_diff is not None:
        if int8_max_diff <= TFLITE_INT8_MAX_ABS_DIFF:
            int8_shipped = True
            logger.info("Int8 TFLite passes parity gate (%.4f <= %.4f); shipping",
                        int8_max_diff, TFLITE_INT8_MAX_ABS_DIFF)
        else:
            logger.warning(
                "Int8 TFLite parity %.4f exceeds gate %.4f; dropping int8 artifact",
                int8_max_diff, TFLITE_INT8_MAX_ABS_DIFF)
            try:
                os.remove(tflite_int8_path)
            except OSError:
                pass

    artifact_files = [H5_NAME, 'scalers.json', 'station_index.json',
                      'basin_index.json', TRAINING_CONFIG_NAME]
    if mlpackage_zip:
        artifact_files.append(MLPACKAGE_ZIP_NAME)
    if tflite_result:
        artifact_files.append(TFLITE_NAME)
    if int8_shipped:
        artifact_files.append(TFLITE_INT8_NAME)
    write_manifest(base_path, config, artifact_files, tflite_mode,
                   tflite_int8_max_abs_diff=int8_max_diff if int8_shipped else None)

    if not mlpackage_zip and not tflite_result:
        raise RuntimeError("Both CoreML and TFLite exports failed")
    return {
        'mlpackage_zip': mlpackage_zip,
        'tflite': tflite_result,
        'tflite_mode': tflite_mode,
        'tflite_int8': tflite_int8_path if int8_shipped else None,
        'tflite_int8_max_abs_diff': int8_max_diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-path', default=None,
                        help='Directory containing lstm_model.h5 and training_config.json. '
                             'Defaults to the repo root (combine_data.get_base_path()).')
    args = parser.parse_args()
    if args.base_path is None:
        import combine_data
        args.base_path = combine_data.get_base_path()
    result = export_all(args.base_path)
    logger.info("Export complete: %s", result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
