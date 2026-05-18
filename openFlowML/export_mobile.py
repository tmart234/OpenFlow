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
MANIFEST_NAME = 'manifest.json'
TRAINING_CONFIG_NAME = 'training_config.json'


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


def write_manifest(base_path, config, artifact_files, tflite_mode):
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

    artifact_files = [H5_NAME, 'scalers.json', 'station_index.json',
                      'basin_index.json', TRAINING_CONFIG_NAME]
    if mlpackage_zip:
        artifact_files.append(MLPACKAGE_ZIP_NAME)
    if tflite_result:
        artifact_files.append(TFLITE_NAME)
    write_manifest(base_path, config, artifact_files, tflite_mode)

    if not mlpackage_zip and not tflite_result:
        raise RuntimeError("Both CoreML and TFLite exports failed")
    return {
        'mlpackage_zip': mlpackage_zip,
        'tflite': tflite_result,
        'tflite_mode': tflite_mode,
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
