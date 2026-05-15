import os
import numpy as np
import pytest

# The model builder needs tensorflow; skip cleanly in the test venv that
# intentionally does NOT install it (tensorflow conflicts with earthaccess on
# typing-extensions). The CI's separate `model-smoke` job installs the core
# training requirements and runs these tests there.
pytest.importorskip('tensorflow')

# Keep the model lightweight + deterministic in this smoke test.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import model
import windowing


def _build_smoke_model():
    return model.build_encoder_decoder(
        num_stations=4,
        num_basins=3,
        encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES),
        encoder_days=windowing.ENCODER_DAYS,
        decoder_days=windowing.DECODER_DAYS,
        lstm_units=8,
        embedding_dim=4,
    )


def test_model_builds_with_expected_output_shape():
    net = _build_smoke_model()
    assert net.output_shape == (None, windowing.DECODER_DAYS, len(windowing.TARGET_FEATURES))


def test_model_fits_one_step_without_nan():
    import tensorflow as tf
    tf.random.set_seed(0)
    net = _build_smoke_model()
    n = 4
    inputs = {
        'encoder_input': np.random.randn(n, windowing.ENCODER_DAYS,
                                         len(windowing.ENCODER_FEATURES)).astype('float32'),
        'decoder_input': np.random.randn(n, windowing.DECODER_DAYS,
                                         len(windowing.DECODER_FEATURES)).astype('float32'),
        'station_input': np.array([1, 2, 3, 4], dtype='int32'),
        'basin_input': np.array([1, 2, 3, 1], dtype='int32'),
    }
    target = np.random.randn(n, windowing.DECODER_DAYS,
                             len(windowing.TARGET_FEATURES)).astype('float32')
    history = net.fit(inputs, target, epochs=1, batch_size=2, verbose=0)
    loss = history.history['loss'][-1]
    assert np.isfinite(loss), f"training step produced non-finite loss {loss}"


def test_model_predicts_for_unseen_station_via_basin_fallback():
    """station_idx=0 ("unknown") must still produce a valid prediction."""
    net = _build_smoke_model()
    inputs = {
        'encoder_input': np.random.randn(1, windowing.ENCODER_DAYS,
                                         len(windowing.ENCODER_FEATURES)).astype('float32'),
        'decoder_input': np.random.randn(1, windowing.DECODER_DAYS,
                                         len(windowing.DECODER_FEATURES)).astype('float32'),
        'station_input': np.array([0], dtype='int32'),   # unseen station
        'basin_input': np.array([1], dtype='int32'),     # known basin
    }
    pred = net.predict(inputs, verbose=0)
    assert pred.shape == (1, windowing.DECODER_DAYS, len(windowing.TARGET_FEATURES))
    assert np.isfinite(pred).all()
