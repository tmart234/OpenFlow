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


def _smoke_inputs(n=4, station_ids=None, basin_ids=None, persistence=None):
    """Synthetic inputs matching the residual-learning model API."""
    rng = np.random.default_rng(0)
    if station_ids is None:
        station_ids = [(i % 4) + 1 for i in range(n)]
    if basin_ids is None:
        basin_ids = [((i + 1) % 3) + 1 for i in range(n)]
    return {
        'encoder_input': rng.standard_normal(
            (n, windowing.ENCODER_DAYS, len(windowing.ENCODER_FEATURES))).astype('float32'),
        'decoder_input': rng.standard_normal(
            (n, windowing.DECODER_DAYS, len(windowing.DECODER_FEATURES))).astype('float32'),
        'persistence_input': (persistence if persistence is not None
                              else rng.standard_normal(
                                  (n, len(windowing.TARGET_FEATURES))).astype('float32')),
        'station_input': np.array(station_ids, dtype='int32'),
        'basin_input': np.array(basin_ids, dtype='int32'),
    }


def test_model_fits_one_step_without_nan():
    import tensorflow as tf
    tf.random.set_seed(0)
    net = _build_smoke_model()
    inputs = _smoke_inputs(n=4)
    rng = np.random.default_rng(1)
    target = rng.standard_normal(
        (4, windowing.DECODER_DAYS, len(windowing.TARGET_FEATURES))).astype('float32')
    history = net.fit(inputs, target, epochs=1, batch_size=2, verbose=0)
    loss = history.history['loss'][-1]
    assert np.isfinite(loss), f"training step produced non-finite loss {loss}"


def test_model_predicts_for_unseen_station_via_basin_fallback():
    """station_idx=0 ("unknown") must still produce a valid prediction."""
    net = _build_smoke_model()
    inputs = _smoke_inputs(n=1, station_ids=[0], basin_ids=[1])  # unseen station, known basin
    pred = net.predict(inputs, verbose=0)
    assert pred.shape == (1, windowing.DECODER_DAYS, len(windowing.TARGET_FEATURES))
    assert np.isfinite(pred).all()


def test_zero_delta_recovers_persistence_exactly():
    """
    The residual head's worst case is the persistence baseline -- if the
    final Dense weights are zeroed, the network's `delta` is constant zero
    and the Add layer's output equals `persistence_input` exactly.
    """
    net = _build_smoke_model()
    delta_head = net.get_layer('delta_head')
    zeroed = [np.zeros_like(w) for w in delta_head.get_weights()]
    delta_head.set_weights(zeroed)
    persistence = np.tile(np.array([[1.5, 2.5]], dtype='float32'), (3, 1))
    inputs = _smoke_inputs(n=3, persistence=persistence)
    pred = net.predict(inputs, verbose=0)
    expected = np.tile(persistence[:, None, :], (1, windowing.DECODER_DAYS, 1))
    np.testing.assert_allclose(pred, expected, atol=1e-5)


def test_station_dropout_layer_is_wired_into_the_model():
    import model as model_mod
    net = _build_smoke_model()
    sd = net.get_layer('station_dropout')
    assert isinstance(sd, model_mod.StationDropout)
    assert 0.0 <= sd.rate < 1.0
