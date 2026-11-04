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


def test_linear_skip_head_starts_at_zero_and_can_be_disabled():
    net = _build_smoke_model()
    skip = net.get_layer('linear_skip')
    assert all((w == 0).all() for w in skip.get_weights())
    import model as model_mod
    plain = model_mod.build_encoder_decoder(
        num_stations=3, num_basins=2, encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES), encoder_days=windowing.ENCODER_DAYS,
        decoder_days=windowing.DECODER_DAYS, linear_skip=False)
    assert 'linear_skip' not in [l.name for l in plain.layers]


def test_average_ensemble_is_the_mean_of_members_and_round_trips(tmp_path):
    import tensorflow as tf
    a, b = _build_smoke_model(), _build_smoke_model()
    ens = model.average_ensemble([a, b])
    x = _smoke_inputs()
    want = (a.predict(x, verbose=0) + b.predict(x, verbose=0)) / 2
    np.testing.assert_allclose(ens.predict(x, verbose=0), want, rtol=1e-5, atol=1e-6)
    assert sorted(ens.input.keys()) == sorted(a.input.keys())
    path = str(tmp_path / 'ens.h5')
    ens.save(path)
    loaded = tf.keras.models.load_model(path, custom_objects={'StationDropout': model.StationDropout},
                                        compile=False)
    np.testing.assert_allclose(loaded.predict(x, verbose=0), want, rtol=1e-5, atol=1e-6)
    assert model.average_ensemble([a]) is a


def test_export_rebuild_without_mask_handles_an_ensemble():
    import export_mobile
    cfg = dict(num_stations=3, num_basins=3, encoder_features=windowing.ENCODER_FEATURES,
               decoder_features=windowing.DECODER_FEATURES, target_features=windowing.TARGET_FEATURES,
               encoder_days=windowing.ENCODER_DAYS, decoder_days=windowing.DECODER_DAYS)
    members = [model.build_encoder_decoder(
        num_stations=3, num_basins=3, encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES), target_features=len(windowing.TARGET_FEATURES),
        encoder_days=windowing.ENCODER_DAYS, decoder_days=windowing.DECODER_DAYS) for _ in range(2)]
    ens = model.average_ensemble(members)
    rebuilt = export_mobile._rebuild_without_mask_zero(ens, cfg)
    x = export_mobile._random_sample_inputs(cfg, n=3)
    x['station_input'][:] = 1
    x['basin_input'][:] = 1
    np.testing.assert_allclose(rebuilt.predict(x, verbose=0), ens.predict(x, verbose=0), atol=1e-6)


def _build_band_model():
    return model.build_encoder_decoder(
        num_stations=4, num_basins=3,
        encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES),
        encoder_days=windowing.ENCODER_DAYS, decoder_days=windowing.DECODER_DAYS,
        lstm_units=8, embedding_dim=4, quantiles=model.DEFAULT_QUANTILES)


def test_band_quantiles_are_ordered_and_centred_on_the_point_forecast():
    net = _build_band_model()
    assert model.has_band(net) and not model.has_band(_build_smoke_model())
    rng = np.random.default_rng(3)
    y = rng.standard_normal((8, windowing.DECODER_DAYS, 2)).astype('float32')
    x = _smoke_inputs(n=8)
    targets, weights = model.fit_targets(net, y, np.ones((8, windowing.DECODER_DAYS), 'float32'))
    hist = net.fit(x, targets, sample_weight=weights, epochs=1, batch_size=4, verbose=0)
    assert np.isfinite(hist.history['loss'][-1])
    pred = net.predict(x, verbose=0)
    point, band = model.point_forecast(pred), model.band_forecast(pred)
    assert band.shape == (8, windowing.DECODER_DAYS, 2, len(model.DEFAULT_QUANTILES))
    np.testing.assert_allclose(band[..., model.DEFAULT_QUANTILES.index(0.5)], point, atol=1e-6)
    assert (np.diff(band, axis=-1) >= 0).all()


def test_band_loss_cannot_move_the_point_forecast():
    """The band reads the decoder through a stop-gradient: zero gradient into the point path."""
    import tensorflow as tf
    net = _build_band_model()
    x = _smoke_inputs(n=4)
    y = np.zeros((4, windowing.DECODER_DAYS, 2), 'float32')
    with tf.GradientTape() as tape:
        out = net(x, training=True)
        loss = model.PinballLoss(model.DEFAULT_QUANTILES)(y, out[model.QUANTILE_OUTPUT])
    band_vars = {v.ref() for name in ('band_hidden', 'band_gaps')
                 for v in net.get_layer(name).trainable_variables}
    grads = tape.gradient(loss, net.trainable_variables)
    for var, grad in zip(net.trainable_variables, grads):
        if var.ref() in band_vars:
            assert grad is not None
        else:
            assert grad is None or float(tf.reduce_max(tf.abs(tf.convert_to_tensor(grad)))) == 0.0, var.name


def test_band_calibration_hits_the_nominal_levels_on_its_own_windows():
    rng = np.random.default_rng(4)
    m = np.zeros((4000, 3, 1))
    y = rng.normal(0.0, 2.0, m.shape)                      # truth spreads 2x the raw band
    raw = m[..., None] + np.array([-1.645, -0.674, 0.0, 0.674, 1.645])  # N(0, 1) quantiles
    scale = model.band_calibration(raw, m, y, model.DEFAULT_QUANTILES)
    assert scale.shape == (3, 1, 2)
    np.testing.assert_allclose(scale, 2.0, rtol=0.1)


def test_calibrated_band_ensemble_round_trips_through_h5(tmp_path):
    import tensorflow as tf
    ens = model.average_ensemble([_build_band_model(), _build_band_model()])
    x = _smoke_inputs(n=6)
    y = np.random.default_rng(5).standard_normal((6, windowing.DECODER_DAYS, 2)).astype('float32')
    scale = model.calibrate_band(ens, x, y, model.DEFAULT_QUANTILES)
    assert len(model.band_layers(ens)) == 2
    want = ens.predict(x, verbose=0)
    assert ens.get_layer('ensemble_mean') is not None
    np.testing.assert_allclose(want[model.QUANTILE_OUTPUT][..., 2], want[model.POINT_OUTPUT], atol=1e-6)
    path = str(tmp_path / 'band.h5')
    ens.save(path)
    loaded = tf.keras.models.load_model(path, custom_objects=model.CUSTOM_OBJECTS, compile=False)
    got = loaded.predict(x, verbose=0)
    for key in (model.POINT_OUTPUT, model.QUANTILE_OUTPUT):
        np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.band_layers(loaded)[0].gap_scale.numpy(), scale)
