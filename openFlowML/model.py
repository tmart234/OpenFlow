"""
Seq2seq encoder-decoder for streamflow forecasting.

Built to address every root cause of the val_loss explosion catalogued in
Phase 3 and the short-horizon underperformance vs the persistence baseline:

  - tanh / sigmoid LSTM activations (Keras defaults; the old `relu` LSTM was
    the prime suspect for unbounded cell-state growth)
  - station + basin embeddings concatenated to every timestep so the model
    can tell apart sites with wildly different flow regimes -- index 0 is
    reserved for unseen stations / basins so the model can fall back
  - `StationDropout`: during training, ~10% of samples have their station_idx
    zeroed before embedding lookup, forcing the model to use the basin
    embedding alone. This is the training side of the wiki's basin-fallback
    design: index 0 is an ordinary, trained "generic station" vector, and
    without this dropout it would never be trained.
  - Encoder reads the history; its (h, c) state seeds a decoder LSTM that
    walks the forecast window with NO flow features in its input (the
    no-leakage invariant lives in windowing.py).
  - Residual learning with a persistence anchor: the model's final output is
    `persistence + delta`, where persistence is the last encoder-day flow
    broadcast across decoder days and delta is what the network actually
    learns. If the network does nothing, prediction = persistence -- so the
    persistence baseline is the worst case, not a competitor.
  - Huber loss is robust to occasional flood outliers; Adam with
    `clipnorm=1.0` plus ReduceLROnPlateau keep optimization stable.
  - Optional probabilistic band (`quantiles=`): a second output of
    non-crossing forecast quantiles (QuantileBand) trained with the pinball
    loss, as in current operational ML flood forecasting (a distributional
    head on the forecast LSTM). The band reads the decoder state through a
    stop-gradient, so it cannot change the point forecast, and its spread is
    calibrated per lead day on the validation split afterwards
    (calibrate_band; split-conformal). The median slot IS the point forecast.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


class StationDropout(layers.Layer):
    """
    Randomly replace station_idx with 0 ("unseen") during training.

    This makes the basin-fallback path a real trained behavior: the model
    sometimes sees a sample where the station embedding is the generic
    "unknown" vector and must rely on the basin embedding to predict. At
    inference time the layer is a no-op (any unseen station is mapped to 0
    by the caller).
    """

    def __init__(self, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0, 1)")
        self.rate = rate

    def call(self, station_input, training=None):
        if not training or self.rate == 0.0:
            return station_input
        keep_mask = tf.cast(
            tf.random.uniform(tf.shape(station_input)) > self.rate,
            station_input.dtype,
        )
        return station_input * keep_mask

    def get_config(self):
        config = super().get_config()
        config['rate'] = self.rate
        return config


# Output names (the mobile apps read the point forecast by this name).
POINT_OUTPUT = 'persistence_plus_delta'
QUANTILE_OUTPUT = 'flow_quantiles'
# Forecast quantiles of the probabilistic band: the 50% and 90% central
# ranges around the median (= the point forecast).
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


class StopGradient(layers.Layer):
    """Identity in the forward pass, no gradient backward."""

    def call(self, inputs):
        return tf.stop_gradient(inputs)


class QuantileBand(layers.Layer):
    """
    Non-crossing forecast quantiles around the point forecast.

    Inputs:
      - legacy 2-input: [point (N, H, T), gaps (N, H, T, Q-1) >= 0]
      - legacy 3-input: [point, gaps, st_scale (N, H, T, 2)]
      - 4-input (Phase 2): [point, gaps, st_scale, persistence_broadcast (N, H, T)]

    The 0.5 slot is the point forecast itself (through a stop-gradient, so the
    band's loss cannot move it). Every other quantile sits a non-negative gap
    beyond its neighbour, so quantiles never cross. Gaps below/above the median
    are multiplied by global `gap_scale[h, t, 0/1]` and per-station `st_scale`.
    When the 4th persistence input is connected, predicted rising limbs
    (`point > persistence`) get an additional upper-tail multiplier
    `rise_upper_scale[h, t]` (>= 1.0) fitted by `calibrate_band` so 90% flood-
    rise coverage reaches nominal levels without widening steady/receding days.
    Uses only elementwise builtins (relu, minimum, add, mul) so CoreML and
    TFLite convert cleanly without Flex ops.
    """

    def __init__(self, quantiles=DEFAULT_QUANTILES, rise_threshold: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = [float(q) for q in quantiles]
        if 0.5 not in self.quantiles or sorted(self.quantiles) != self.quantiles:
            raise ValueError("quantiles must be ascending and include 0.5")
        self.mid = self.quantiles.index(0.5)
        self.rise_threshold = float(rise_threshold)
        self.rise_upper_scale = None

    def build(self, input_shape):
        _, horizon, targets = input_shape[0]
        self.gap_scale = self.add_weight(
            name='gap_scale', shape=(horizon, targets, 2),
            initializer='ones', trainable=False)
        if len(input_shape) == 4:
            self.rise_upper_scale = self.add_weight(
                name='rise_upper_scale', shape=(horizon, targets),
                initializer='ones', trainable=False)
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 4:
            point, gaps, st_scale, persistence = inputs
            lower = self.gap_scale[..., 0] * st_scale[..., 0]
            rise_w = tf.minimum(
                tf.nn.relu(tf.stop_gradient(point) - persistence) / self.rise_threshold,
                1.0,
            )
            rise_mult = 1.0 + rise_w * (self.rise_upper_scale - 1.0)
            upper = self.gap_scale[..., 1] * st_scale[..., 1] * rise_mult
        elif len(inputs) == 3:
            point, gaps, st_scale = inputs
            lower = self.gap_scale[..., 0] * st_scale[..., 0]
            upper = self.gap_scale[..., 1] * st_scale[..., 1]
        else:
            point, gaps = inputs
            lower, upper = self.gap_scale[..., 0], self.gap_scale[..., 1]
        out = {self.mid: tf.stop_gradient(point)}
        for k in range(self.mid - 1, -1, -1):
            out[k] = out[k + 1] - gaps[..., k] * lower
        for k in range(self.mid, len(self.quantiles) - 1):
            out[k + 1] = out[k] + gaps[..., k] * upper
        return tf.stack([out[k] for k in range(len(self.quantiles))], axis=-1)

    def get_config(self):
        config = super().get_config()
        config['quantiles'] = self.quantiles
        config['rise_threshold'] = self.rise_threshold
        return config


class PinballLoss(tf.keras.losses.Loss):
    """Quantile (pinball) loss averaged over the target and quantile axes."""

    def __init__(self, quantiles=DEFAULT_QUANTILES, name='pinball', **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantiles = [float(q) for q in quantiles]

    def call(self, y_true, y_pred):
        q = tf.constant(self.quantiles, dtype=y_pred.dtype)
        err = tf.cast(y_true, y_pred.dtype)[..., None] - y_pred
        return tf.reduce_mean(tf.maximum(q * err, (q - 1.0) * err), axis=[-2, -1])

    def get_config(self):
        config = super().get_config()
        config['quantiles'] = self.quantiles
        return config


class StationUpstreamGain(layers.Layer):
    """
    Per-river response to a change in routed upstream flow.

    Inputs [decoder_input (N, H, F), gain (N, T)]: returns the decoder's
    `column` (upstream_change) times the river's gain, (N, H, T). The gain
    comes from a zero-initialised per-station embedding, so a river whose
    upstream gauges carry nearly all of its flow (Colorado at the state line)
    and one where they carry a fraction (Taos Box below the San Luis Valley)
    each get their own coefficient instead of fighting over the shared one.
    """

    def __init__(self, column: int, **kwargs):
        super().__init__(**kwargs)
        self.column = int(column)

    def call(self, inputs):
        decoder, gain = inputs
        return decoder[..., self.column:self.column + 1] * gain[:, None, :]

    def get_config(self):
        config = super().get_config()
        config['column'] = self.column
        return config


# Everything tf.keras.models.load_model needs for an OpenFlow .h5.
CUSTOM_OBJECTS = {'StationDropout': StationDropout, 'StopGradient': StopGradient,
                  'StationUpstreamGain': StationUpstreamGain,
                  'QuantileBand': QuantileBand, 'PinballLoss': PinballLoss}


def build_encoder_decoder(
    *,
    num_stations: int,
    num_basins: int,
    encoder_features: int,
    decoder_features: int,
    target_features: int,
    encoder_days: int,
    decoder_days: int,
    lstm_units: int = 64,
    embedding_dim: int = 8,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    station_dropout: float = 0.1,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
    huber_delta: float = 1.0,
    linear_skip: bool = True,
    upstream_change_index=None,
    quantiles=None,
    band_units: int = 32,
    static_features: int = 0,
    static_units: int = 32,
) -> Model:
    """
    Build and compile the encoder-decoder model.

    num_stations / num_basins are the SIZES of the vocabularies (not the max
    index). Index 0 is reserved for unknown, so the Embedding input_dim is
    set to num_*+1.

    With `upstream_change_index` (the decoder column of upstream_change) the
    linear skip also gets a per-river gain on that column
    (StationUpstreamGain), zero at the start of training. When
    `static_features > 0` (Phase 5), a 6th `static_input` `(N, S)` passes
    through a compact 2-layer tanh MLP concatenated into the station/basin
    context AND adds an attribute-predicted base upstream gain so unseen
    rivers (`station_input = 0`) get physically informed conditioning and
    mass-balance gain.

    With `quantiles` (ascending, including 0.5; e.g. DEFAULT_QUANTILES) the
    model has two outputs, {POINT_OUTPUT: (N, H, T), QUANTILE_OUTPUT:
    (N, H, T, Q)}; without, the single point output as before.
    """
    encoder_input = layers.Input(shape=(encoder_days, encoder_features), name='encoder_input')
    decoder_input = layers.Input(shape=(decoder_days, decoder_features), name='decoder_input')
    persistence_input = layers.Input(shape=(target_features,), name='persistence_input')
    station_input = layers.Input(shape=(), dtype='int32', name='station_input')
    basin_input = layers.Input(shape=(), dtype='int32', name='basin_input')
    static_input = (layers.Input(shape=(int(static_features),), dtype='float32', name='static_input')
                    if static_features and int(static_features) > 0 else None)

    # Training-time basin-fallback augmentation: randomly zero some station IDs
    # before they reach the embedding so the model practices predicting with
    # only the basin embedding (and static attributes, if enabled) available.
    station_input_masked = StationDropout(rate=station_dropout, name='station_dropout')(station_input)

    # Index 0 ("unseen") is a learned vector like any other, trained by
    # StationDropout. No mask_zero: a Keras mask on these scalar lookups is
    # dropped at RepeatVector and never reached the LSTMs, so it changed
    # nothing except making the graph harder to convert for TFLite.
    station_emb = layers.Embedding(
        input_dim=num_stations + 1, output_dim=embedding_dim,
        name='station_embedding')(station_input_masked)
    basin_emb = layers.Embedding(
        input_dim=num_basins + 1, output_dim=embedding_dim,
        name='basin_embedding')(basin_input)
    static_emb = None
    if static_input is not None:
        static_h = layers.Dense(static_units, activation='tanh', name='static_mlp_hidden')(static_input)
        static_emb = layers.Dense(embedding_dim * 2, activation='tanh', name='static_mlp_out')(static_h)
        context = layers.Concatenate(name='station_basin_context')([station_emb, basin_emb, static_emb])
    else:
        context = layers.Concatenate(name='station_basin_context')([station_emb, basin_emb])

    # Broadcast the context across every encoder / decoder timestep so the
    # LSTMs see "who is this station, what basin is it in" at every step.
    enc_context = layers.RepeatVector(encoder_days)(context)
    dec_context = layers.RepeatVector(decoder_days)(context)
    persistence_broadcast = layers.RepeatVector(decoder_days, name='persistence_broadcast')(persistence_input)
    enc_input_full = layers.Concatenate(axis=-1)([encoder_input, enc_context])
    # The decoder also sees the persistence anchor at every step -- explicit
    # access to "where flow just was" lets it learn near-zero deltas at short
    # horizons and grow corrections at longer horizons.
    decoder_seen = decoder_input
    dec_input_full = layers.Concatenate(axis=-1)([decoder_seen, persistence_broadcast, dec_context])

    # Encoder: read the history, return final cell + hidden state.
    _, enc_state_h, enc_state_c = layers.LSTM(
        lstm_units,
        return_state=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_initializer='glorot_uniform',
        name='encoder_lstm',
    )(enc_input_full)

    # Decoder: walk the forecast window starting from the encoder state.
    decoder_seq = layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_initializer='glorot_uniform',
        name='decoder_lstm',
    )(dec_input_full, initial_state=[enc_state_h, enc_state_c])

    # Project to a per-day delta on (Min Flow, Max Flow) in log-z-scored space.
    delta = layers.TimeDistributed(
        layers.Dense(target_features, activation='linear'),
        name='delta_head',
    )(decoder_seq)

    # Linear skip from the forecast-time inputs straight to the delta, so a
    # direct relationship -- "routed upstream flow rises by X" (decoder
    # upstream_change) or "N mm of rain falls on the local zone" -> "flow
    # here rises" -- is one learned coefficient instead of something the LSTM
    # must discover from few flood windows. Zero-initialised: training starts
    # from the LSTM-only behaviour.
    if linear_skip:
        skip = layers.TimeDistributed(
            layers.Dense(target_features, use_bias=False, kernel_initializer='zeros'),
            name='linear_skip',
        )(decoder_seen)
        delta = layers.Add(name='delta_plus_skip')([delta, skip])
    if upstream_change_index is not None:
        gain = layers.Embedding(
            input_dim=num_stations + 1, output_dim=target_features,
            embeddings_initializer='zeros', name='station_upstream_gain_table',
        )(station_input_masked)
        if static_emb is not None:
            # Attribute-gated base gain + per-station residual so unseen gauged
            # rivers immediately receive a physically informed upstream gain.
            attr_gain = layers.Dense(
                target_features, kernel_initializer='zeros', bias_initializer='zeros',
                name='static_upstream_gain',
            )(static_emb)
            gain = layers.Add(name='combined_upstream_gain')([attr_gain, gain])
        station_skip = StationUpstreamGain(upstream_change_index, name='station_upstream_gain')(
            [decoder_seen, gain])
        delta = layers.Add(name='delta_plus_station_skip')([delta, station_skip])

    # Final output = persistence + delta. Worst-case (delta = 0) reproduces the
    # persistence baseline; any nonzero output is an improvement over it.
    output = layers.Add(name=POINT_OUTPUT)([delta, persistence_broadcast])
    inputs = {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'persistence_input': persistence_input,
        'station_input': station_input,
        'basin_input': basin_input,
    }
    if static_input is not None:
        inputs['static_input'] = static_input
    point_loss = tf.keras.losses.Huber(delta=huber_delta)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    if not quantiles:
        model = Model(inputs=inputs, outputs=output, name='flow_seq2seq_residual')
        model.compile(optimizer=optimizer, loss=point_loss, metrics=['mae'])
        return model

    # Probabilistic band. A small head reads the decoder state (rising limb,
    # snowmelt season, rain in the forecast, upstream known or held ...) and
    # emits the non-negative gaps between adjacent quantiles. Its input is
    # stop-gradiented, so the band never changes the point forecast.
    quantiles = [float(q) for q in quantiles]
    band_hidden = layers.TimeDistributed(
        layers.Dense(band_units, activation='tanh'), name='band_hidden',
    )(StopGradient(name='band_stop_gradient')(decoder_seq))
    band_raw = layers.TimeDistributed(
        layers.Dense(target_features * (len(quantiles) - 1),
                     bias_initializer=tf.keras.initializers.Constant(-2.0)),
        name='band_gaps',
    )(band_hidden)
    gaps = layers.Activation('softplus', name='band_softplus')(
        layers.Reshape((decoder_days, target_features, len(quantiles) - 1))(band_raw))
    st_scale_flat = layers.Embedding(
        input_dim=num_stations + 1, output_dim=decoder_days * target_features * 2,
        embeddings_initializer='ones', trainable=False, name='band_station_scale',
    )(station_input)
    st_scale = layers.Reshape((decoder_days, target_features, 2),
                              name='band_station_scale_reshape')(st_scale_flat)
    band = QuantileBand(quantiles, name=QUANTILE_OUTPUT)([output, gaps, st_scale, persistence_broadcast])

    model = Model(inputs=inputs, outputs={POINT_OUTPUT: output, QUANTILE_OUTPUT: band},
                  name='flow_seq2seq_residual')
    model.compile(
        optimizer=optimizer,
        loss={POINT_OUTPUT: point_loss, QUANTILE_OUTPUT: PinballLoss(quantiles)},
        metrics={POINT_OUTPUT: ['mae']},
    )
    return model


def has_band(net) -> bool:
    """True when `net` has the QUANTILE_OUTPUT (dict outputs)."""
    return isinstance(net.output, dict) and QUANTILE_OUTPUT in net.output


def has_static_input(net) -> bool:
    """True when `net` accepts the Phase 5 `static_input` tensor."""
    return isinstance(net.input, dict) and 'static_input' in net.input


def point_forecast(pred):
    """The (N, H, T) point forecast from a model.predict() result of either kind."""
    return pred[POINT_OUTPUT] if isinstance(pred, dict) else pred


def band_forecast(pred):
    """The (N, H, T, Q) quantiles from a model.predict() result, or None."""
    return pred.get(QUANTILE_OUTPUT) if isinstance(pred, dict) else None


def fit_targets(net, y, w=None):
    """(y, w) in the structure net.fit expects: dicts for a two-output model."""
    if not has_band(net):
        return y, w
    return ({POINT_OUTPUT: y, QUANTILE_OUTPUT: y},
            None if w is None else {POINT_OUTPUT: w, QUANTILE_OUTPUT: w})


def band_layers(net):
    """Every QuantileBand in `net` (one per member of an averaged ensemble)."""
    found = []
    for layer in net.layers:
        if isinstance(layer, QuantileBand):
            found.append(layer)
        elif isinstance(layer, Model):
            found.extend(band_layers(layer))
    return found


def band_station_scale_layers(net):
    """Every per-station conformal multiplier Embedding in `net`."""
    found = []
    for layer in net.layers:
        if layer.name == 'band_station_scale':
            found.append(layer)
        elif isinstance(layer, Model):
            found.extend(band_station_scale_layers(layer))
    return found


def band_calibration(quantiles_pred, point_pred, y, quantiles, lo=0.25, hi=4.0):
    """
    Per-lead, per-target, per-side scale factors (H, T, 2) for the band's gaps
    (normalised split-conformal; Lei et al. 2018, Romano et al. 2019).

    On held-out windows the outer quantiles should cover 1 - q_low below and
    q_high above the median. For each lead day / target,
      lower = (1 - q_low) quantile of (median - y) / (median - q_low_pred)
      upper = q_high       quantile of (y - median) / (q_high_pred - median)
    i.e. the factor that makes the outer quantiles hit their nominal level on
    these windows; every gap on that side scales by it, so the quantiles stay
    ordered. Clipped to [lo, hi].
    """
    q = np.asarray(quantiles_pred, dtype='float64')
    m = np.asarray(point_pred, dtype='float64')
    y = np.asarray(y, dtype='float64')
    r_lo = (m - y) / np.maximum(m - q[..., 0], 1e-6)
    r_hi = (y - m) / np.maximum(q[..., -1] - m, 1e-6)
    lower = np.quantile(r_lo, 1.0 - quantiles[0], axis=0)
    upper = np.quantile(r_hi, quantiles[-1], axis=0)
    return np.clip(np.stack([lower, upper], axis=-1), lo, hi).astype('float32')


def set_band_scale(net, scale):
    """Write global calibration factors (H, T, 2) into every QuantileBand of `net`."""
    for layer in band_layers(net):
        layer.gap_scale.assign(np.asarray(scale, dtype='float32'))


def get_band_station_scale(net):
    """Return the (num_stations + 1, H * T * 2) per-station multiplier matrix, or None."""
    st_layers = band_station_scale_layers(net)
    return st_layers[0].get_weights()[0].copy() if st_layers else None


def set_band_station_scale(net, rel_scale):
    """Write per-station residual calibration multipliers into every member of `net`."""
    arr = np.asarray(rel_scale, dtype='float32')
    for layer in band_station_scale_layers(net):
        layer.set_weights([arr.reshape(layer.input_dim, layer.output_dim)])


def get_rise_upper_scale(net):
    """Return the (H, T) rising-limb upper-tail multiplier matrix, or None."""
    for layer in band_layers(net):
        if getattr(layer, 'rise_upper_scale', None) is not None:
            return layer.rise_upper_scale.numpy().copy()
    return None


def set_rise_upper_scale(net, scale):
    """Write (H, T) rising-limb upper-tail multipliers into every QuantileBand of `net`."""
    arr = np.asarray(scale, dtype='float32')
    for layer in band_layers(net):
        if getattr(layer, 'rise_upper_scale', None) is not None:
            layer.rise_upper_scale.assign(arr)


def calibrate_band(net, inputs, y, quantiles, min_station_windows: int = 50,
                   min_rise_windows: int = 25, rise_upper_max: float = 2.5):
    """
    Calibrate `net`'s band on held-out (inputs, y) and return global factors (H, T, 2).

    1. Fit global per-lead, per-target, per-side split-conformal `gap_scale`.
    2. Fit per-station multipliers (`band_station_scale`) for stations with
       at least `min_station_windows` held-out windows (index 0 stays 1.0).
    3. When `QuantileBand` carries `rise_upper_scale`, fit an upper-tail
       multiplier (>= 1.0) on predicted rising limbs (`point > persistence`)
       so 90% coverage during floods reaches nominal level without widening
       steady or receding days.
    """
    blayers = band_layers(net)
    set_band_scale(net, np.ones_like(blayers[0].gap_scale.numpy()))
    st_layers = band_station_scale_layers(net)
    if st_layers:
        set_band_station_scale(net, np.ones((st_layers[0].input_dim, st_layers[0].output_dim), dtype='float32'))
    has_rise = getattr(blayers[0], 'rise_upper_scale', None) is not None
    if has_rise:
        set_rise_upper_scale(net, np.ones_like(blayers[0].rise_upper_scale.numpy()))
    pred = net.predict(inputs, verbose=0, batch_size=512)
    q_pred, p_pred = band_forecast(pred), point_forecast(pred)
    scale = band_calibration(q_pred, p_pred, y, quantiles)
    set_band_scale(net, scale)
    if st_layers and isinstance(inputs, dict) and 'station_input' in inputs:
        vocab = st_layers[0].input_dim
        rel = np.ones((vocab,) + scale.shape, dtype='float32')
        st = np.asarray(inputs['station_input']).reshape(-1)
        for sid in range(1, vocab):
            m = st == sid
            if int(m.sum()) >= min_station_windows:
                st_scale = band_calibration(q_pred[m], p_pred[m], y[m], quantiles)
                rel[sid] = st_scale / np.maximum(scale, 1e-6)
        set_band_station_scale(net, rel)
    if has_rise and isinstance(inputs, dict) and 'persistence_input' in inputs:
        # Evaluate post-global + post-station band, then calibrate upper tail on
        # predicted rising limbs only.
        pred2 = net.predict(inputs, verbose=0, batch_size=512)
        q2, p2 = band_forecast(pred2), point_forecast(pred2)
        pers = np.asarray(inputs['persistence_input'], dtype='float64')[:, None, :]
        thresh = max(float(getattr(blayers[0], 'rise_threshold', 0.15)), 1e-6)
        rise_w = np.clip((np.asarray(p2, dtype='float64') - pers) / thresh, 0.0, 1.0)
        y_arr = np.asarray(y, dtype='float64')
        r_hi = (y_arr - p2) / np.maximum(q2[..., -1] - p2, 1e-6)
        horizon, targets = scale.shape[0], scale.shape[1]
        rise_scale = np.ones((horizon, targets), dtype='float32')
        for h in range(horizon):
            for t in range(targets):
                active = rise_w[:, h, t] >= 0.3
                if int(active.sum()) >= min_rise_windows:
                    needed = float(np.quantile(r_hi[active, h, t], quantiles[-1]))
                    med_w = float(np.median(rise_w[active, h, t]))
                    mult = 1.0 + (needed - 1.0) / max(med_w, 0.5)
                    rise_scale[h, t] = float(np.clip(mult, 1.0, rise_upper_max))
        set_rise_upper_scale(net, rise_scale)
    return scale


def average_ensemble(members, name: str = 'flow_seq2seq_ensemble') -> Model:
    """
    One Keras model that runs every member on the same inputs and averages
    their outputs (scaled log space, i.e. a geometric mean in cfs). Members
    trained from different random seeds make different small errors;
    averaging removes much of that noise (most visibly on day 1) without
    flattening a peak they agree on. The wrapper keeps the members' input
    names and shapes, so the mobile contract is unchanged.
    """
    members = list(members)
    if len(members) == 1:
        return members[0]
    first = members[0]
    inputs = {k: layers.Input(shape=t.shape[1:], dtype=t.dtype, name=k)
              for k, t in first.input.items()}
    outputs = []
    for i, m in enumerate(members):
        m._name = f'member_{i}'
        outputs.append(m(inputs))
    if isinstance(outputs[0], dict):
        # Quantiles average member by member (quantile averaging keeps them
        # ordered); the point output keeps the 'ensemble_mean' layer.
        merged = {k: layers.Average(name='ensemble_mean' if k == POINT_OUTPUT else f'ensemble_{k}')(
            [o[k] for o in outputs]) for k in outputs[0]}
        model = Model(inputs=inputs, outputs=merged, name=name)
        model.compile(optimizer=first.optimizer.__class__(), loss=first.loss,
                      metrics={POINT_OUTPUT: ['mae']})
        return model
    model = Model(inputs=inputs, outputs=layers.Average(name='ensemble_mean')(outputs), name=name)
    model.compile(optimizer=first.optimizer.__class__(), loss=first.loss, metrics=['mae'])
    return model
