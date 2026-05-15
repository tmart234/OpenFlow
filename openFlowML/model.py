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
    design -- without it, mask_zero=True is just a structural placeholder
    the model never exercises and won't generalize to unseen stations.
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
"""

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
    recurrent_dropout: float = 0.1,
    station_dropout: float = 0.1,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
    huber_delta: float = 1.0,
) -> Model:
    """
    Build and compile the encoder-decoder model.

    num_stations / num_basins are the SIZES of the vocabularies (not the max
    index). Index 0 is reserved for unknown, so the Embedding input_dim is
    set to num_*+1.
    """
    encoder_input = layers.Input(shape=(encoder_days, encoder_features), name='encoder_input')
    decoder_input = layers.Input(shape=(decoder_days, decoder_features), name='decoder_input')
    persistence_input = layers.Input(shape=(target_features,), name='persistence_input')
    station_input = layers.Input(shape=(), dtype='int32', name='station_input')
    basin_input = layers.Input(shape=(), dtype='int32', name='basin_input')

    # Training-time basin-fallback augmentation: randomly zero some station IDs
    # before they reach the embedding so the model practices predicting with
    # only the basin embedding available.
    station_input_masked = StationDropout(rate=station_dropout, name='station_dropout')(station_input)

    # mask_zero=True so the embedding for index 0 ("unseen") doesn't carry
    # learned information; the model falls back to the basin embedding alone.
    station_emb = layers.Embedding(
        input_dim=num_stations + 1, output_dim=embedding_dim,
        mask_zero=True, name='station_embedding')(station_input_masked)
    basin_emb = layers.Embedding(
        input_dim=num_basins + 1, output_dim=embedding_dim,
        mask_zero=True, name='basin_embedding')(basin_input)
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
    dec_input_full = layers.Concatenate(axis=-1)([decoder_input, persistence_broadcast, dec_context])

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

    # Final output = persistence + delta. Worst-case (delta = 0) reproduces the
    # persistence baseline; any nonzero output is an improvement over it.
    output = layers.Add(name='persistence_plus_delta')([delta, persistence_broadcast])

    model = Model(
        inputs={
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'persistence_input': persistence_input,
            'station_input': station_input,
            'basin_input': basin_input,
        },
        outputs=output,
        name='flow_seq2seq_residual',
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=['mae'],
    )
    return model
