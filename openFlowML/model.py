"""
Seq2seq encoder-decoder for streamflow forecasting.

The architecture mirrors the wiki design and addresses the val_loss explosion
of the previous interim model:

  - tanh / sigmoid LSTM activations (Keras defaults; the old `relu` LSTM was
    the prime suspect for unbounded cell-state growth)
  - station + basin embeddings concatenated to every timestep so the model
    can tell apart sites with wildly different flow regimes -- index 0 is
    reserved for unseen stations / basins so the model can fall back
  - encoder consumes the history window; its final (h, c) state seeds the
    decoder, which then walks the forecast window with NO flow features in
    its input (the no-leakage invariant lives in windowing.py)
  - Huber loss is robust to the occasional flood outlier; Adam with
    `clipnorm=1.0` plus ReduceLROnPlateau keeps optimization stable
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


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
    station_input = layers.Input(shape=(), dtype='int32', name='station_input')
    basin_input = layers.Input(shape=(), dtype='int32', name='basin_input')

    # mask_zero=True so the embedding for index 0 ("unseen") doesn't carry
    # learned information; the model falls back to the basin embedding alone.
    station_emb = layers.Embedding(
        input_dim=num_stations + 1, output_dim=embedding_dim,
        mask_zero=True, name='station_embedding')(station_input)
    basin_emb = layers.Embedding(
        input_dim=num_basins + 1, output_dim=embedding_dim,
        mask_zero=True, name='basin_embedding')(basin_input)
    # Each embedding is (batch, embedding_dim); concat to a per-sample context.
    context = layers.Concatenate(name='station_basin_context')([station_emb, basin_emb])

    # Broadcast the context across every encoder / decoder timestep so the
    # LSTMs see "who is this station, what basin is it in" at every step.
    enc_context = layers.RepeatVector(encoder_days)(context)
    dec_context = layers.RepeatVector(decoder_days)(context)
    enc_input_full = layers.Concatenate(axis=-1)([encoder_input, enc_context])
    dec_input_full = layers.Concatenate(axis=-1)([decoder_input, dec_context])

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

    # Project to (Min Flow, Max Flow) per forecast day in log-z-scored space.
    output = layers.TimeDistributed(
        layers.Dense(target_features, activation='linear'),
        name='output_head',
    )(decoder_seq)

    model = Model(
        inputs={
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'station_input': station_input,
            'basin_input': basin_input,
        },
        outputs=output,
        name='flow_seq2seq',
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=['mae'],
    )
    return model
