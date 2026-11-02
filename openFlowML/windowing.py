"""
Station-aware window construction for sequence-to-sequence streamflow forecasting.

The single most important invariant this module enforces is the one that broke
the old pipeline: **flow values never appear in the decoder window**. The
encoder sees the full history (flow + temperature + SWE + calendar); the
decoder sees only features available at forecast time (forecast temperature
and the calendar). The target is the flow during the decoder days.

Other invariants this module is responsible for:
  - windows are built PER STATION; a sample never spans two stations
  - rows must be on a regular daily index; a window with any internal NaN
    in any of its features or targets is dropped (gap-aware)
  - each sample carries its anchor date so the chronological train/val/test
    split (with an embargo gap >= horizon) can be done correctly
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Encoder window: everything we know up to the prediction time.
#   flow (observations); SWE -- snowpack drives snowmelt-fed runoff in CO;
#   precipitation -- recent rainfall on the basin, the second-most-important
#   short-horizon driver of stream rise after temperature-driven snowmelt;
#   soil_moisture (SMAP L3 enhanced) -- antecedent wetness gates infiltration
#   vs runoff; sm_observed -- 1 = real / short-gap-interpolated SMAP, 0 =
#   imputed via combine_data's median fallback; drought_index (USDM) -- HUC8
#   drought intensity 0..500; reservoir_storage / reservoir_release (USBR
#   RISE) -- upstream regulation state for regulated rivers, defaults to 0
#   for unmapped (unregulated) stations; reservoir_observed -- 1 when the
#   station actually has reservoir data, 0 when it's the unregulated default.
#   upstream_flow -- routed daily inflow from the station's upstream gauges
#   (data/get_upstream.py: sum_i weight_i * Q_i(d - lag_i)); this is how the
#   model sees a flood wave that is still upstream, which local weather and
#   the station's own history cannot show; upstream_observed -- 1 when the
#   station has upstream gauges and that day's routed value is real, 0 for
#   the placeholder (stations without upstream gauges, data gaps).
ENCODER_FEATURES = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX', 'precipitation',
                    'SWE', 'soil_moisture', 'sm_observed',
                    'drought_index',
                    'reservoir_storage', 'reservoir_release', 'reservoir_observed',
                    'upstream_flow', 'upstream_observed',
                    'doy_sin', 'doy_cos']
# Decoder window: ONLY features available at forecast time. No flow (that's
# what we're predicting), no SWE / no soil_moisture / no drought / no
# reservoir state (none of these have a skillful 14-day forecast available).
# Precipitation IS forecast-available -- Open-Meteo (and the NWS/NOAA QPF
# layer behind it) ships a 14-day daily precipitation_sum that get_forecast.py
# fetches at inference time, so it goes in the decoder window.
#
# Upstream flow is also forecast-available: routed with a travel-time lag, the
# first min(lag_days) decoder days are ALREADY OBSERVED upstream, and the NWS
# River Forecast Center publishes ~5-day flow forecasts at the upstream gauges
# after that. So decoder day d carries the routed upstream value for d <=
# upstream_known_days (per station, see get_upstream.known_days) and HOLDS the
# last known value after that; `upstream_known` (0/1, built per window here,
# not a dataset column) tells the model which days are real vs held. Like the
# weather features, training uses the observed upstream flow as a "perfect"
# forecast for the known days.
DECODER_FEATURES = ['TMIN', 'TMAX', 'precipitation',
                    'upstream_flow', 'upstream_known',
                    'doy_sin', 'doy_cos']
# Decoder features built per window rather than read from a dataset column.
WINDOW_BUILT_DECODER_FEATURES = ['upstream_known']
# Per-station column (unscaled int) giving how many leading decoder days have
# a known upstream value. Missing column == 0 (no upstream gauges).
UPSTREAM_KNOWN_DAYS_COLUMN = 'upstream_known_days'
# Target: log-z-scored flow during the decoder days. Both columns are already
# log1p-transformed and z-scored by normalize_data, so MSE/Huber here behaves.
TARGET_FEATURES = ['Min Flow', 'Max Flow']

ENCODER_DAYS = 60
DECODER_DAYS = 14


@dataclass
class WindowedSample:
    encoder_X: np.ndarray           # (encoder_days, |encoder_features|)
    decoder_X: np.ndarray           # (decoder_days, |decoder_features|)
    target_Y: np.ndarray            # (decoder_days, |target_features|)
    # The last encoder-day flow values, in the same log-z-scored space as the
    # target. The Phase 3 model is wired as a residual forecaster -- it
    # predicts `delta = target - persistence_anchor` and reconstructs
    # `prediction = persistence_anchor + delta` internally, so its worst case
    # is the persistence baseline.
    persistence_anchor: np.ndarray  # (|target_features|,)
    station_idx: int
    basin_idx: int
    site_id: str
    anchor_date: pd.Timestamp       # first day of the encoder window


def upstream_decoder_window(future_values: np.ndarray,
                            future_observed: np.ndarray,
                            known_days: int,
                            last_value: float,
                            last_observed: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decoder-window upstream inputs for one forecast.

    future_values / future_observed: the routed upstream_flow (scaled) and its
    upstream_observed indicator for each decoder day. For day d (0-based):
      - d < known_days and observed  -> use the value, upstream_known = 1
      - otherwise                    -> hold the last known value (starting
                                        from the last encoder day), known = 0
    This is THE rule inference must reproduce (docs/INFERENCE.md); keep it the
    single implementation both training and flow_forecast.py call.
    """
    n = len(future_values)
    values = np.empty(n, dtype=float)
    known = np.zeros(n, dtype=float)
    hold = float(last_value) if last_observed else 0.0
    for d in range(n):
        if d < known_days and future_observed[d] >= 0.5 and np.isfinite(future_values[d]):
            values[d] = future_values[d]
            known[d] = 1.0
            hold = values[d]
        else:
            values[d] = hold
    return values, known


@dataclass
class SplitWindows:
    train: List[WindowedSample]
    val: List[WindowedSample]
    test: List[WindowedSample]


def _build_windows_for_station(station_df: pd.DataFrame,
                               encoder_days: int,
                               decoder_days: int) -> List[WindowedSample]:
    """
    Slide windows over a single station's contiguous daily series.

    The caller is responsible for passing one station's data at a time so this
    function never produces a sample spanning two stations.
    """
    df = station_df.sort_values('Date').reset_index(drop=True)
    total = encoder_days + decoder_days
    samples: List[WindowedSample] = []
    if len(df) < total:
        return samples

    dates = pd.to_datetime(df['Date']).to_numpy()
    # If two consecutive rows are not exactly one day apart the station has a
    # hole; we use this to reject any window straddling that hole.
    day_gaps = np.diff(dates).astype('timedelta64[D]').astype(int)

    enc_matrix = df[ENCODER_FEATURES].to_numpy(dtype=float)
    dec_frame = df.reindex(columns=DECODER_FEATURES)
    for col in WINDOW_BUILT_DECODER_FEATURES:
        dec_frame[col] = 0.0
    dec_matrix = dec_frame.to_numpy(dtype=float)
    up_values = df['upstream_flow'].to_numpy(dtype=float)
    up_observed = df['upstream_observed'].to_numpy(dtype=float)
    known_days = (int(df[UPSTREAM_KNOWN_DAYS_COLUMN].iloc[0])
                  if UPSTREAM_KNOWN_DAYS_COLUMN in df.columns else 0)
    dec_up_col = DECODER_FEATURES.index('upstream_flow')
    dec_known_col = DECODER_FEATURES.index('upstream_known')
    tgt_matrix = df[TARGET_FEATURES].to_numpy(dtype=float)
    # Indices within ENCODER_FEATURES of the columns that match TARGET_FEATURES,
    # so we can lift the persistence anchor straight from the encoder slice.
    target_indices_in_encoder = [ENCODER_FEATURES.index(c) for c in TARGET_FEATURES]

    station_idx = int(df['station_idx'].iloc[0])
    basin_idx = int(df['basin_idx'].iloc[0]) if 'basin_idx' in df.columns else 0
    site_id = str(df['site_id'].iloc[0])

    for start in range(len(df) - total + 1):
        # The window spans [start, start + total). All internal day-to-day
        # gaps must be exactly 1 day or the window straddles a missing period.
        if day_gaps[start:start + total - 1].max() != 1:
            continue
        enc = enc_matrix[start:start + encoder_days]
        dec = dec_matrix[start + encoder_days:start + total].copy()
        tgt = tgt_matrix[start + encoder_days:start + total]
        last = start + encoder_days - 1
        dec[:, dec_up_col], dec[:, dec_known_col] = upstream_decoder_window(
            up_values[start + encoder_days:start + total],
            up_observed[start + encoder_days:start + total],
            known_days, up_values[last], up_observed[last])
        if not (np.isfinite(enc).all() and np.isfinite(dec).all() and np.isfinite(tgt).all()):
            continue
        persistence_anchor = enc[-1, target_indices_in_encoder].astype(float)
        samples.append(WindowedSample(
            encoder_X=enc,
            decoder_X=dec,
            target_Y=tgt,
            persistence_anchor=persistence_anchor,
            station_idx=station_idx,
            basin_idx=basin_idx,
            site_id=site_id,
            anchor_date=pd.Timestamp(dates[start]),
        ))
    return samples


def build_windows(df: pd.DataFrame,
                  encoder_days: int = ENCODER_DAYS,
                  decoder_days: int = DECODER_DAYS) -> List[WindowedSample]:
    """
    Build station-aware (encoder, decoder, target) windows from the normalized
    dataset. The caller is expected to pass the output of normalize_data
    (one row per (station, date), with feature columns scaled and station_idx
    and basin_idx assigned).
    """
    required = (set(ENCODER_FEATURES) | set(DECODER_FEATURES) | set(TARGET_FEATURES) | {
        'Date', 'site_id', 'station_idx'}) - set(WINDOW_BUILT_DECODER_FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for windowing: {missing}")

    samples: List[WindowedSample] = []
    for site_id, station_df in df.groupby('site_id'):
        station_samples = _build_windows_for_station(station_df, encoder_days, decoder_days)
        logger.info("Site %s: %d windows", site_id, len(station_samples))
        samples.extend(station_samples)
    return samples


def chronological_split(samples: List[WindowedSample],
                        val_frac: float = 0.15,
                        test_frac: float = 0.15,
                        embargo_days: int = DECODER_DAYS) -> SplitWindows:
    """
    Per-station chronological split with an embargo gap between segments.

    For each station the earliest 70% of anchor dates become train, then an
    `embargo_days`-wide gap drops samples whose encoder window might overlap
    val, then val, another embargo, then test. The embargo defaults to the
    forecast horizon, which is the minimum gap that guarantees no temporal
    overlap between the input or target windows of adjacent splits.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    train_frac = 1.0 - val_frac - test_frac

    train: List[WindowedSample] = []
    val: List[WindowedSample] = []
    test: List[WindowedSample] = []

    # Group by station so each station contributes chronologically.
    by_station = {}
    for s in samples:
        by_station.setdefault(s.site_id, []).append(s)

    for site_id, station_samples in by_station.items():
        station_samples.sort(key=lambda s: s.anchor_date)
        anchors = [s.anchor_date for s in station_samples]
        if not anchors:
            continue
        span = (anchors[-1] - anchors[0]).days
        train_end = anchors[0] + pd.Timedelta(days=int(span * train_frac))
        val_start = train_end + pd.Timedelta(days=embargo_days)
        val_end = val_start + pd.Timedelta(days=int(span * val_frac))
        test_start = val_end + pd.Timedelta(days=embargo_days)
        for s in station_samples:
            if s.anchor_date <= train_end:
                train.append(s)
            elif val_start <= s.anchor_date <= val_end:
                val.append(s)
            elif s.anchor_date >= test_start:
                test.append(s)
            # samples falling into an embargo gap are intentionally dropped
        logger.info("Site %s split: train=%d val=%d test=%d",
                    site_id, sum(1 for s in train if s.site_id == site_id),
                    sum(1 for s in val if s.site_id == site_id),
                    sum(1 for s in test if s.site_id == site_id))
    return SplitWindows(train=train, val=val, test=test)


def stack(samples: List[WindowedSample]) -> Tuple[dict, np.ndarray]:
    """
    Stack a list of samples into the model-input dict + target tensor expected
    by the encoder-decoder Keras model.
    """
    if not samples:
        empty = np.zeros((0,))
        return ({'encoder_input': empty, 'decoder_input': empty,
                 'persistence_input': empty,
                 'station_input': empty, 'basin_input': empty}, empty)
    inputs = {
        'encoder_input': np.stack([s.encoder_X for s in samples]).astype('float32'),
        'decoder_input': np.stack([s.decoder_X for s in samples]).astype('float32'),
        'persistence_input': np.stack([s.persistence_anchor for s in samples]).astype('float32'),
        'station_input': np.array([s.station_idx for s in samples], dtype='int32'),
        'basin_input': np.array([s.basin_idx for s in samples], dtype='int32'),
    }
    targets = np.stack([s.target_Y for s in samples]).astype('float32')
    return inputs, targets
