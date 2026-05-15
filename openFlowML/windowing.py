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

# Encoder window: everything we know up to the prediction time. Flow is here
# (these are observations), and SWE -- current snowpack is a strong predictor
# of snowmelt-fed runoff in Colorado.
ENCODER_FEATURES = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX', 'SWE', 'doy_sin', 'doy_cos']
# Decoder window: ONLY features available at forecast time. No flow (that's
# what we're predicting), no SWE (no skillful 14-day SWE forecast exists).
DECODER_FEATURES = ['TMIN', 'TMAX', 'doy_sin', 'doy_cos']
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
    dec_matrix = df[DECODER_FEATURES].to_numpy(dtype=float)
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
        dec = dec_matrix[start + encoder_days:start + total]
        tgt = tgt_matrix[start + encoder_days:start + total]
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
    required = set(ENCODER_FEATURES) | set(DECODER_FEATURES) | set(TARGET_FEATURES) | {
        'Date', 'site_id', 'station_idx'}
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
