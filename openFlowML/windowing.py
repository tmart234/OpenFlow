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
from typing import List, Optional, Tuple

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
#   rain_/snow_{local,mid,far} -- basin-mean daily rain and snowfall (water
#   equivalent) over the station's contributing area, split into river-
#   distance zones (data/basin_zones.py): rain near the gauge arrives within a
#   day, rain on the headwaters days later, and snow is stored rather than
#   run off. The point `precipitation` (nearest rain gauge) stays as a
#   fallback; basin_precip_observed -- 1 when the zone values are real.
BASIN_PRECIP_FEATURES = ['rain_local', 'rain_mid', 'rain_far',
                         'snow_local', 'snow_mid', 'snow_far']
ENERGY_FEATURES = ['srad', 'pet', 'vpd']
INTENSITY_FEATURES = ['precip_max1h', 'precip_max3h', 'precip_wet_hours']
HYDRO_SM_FEATURES = ['sm_surface', 'sm_rootzone']
GRIDDED_SWE_FEATURES = ['swe_low', 'swe_mid', 'swe_high',
                        'delta_swe_mid', 'delta_swe_high']
ENSEMBLE_SPREAD_FEATURES = ['precip_ens_std', 'precip_ens_p90', 'temp_ens_std']

ENCODER_FEATURES = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX', 'precipitation',
                    'SWE', 'soil_moisture', 'sm_observed',
                    'drought_index',
                    'reservoir_storage', 'reservoir_release', 'reservoir_observed',
                    'upstream_flow', 'upstream_observed',
                    *BASIN_PRECIP_FEATURES, 'basin_precip_observed',
                    *ENERGY_FEATURES, 'energy_observed',
                    *INTENSITY_FEATURES, 'precip_intensity_observed',
                    *HYDRO_SM_FEATURES, 'hydro_sm_observed',
                    *GRIDDED_SWE_FEATURES, 'gridded_swe_observed',
                    'prior_flow', 'prior_observed',
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
# not a dataset column) tells the model which days are real vs held.
#
# Basin zone rain/snow are forecast-available too, from NWS gridpoint QPF,
# which reaches ~7 days: decoder days d < BASIN_PRECIP_KNOWN_DAYS carry the
# value, later days are 0.0 (the scaled mean) with basin_precip_known = 0.
#
# Phase 1-3 forecast-available decoder additions:
#   - srad / pet / vpd + energy_known: radiation & evaporative melt energy
#   - precip_max1h / precip_max3h / precip_wet_hours + precip_intensity_known:
#     convective vs stratiform storm intensity (critical for Aug-Oct monsoons)
#   - precip_ens_std / precip_ens_p90 / temp_ens_std + ens_spread_known:
#     weather forecast uncertainty from GEFS reforecasts / Open-Meteo ensemble
#   - forcing_is_forecast (0 = observed weather, 1 = real archived/live forecast)
#     and lead_day_norm ((d + 1) / decoder_days) so the decoder discounts noisy
#     long-lead QPF appropriately
#   - prior_flow / prior_known / prior_change: NWS RFC + NOAA NWM v3 forecast
#     prior capturing dam releases and physical snowmelt/routing guidance
DECODER_FEATURES = ['TMIN', 'TMAX', 'precipitation',
                    'upstream_flow', 'upstream_known', 'upstream_change',
                    *BASIN_PRECIP_FEATURES, 'basin_precip_known',
                    *ENERGY_FEATURES, 'energy_known',
                    *INTENSITY_FEATURES, 'precip_intensity_known',
                    *ENSEMBLE_SPREAD_FEATURES, 'ens_spread_known',
                    'forcing_is_forecast', 'lead_day_norm',
                    'prior_flow', 'prior_known', 'prior_change',
                    'doy_sin', 'doy_cos']
# Decoder features built per window rather than read from a dataset column.
WINDOW_BUILT_DECODER_FEATURES = [
    'upstream_known', 'upstream_change', 'basin_precip_known',
    'energy_known', 'precip_intensity_known',
    *ENSEMBLE_SPREAD_FEATURES, 'ens_spread_known',
    'forcing_is_forecast', 'lead_day_norm',
    'prior_known', 'prior_change',
]
# Phase 1-3 auxiliary dataset columns that default cleanly to 0.0 when absent
# from legacy datasets or minimal synthetic fixtures (Phase 0 columns remain
# strictly required).
OPTIONAL_DATASET_FEATURES = [
    *ENERGY_FEATURES, 'energy_observed',
    *INTENSITY_FEATURES, 'precip_intensity_observed',
    *HYDRO_SM_FEATURES, 'hydro_sm_observed',
    *GRIDDED_SWE_FEATURES, 'gridded_swe_observed',
    'prior_flow', 'prior_observed',
]
BASIN_PRECIP_KNOWN_DAYS = 7
ENERGY_KNOWN_DAYS = 14
PRECIP_INTENSITY_KNOWN_DAYS = 7
PRIOR_KNOWN_DAYS = 10
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


def _inv_scaled(z, params: dict) -> np.ndarray:
    """Invert a single column's scaler params back to non-negative raw units."""
    x = np.asarray(z, dtype=float) * float(params['scale']) + float(params['mean'])
    if params.get('transform') == 'log1p':
        return np.maximum(np.expm1(np.clip(x, -10.0, 20.0)), 0.0)
    return np.maximum(x, 0.0)


def _scale_value(raw: float, params: dict) -> float:
    """Scale a single raw value with persisted scaler params."""
    x = float(raw)
    if params.get('transform') == 'log1p':
        x = float(np.log1p(max(x, 0.0)))
    return (x - float(params['mean'])) / max(float(params['scale']), 1e-8)


def upstream_change(values: np.ndarray, known: np.ndarray, last_value: float,
                    last_observed: float = 1.0, *,
                    hold_beyond_known: bool = True,
                    scalers: Optional[dict] = None,
                    anchor_flow_scaled: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Decoder `upstream_change`: expected log-flow change from routed upstream
    inflow relative to the last encoder day.

    When `scalers` (with 'Min Flow', 'Max Flow', 'upstream_flow') and
    `anchor_flow_scaled` (the last encoder day's scaled [Min Flow, Max Flow])
    are provided, computes the physically consistent additive mass-balance
    change in scaled log-flow units:
        (log1p(max(q0 + u_dec - u0, 0)) - log1p(max(q0, 0))) / flow_scale
    Otherwise falls back to `values - last_value` in scaled space.

    By default (`hold_beyond_known=True`) the last known change is held past
    `known_days` (matching `upstream_decoder_window` and removing the Day K+1
    zero-cliff). Pass `hold_beyond_known=False` only for legacy releases
    trained before this fix.
    """
    values = np.asarray(values, dtype=float)
    known = np.asarray(known, dtype=float) >= 0.5
    if not last_observed or not np.isfinite(last_value):
        return np.zeros_like(values)
    if (scalers and anchor_flow_scaled is not None
            and all(k in scalers for k in ('Min Flow', 'Max Flow', 'upstream_flow'))):
        anchor = np.asarray(anchor_flow_scaled, dtype=float)
        if anchor.size >= 2 and np.isfinite(anchor[:2]).all():
            u_dec = _inv_scaled(values, scalers['upstream_flow'])
            u0 = float(_inv_scaled(float(last_value), scalers['upstream_flow']))
            q0_min = float(_inv_scaled(anchor[0], scalers['Min Flow']))
            q0_max = float(_inv_scaled(anchor[1], scalers['Max Flow']))
            q0_mid = 0.5 * (q0_min + q0_max)
            flow_scale = 0.5 * (float(scalers['Min Flow']['scale'])
                                + float(scalers['Max Flow']['scale']))
            delta = (np.log1p(np.maximum(q0_mid + (u_dec - u0), 0.0))
                     - np.log1p(max(q0_mid, 0.0))) / max(flow_scale, 1e-8)
            return delta if hold_beyond_known else np.where(known, delta, 0.0)
    diff = values - float(last_value)
    return diff if hold_beyond_known else np.where(known, diff, 0.0)


def masked_decoder_window(values: np.ndarray, observed: np.ndarray,
                          known_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic forecast-horizon mask for multi-column decoder feature blocks
    (basin precip, radiation/PET/VPD energy, sub-daily rainfall intensity).
    Day d keeps its values (known = 1) when d < known_days and observed;
    otherwise values become 0.0 (scaled mean) and known = 0.
    """
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=float)
    finite_rows = np.isfinite(values).all(axis=1) if values.ndim == 2 else np.isfinite(values)
    known = (np.arange(len(values)) < known_days) & (observed >= 0.5) & finite_rows
    if values.ndim == 2:
        return np.where(known[:, None], values, 0.0), known.astype(float)
    return np.where(known, values, 0.0), known.astype(float)


def basin_precip_decoder_window(values: np.ndarray, observed: np.ndarray,
                                known_days: int = BASIN_PRECIP_KNOWN_DAYS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decoder-window basin precip inputs for one forecast. values[H, 6] are
    the scaled zone rain/snow per decoder day, observed[H] their indicator.
    Day d keeps its values (known = 1) when d < known_days and observed;
    otherwise the values become 0.0 (scaled mean) and known = 0. The single
    implementation shared by training and inference (docs/INFERENCE.md).
    """
    return masked_decoder_window(values, observed, known_days)


def prior_decoder_window(future_values: np.ndarray,
                         future_observed: np.ndarray,
                         last_value: float,
                         last_observed: float = 0.0,
                         known_days: int = PRIOR_KNOWN_DAYS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decoder-window physics/operator prior (NWS RFC / NOAA NWM v3) inputs:
    returns (prior_flow, prior_known, prior_change). Holds the last known prior
    past `known_days`; `prior_change` is relative to the anchor day's prior
    (or day 0 prior when the anchor day had no prior observation).
    """
    vals, known = upstream_decoder_window(
        future_values, future_observed, known_days, last_value, last_observed)
    if last_observed >= 0.5 and np.isfinite(last_value):
        ref = float(last_value)
        chg = vals - ref
    elif (known >= 0.5).any():
        first_idx = int(np.argmax(known >= 0.5))
        ref = float(vals[first_idx])
        chg = np.where(np.arange(len(vals)) >= first_idx, vals - ref, 0.0)
    else:
        vals = np.zeros_like(vals)
        chg = np.zeros_like(vals)
    return vals, known, chg


def scale_ensemble_spread(precip_ens_std: float, precip_ens_p90: float,
                          temp_ens_std: float, scalers: Optional[dict]) -> Tuple[float, float, float]:
    """
    Scale weather ensemble uncertainty features onto the model's normalized
    coordinate system using the existing precipitation and TMAX scalers.
    Shared by windowing.py (training) and ml_inference.py (live inference).
    """
    p_sc = (scalers or {}).get('precipitation') or {'mean': 0.0, 'scale': 1.0, 'transform': 'log1p'}
    t_sc = (scalers or {}).get('TMAX') or {'mean': 0.0, 'scale': 100.0, 'transform': 'identity'}
    p_scale = max(float(p_sc.get('scale', 1.0)), 1e-6)
    p_mean = float(p_sc.get('mean', 0.0))
    t_scale = max(float(t_sc.get('scale', 100.0)), 1e-6)
    p_std_s = float(np.log1p(max(float(precip_ens_std), 0.0))) / p_scale
    p_p90_s = (float(np.log1p(max(float(precip_ens_p90), 0.0))) - p_mean) / p_scale
    t_std_s = max(float(temp_ens_std), 0.0) / t_scale
    return p_std_s, p_p90_s, t_std_s


def _index_reforecast_for_site(forecast_table: Optional[pd.DataFrame], site_id: str) -> dict:
    """Index a station's archived weather forecasts by (issue_date, lead_day)."""
    if forecast_table is None or forecast_table.empty:
        return {}
    sub = forecast_table[forecast_table['site_id'] == site_id]
    if sub.empty:
        return {}
    out = {}
    for row in sub.to_dict('records'):
        key = (pd.Timestamp(row['issue_date']).normalize(), int(row['lead_day']))
        out[key] = row
    return out


def apply_reforecast_to_window(dec: np.ndarray, issue_date: pd.Timestamp,
                               site_lookup: dict, scalers: Optional[dict],
                               dec_features: List[str] = DECODER_FEATURES) -> np.ndarray:
    """
    Replace decoder weather features on days where an archived real forecast
    exists for `(issue_date, lead_day)` and populate ensemble spread +
    `forcing_is_forecast = 1.0`. Pure helper shared with tests.
    """
    if not site_lookup or not scalers:
        return dec
    issue = pd.Timestamp(issue_date).normalize()
    f_idx = {name: i for i, name in enumerate(dec_features)}
    for d in range(len(dec)):
        row = site_lookup.get((issue, d + 1))
        if row is None:
            continue
        replaced_any = False
        for fc_col, dec_col in (('fc_TMIN', 'TMIN'), ('fc_TMAX', 'TMAX'),
                                ('fc_precipitation', 'precipitation'), ('fc_srad', 'srad')):
            val = row.get(fc_col)
            if (dec_col in f_idx and dec_col in scalers
                    and val is not None and np.isfinite(val)):
                dec[d, f_idx[dec_col]] = _scale_value(float(val), scalers[dec_col])
                replaced_any = True
        if replaced_any and 'forcing_is_forecast' in f_idx:
            dec[d, f_idx['forcing_is_forecast']] = 1.0
        p_std = row.get('precip_ens_std')
        p_p90 = row.get('precip_ens_p90')
        t_std = row.get('temp_ens_std')
        if (all(c in f_idx for c in ENSEMBLE_SPREAD_FEATURES)
                and p_std is not None and p_p90 is not None and t_std is not None
                and np.isfinite(p_std) and np.isfinite(p_p90) and np.isfinite(t_std)):
            s_pstd, s_pp90, s_tstd = scale_ensemble_spread(p_std, p_p90, t_std, scalers)
            dec[d, f_idx['precip_ens_std']] = s_pstd
            dec[d, f_idx['precip_ens_p90']] = s_pp90
            dec[d, f_idx['temp_ens_std']] = s_tstd
            if 'ens_spread_known' in f_idx:
                dec[d, f_idx['ens_spread_known']] = 1.0
    return dec


@dataclass
class SplitWindows:
    train: List[WindowedSample]
    val: List[WindowedSample]
    test: List[WindowedSample]


def _build_windows_for_station(station_df: pd.DataFrame,
                               encoder_days: int,
                               decoder_days: int,
                               scalers: Optional[dict] = None,
                               hold_upstream_beyond_known: bool = True,
                               forecast_table: Optional[pd.DataFrame] = None) -> List[WindowedSample]:
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

    # Default absent Phase 1-3 optional dataset columns to 0.0 (unobserved).
    for col in OPTIONAL_DATASET_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

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
    dec_change_col = DECODER_FEATURES.index('upstream_change')
    bp_cols = [DECODER_FEATURES.index(c) for c in BASIN_PRECIP_FEATURES]
    bp_known_col = DECODER_FEATURES.index('basin_precip_known')
    bp_observed = df['basin_precip_observed'].to_numpy(dtype=float)

    en_cols = [DECODER_FEATURES.index(c) for c in ENERGY_FEATURES]
    en_known_col = DECODER_FEATURES.index('energy_known')
    en_observed = df['energy_observed'].to_numpy(dtype=float)

    int_cols = [DECODER_FEATURES.index(c) for c in INTENSITY_FEATURES]
    int_known_col = DECODER_FEATURES.index('precip_intensity_known')
    int_observed = df['precip_intensity_observed'].to_numpy(dtype=float)

    pr_col = DECODER_FEATURES.index('prior_flow')
    pr_known_col = DECODER_FEATURES.index('prior_known')
    pr_change_col = DECODER_FEATURES.index('prior_change')
    pr_values = df['prior_flow'].to_numpy(dtype=float)
    pr_observed = df['prior_observed'].to_numpy(dtype=float)

    lead_col = DECODER_FEATURES.index('lead_day_norm')
    lead_norm = np.arange(1, decoder_days + 1, dtype=float) / float(decoder_days)

    tgt_matrix = df[TARGET_FEATURES].to_numpy(dtype=float)
    # Indices within ENCODER_FEATURES of the columns that match TARGET_FEATURES,
    # so we can lift the persistence anchor straight from the encoder slice.
    target_indices_in_encoder = [ENCODER_FEATURES.index(c) for c in TARGET_FEATURES]

    station_idx = int(df['station_idx'].iloc[0])
    basin_idx = int(df['basin_idx'].iloc[0]) if 'basin_idx' in df.columns else 0
    site_id = str(df['site_id'].iloc[0])
    site_fc_lookup = _index_reforecast_for_site(forecast_table, site_id)

    for start in range(len(df) - total + 1):
        # The window spans [start, start + total). All internal day-to-day
        # gaps must be exactly 1 day or the window straddles a missing period.
        if day_gaps[start:start + total - 1].max() != 1:
            continue
        enc = enc_matrix[start:start + encoder_days]
        dec = dec_matrix[start + encoder_days:start + total].copy()
        tgt = tgt_matrix[start + encoder_days:start + total]
        last = start + encoder_days - 1
        persistence_anchor = enc[-1, target_indices_in_encoder].astype(float)
        dec[:, dec_up_col], dec[:, dec_known_col] = upstream_decoder_window(
            up_values[start + encoder_days:start + total],
            up_observed[start + encoder_days:start + total],
            known_days, up_values[last], up_observed[last])
        dec[:, dec_change_col] = upstream_change(
            dec[:, dec_up_col], dec[:, dec_known_col], up_values[last], up_observed[last],
            hold_beyond_known=hold_upstream_beyond_known,
            scalers=scalers, anchor_flow_scaled=persistence_anchor)
        dec[:, bp_cols], dec[:, bp_known_col] = basin_precip_decoder_window(
            dec[:, bp_cols], bp_observed[start + encoder_days:start + total])
        dec[:, en_cols], dec[:, en_known_col] = masked_decoder_window(
            dec[:, en_cols], en_observed[start + encoder_days:start + total], ENERGY_KNOWN_DAYS)
        dec[:, int_cols], dec[:, int_known_col] = masked_decoder_window(
            dec[:, int_cols], int_observed[start + encoder_days:start + total], PRECIP_INTENSITY_KNOWN_DAYS)
        dec[:, pr_col], dec[:, pr_known_col], dec[:, pr_change_col] = prior_decoder_window(
            pr_values[start + encoder_days:start + total],
            pr_observed[start + encoder_days:start + total],
            pr_values[last], pr_observed[last], known_days=PRIOR_KNOWN_DAYS)
        dec[:, lead_col] = lead_norm
        if site_fc_lookup:
            dec = apply_reforecast_to_window(dec, pd.Timestamp(dates[last]), site_fc_lookup, scalers)
        if not (np.isfinite(enc).all() and np.isfinite(dec).all() and np.isfinite(tgt).all()):
            continue
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
                  decoder_days: int = DECODER_DAYS,
                  scalers: Optional[dict] = None,
                  hold_upstream_beyond_known: bool = True,
                  forecast_table: Optional[pd.DataFrame] = None) -> List[WindowedSample]:
    """
    Build station-aware (encoder, decoder, target) windows from the normalized
    dataset. The caller is expected to pass the output of normalize_data
    (one row per (station, date), with feature columns scaled and station_idx
    and basin_idx assigned). Passing `scalers` enables physically consistent
    mass-balance `upstream_change` in scaled log-flow space; passing
    `forecast_table` (GEFS / Open-Meteo archived forecasts) injects real
    weather forecasts + ensemble spread into the decoder window.
    """
    required = ((set(ENCODER_FEATURES) | set(DECODER_FEATURES) | set(TARGET_FEATURES) | {
        'Date', 'site_id', 'station_idx'})
        - set(WINDOW_BUILT_DECODER_FEATURES)
        - set(OPTIONAL_DATASET_FEATURES))
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for windowing: {missing}")

    samples: List[WindowedSample] = []
    for site_id, station_df in df.groupby('site_id'):
        station_samples = _build_windows_for_station(
            station_df, encoder_days, decoder_days,
            scalers=scalers, hold_upstream_beyond_known=hold_upstream_beyond_known,
            forecast_table=forecast_table)
        logger.info("Site %s: %d windows", site_id, len(station_samples))
        samples.extend(station_samples)
    return samples


def build_windows_for_config(df: pd.DataFrame,
                             config: Optional[dict] = None,
                             scalers: Optional[dict] = None,
                             forecast_table: Optional[pd.DataFrame] = None) -> List[WindowedSample]:
    """Build windows matching a saved training_config.json (including legacy releases)."""
    cfg = config or {}
    enc_days = int(cfg.get('encoder_days', ENCODER_DAYS))
    dec_days = int(cfg.get('decoder_days', DECODER_DAYS))
    mode = cfg.get('upstream_change_mode', 'legacy')
    if mode == 'mass_balance_hold':
        return build_windows(df, enc_days, dec_days, scalers=scalers,
                             hold_upstream_beyond_known=True, forecast_table=forecast_table)
    if mode == 'scaled_hold':
        return build_windows(df, enc_days, dec_days, scalers=None,
                             hold_upstream_beyond_known=True, forecast_table=forecast_table)
    return build_windows(df, enc_days, dec_days, scalers=None,
                         hold_upstream_beyond_known=False, forecast_table=forecast_table)


def chronological_split(samples: List[WindowedSample],
                        val_frac: float = 0.15,
                        test_frac: float = 0.15,
                        embargo_days: int = DECODER_DAYS) -> SplitWindows:
    """
    Global chronological split across all stations with an embargo gap between
    segments.

    Cutoffs are computed once from the global [min, max] anchor dates across
    all stations so a station with a shorter record cannot leak future years
    into training while longer-record stations are in validation or test.
    The embargo defaults to the forecast horizon, guaranteeing that adjacent
    splits' target windows never overlap.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    train_frac = 1.0 - val_frac - test_frac

    train: List[WindowedSample] = []
    val: List[WindowedSample] = []
    test: List[WindowedSample] = []

    if not samples:
        return SplitWindows(train=train, val=val, test=test)

    all_anchors = [s.anchor_date for s in samples]
    first_date = min(all_anchors)
    last_date = max(all_anchors)
    span = (last_date - first_date).days
    train_end = first_date + pd.Timedelta(days=int(span * train_frac))
    val_start = train_end + pd.Timedelta(days=embargo_days)
    val_end = val_start + pd.Timedelta(days=int(span * val_frac))
    test_start = val_end + pd.Timedelta(days=embargo_days)

    by_station = {}
    for s in samples:
        by_station.setdefault(s.site_id, []).append(s)

    for site_id, station_samples in by_station.items():
        station_samples.sort(key=lambda s: s.anchor_date)
        n_tr = n_va = n_te = 0
        for s in station_samples:
            if s.anchor_date <= train_end:
                train.append(s)
                n_tr += 1
            elif val_start <= s.anchor_date <= val_end:
                val.append(s)
                n_va += 1
            elif s.anchor_date >= test_start:
                test.append(s)
                n_te += 1
            # samples falling into an embargo gap are intentionally dropped
        logger.info("Site %s split: train=%d val=%d test=%d", site_id, n_tr, n_va, n_te)
    return SplitWindows(train=train, val=val, test=test)


def water_year(date) -> int:
    """US water year: Oct 1 of year Y-1 through Sep 30 of year Y is water year Y."""
    date = pd.Timestamp(date)
    return date.year + (date.month >= 10)


def water_year_folds(samples: List[WindowedSample],
                     n_folds: int = 5,
                     encoder_days: int = ENCODER_DAYS,
                     decoder_days: int = DECODER_DAYS) -> List[SplitWindows]:
    """
    Leave-years-out cross-validation splits.

    The water years the samples' issue dates cover are cut into `n_folds`
    contiguous blocks. Fold k tests on block k and validates (early stopping,
    band calibration) on the water year just before it -- just after it for
    the first block. A window goes to test or val when its issue date and
    its last forecast day both fall in those years; it goes to train only
    when no day of it, from the first encoder day to the last forecast day,
    touches them. So every water year is tested exactly once, no window
    shares a day with the years it is scored on, and each fold scores
    several snow seasons instead of the ~2 the chronological split holds.
    """
    def span(s):
        issue = s.anchor_date + pd.Timedelta(days=encoder_days - 1)
        return s.anchor_date, issue, issue + pd.Timedelta(days=decoder_days)

    years = sorted({water_year(span(s)[1]) for s in samples})
    if len(years) < n_folds + 1:
        raise ValueError(f"{len(years)} water years cannot make {n_folds} folds plus a validation year")
    blocks = [list(map(int, b)) for b in np.array_split(years, n_folds)]
    folds = []
    for k, block in enumerate(blocks):
        test_years = set(block)
        val_years = {min(block) - 1} if k > 0 else {max(block) + 1}
        held = test_years | val_years
        train, val, test = [], [], []
        for s in samples:
            start, issue, end = span(s)
            wy_issue, wy_end = water_year(issue), water_year(end)
            if wy_issue in test_years and wy_end in test_years:
                test.append(s)
            elif wy_issue in val_years and wy_end in val_years:
                val.append(s)
            elif water_year(start) not in held and wy_issue not in held and wy_end not in held:
                train.append(s)
        folds.append(SplitWindows(train=train, val=val, test=test))
    return folds


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
