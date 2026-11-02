import os
import json
import logging

import windowing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
Feature engineering and normalization for the combined training dataset.

Given a per-station, daily-indexed frame from combine_data this produces:
  - cyclical day-of-year features (doy_sin, doy_cos) that are continuous across
    year boundaries and leap-year aware
  - an integer station index (station_idx); index 0 is reserved for unseen
    stations so the Phase 3 embedding layer falls back to a generic vector
  - an integer basin index (basin_idx) keyed by HUC8, also with 0 reserved
  - LOG-transformed flow columns (log1p) before z-scoring -- streamflow is
    log-normal, so this prevents floods from dominating the loss and lets the
    model spread error across base/low/peak flow regimes
  - z-score normalized numeric columns. Scaler parameters AND the
    log-transform flag are persisted (scalers.json) so inference can reproduce
    the exact transform and invert the flow predictions correctly. The station
    + basin maps are persisted alongside.
"""

logger = logging.getLogger(__name__)

# Required: rows missing any of these are dropped (no pooled-mean fill).
CORE_REQUIRED = ['TMIN', 'TMAX', 'Min Flow', 'Max Flow']
# Optional continuous columns that get scaled when present. SWE, soil_moisture,
# drought_index, the two reservoir columns, and precipitation are all
# combine_data imputes when missing rather than dropping the row. precipitation
# is treated similarly to flow (log1p before z-score) -- daily precip is
# heavy-tailed and the same rationale that justifies log1p for streamflow
# (spread error across base/peak regimes) applies here.
BASIN_PRECIP_COLUMNS = list(windowing.BASIN_PRECIP_FEATURES)
OPTIONAL_NUMERIC = ['SWE', 'soil_moisture', 'drought_index',
                    'reservoir_storage', 'reservoir_release',
                    'precipitation', 'upstream_flow', *BASIN_PRECIP_COLUMNS]
# Binary indicator columns: included as model inputs but NOT z-scored. Scaling
# a 0/1 indicator destroys the semantics (the model needs to see 0 vs 1 as
# distinct cases, not as samples from a centered normal).
#   sm_observed         -- 1 = real / short-gap-interpolated SMAP retrieval
#   reservoir_observed  -- 1 = station has a USBR mapping AND fetch succeeded
#   upstream_observed   -- 1 = station has upstream gauges AND the routed value is real
#   basin_precip_observed -- 1 = station has basin zones AND gridMET had the day
INDICATOR_COLUMNS = ['sm_observed', 'reservoir_observed', 'upstream_observed',
                     'basin_precip_observed']
# Columns whose scaler is fit ONLY on rows where the paired indicator is 1,
# and whose unobserved rows are set to 0.0 (the observed mean) after
# scaling. Upstream flow is 0 cfs for every station without upstream gauges;
# fitting the scaler on those placeholders would make the real values'
# scale meaningless, and log1p(0) would put the placeholder far out in the
# tail. The indicator column tells the model the placeholder isn't real.
SCALER_FIT_MASKS = {'upstream_flow': 'upstream_observed',
                    **{c: 'basin_precip_observed' for c in BASIN_PRECIP_COLUMNS}}
# Per-station metadata ints passed through unscaled (not model features).
PASSTHROUGH_INT_COLUMNS = ['upstream_known_days']
NUMERIC_COLUMNS = CORE_REQUIRED + OPTIONAL_NUMERIC
# Streamflow is log-normal -- log1p before z-scoring is standard hydrology
# practice. Temperature/SWE stay linear. Precipitation is also heavy-tailed
# (most days zero, occasional storms in the 99th percentile dominate the
# raw scale), so it gets the same log1p treatment as flow.
LOG_TRANSFORM_COLUMNS = {'Min Flow', 'Max Flow', 'precipitation', 'upstream_flow',
                         *BASIN_PRECIP_COLUMNS}


def add_day_of_year_features(data):
    """
    Add continuous, leap-year-aware cyclical day-of-year features.

    The angle wraps smoothly through the year boundary, so Dec 31 and Jan 1 are
    adjacent in feature space (unlike a 0->1 ramp).
    """
    dates = pd.to_datetime(data['Date'])
    day_of_year = dates.dt.dayofyear
    days_in_year = 365 + dates.dt.is_leap_year.astype(int)
    angle = 2.0 * np.pi * (day_of_year - 1) / days_in_year
    data['doy_sin'] = np.sin(angle)
    data['doy_cos'] = np.cos(angle)
    return data


def build_station_index(site_ids):
    """
    Map each station id string to a stable integer index.

    Index 0 is reserved for unseen stations so the embedding can fall back to
    a generic representation (basin fallback path).
    """
    unique_ids = sorted(pd.unique(pd.Series(site_ids).dropna()))
    return {site_id: idx + 1 for idx, site_id in enumerate(unique_ids)}


def build_basin_index(huc8_ids):
    """
    Map each HUC8 string to a stable integer index. 0 reserved for unknown.

    Empty strings (when the HUC lookup failed) collapse to the unknown bucket.
    """
    series = pd.Series(huc8_ids).dropna()
    series = series[series.astype(str).str.len() > 0]
    unique_ids = sorted(pd.unique(series.astype(str)))
    return {huc: idx + 1 for idx, huc in enumerate(unique_ids)}


def fit_scalers(data, columns, log_columns=LOG_TRANSFORM_COLUMNS, fit_masks=None):
    """
    Fit a StandardScaler per column and return its parameters.

    For columns named in `log_columns`, we fit on log1p-transformed values --
    streamflow is log-normal, and using log space is the standard hydrology
    move. The returned record carries a `transform` flag so apply/invert can
    reproduce it.

    `fit_masks` maps column -> indicator column; when given (and the indicator
    has any 1s) the scaler is fit only on rows where the indicator is 1.
    """
    fit_masks = fit_masks or {}
    scalers = {}
    for column in columns:
        frame = data
        indicator = fit_masks.get(column)
        if indicator and indicator in data.columns and (data[indicator] == 1).any():
            frame = data[data[indicator] == 1]
        values = frame[[column]].astype('float64').values
        transform = 'log1p' if column in log_columns else 'identity'
        if transform == 'log1p':
            values = np.log1p(values)
        scaler = StandardScaler()
        scaler.fit(values)
        scalers[column] = {
            'mean': float(scaler.mean_[0]),
            'scale': float(scaler.scale_[0]),
            'transform': transform,
        }
    return scalers


def apply_scalers(data, scalers):
    """
    Apply persisted scaler parameters (including any log1p transform) in place.

    Inverse: x_raw = expm1(z * scale + mean) for log1p columns,
             x_raw = z * scale + mean        otherwise.
    """
    for column, params in scalers.items():
        if column not in data.columns:
            continue
        values = data[column].astype('float64').values
        if params.get('transform', 'identity') == 'log1p':
            values = np.log1p(values)
        data[column] = (values - params['mean']) / params['scale']
    return data


def normalize_data(data, artifacts_dir=None):
    """
    Feature-engineer and normalize the combined dataset.

    Args:
        data: per-station daily-indexed DataFrame from combine_data with
              columns Date, site_id, huc8, TMIN, TMAX, Min Flow, Max Flow,
              and optionally SWE.
        artifacts_dir: if given, scalers.json, station_index.json, and
              basin_index.json are written here so inference can reproduce the
              full transform pipeline.

    Returns the normalized DataFrame. Date, site_id, huc8, station_idx, and
    basin_idx are all preserved so Phase 3 can do per-station chronological
    splitting and station/basin embedding lookup.
    """
    try:
        data = data.copy()

        missing_core = [c for c in CORE_REQUIRED if c not in data.columns]
        if missing_core:
            raise ValueError(f"Missing required columns in the data: {missing_core}")

        # Coerce every numeric column we know about, including SWE if present,
        # plus the binary indicator columns (still numeric, just not scaled).
        present_numeric = [c for c in NUMERIC_COLUMNS if c in data.columns]
        present_indicator = [c for c in INDICATOR_COLUMNS if c in data.columns]
        for column in present_numeric + present_indicator:
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # All optional numerics are slow-varying auxiliaries; combine_data
        # imputes them with smarter per-source logic. The safety net here
        # defaults any remaining missing value to 0 instead of dropping the
        # row, so a station with no nearby SNOTEL / no SMAP retrieval /
        # outside USDM coverage / no upstream reservoir / no NCEI precip
        # still contributes. Precipitation 0 = "no rain" is a legitimate
        # default (unlike soil moisture, which gets median imputation in
        # combine_data); the model can treat "no observation today" the
        # same as "no rain today" without much harm.
        for column in ('SWE', 'soil_moisture', 'drought_index',
                       'reservoir_storage', 'reservoir_release',
                       'precipitation', 'upstream_flow', *BASIN_PRECIP_COLUMNS):
            if column in data.columns:
                data[column] = data[column].fillna(0.0)
        # Indicators default to 0 ("not observed") when missing.
        for column in present_indicator:
            data[column] = data[column].fillna(0).astype('int64')
        for column in PASSTHROUGH_INT_COLUMNS:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0).astype('int64')

        # combine_data owns per-station gap handling for flow + temperature;
        # anything still missing in the core columns here is dropped rather
        # than filled with a pooled (cross-station) mean.
        before = len(data)
        data = data.dropna(subset=CORE_REQUIRED).reset_index(drop=True)
        if len(data) < before:
            logger.warning("Dropped %d rows with missing core values", before - len(data))

        data = add_day_of_year_features(data)

        # Station embedding key.
        if 'site_id' in data.columns:
            station_index = build_station_index(data['site_id'])
            data['station_idx'] = data['site_id'].map(station_index).fillna(0).astype(int)
        else:
            station_index = {}
            logger.warning("No 'site_id' column found; station_idx not added")

        # Basin embedding key (HUC8 -> int).
        if 'huc8' in data.columns:
            basin_index = build_basin_index(data['huc8'])
            data['basin_idx'] = data['huc8'].astype(str).map(basin_index).fillna(0).astype(int)
        else:
            basin_index = {}
            logger.warning("No 'huc8' column found; basin_idx not added")

        scalers = fit_scalers(data, present_numeric, fit_masks=SCALER_FIT_MASKS)
        data = apply_scalers(data, scalers)
        for column, indicator in SCALER_FIT_MASKS.items():
            if column in data.columns and indicator in data.columns:
                data.loc[data[indicator] == 0, column] = 0.0

        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            with open(os.path.join(artifacts_dir, 'scalers.json'), 'w') as f:
                json.dump(scalers, f, indent=2)
            with open(os.path.join(artifacts_dir, 'station_index.json'), 'w') as f:
                json.dump(station_index, f, indent=2)
            with open(os.path.join(artifacts_dir, 'basin_index.json'), 'w') as f:
                json.dump(basin_index, f, indent=2)
            logger.info("Wrote scalers/station/basin index JSON to %s", artifacts_dir)

        return data
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return None


if __name__ == "__main__":
    df = pd.read_csv('combined_data_all_sites.csv')
    normalized_data = normalize_data(df, artifacts_dir='.')
