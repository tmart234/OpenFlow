import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
Feature engineering and normalization for the combined training dataset.

Given a per-station, daily-indexed frame from combine_data, this produces:
  - cyclical day-of-year features (doy_sin, doy_cos) that are continuous across
    year boundaries and leap-year aware -- replacing the old monotonic
    date_normalized ramp which had a discontinuity every Dec 31 -> Jan 1
  - an integer station index (station_idx); index 0 is reserved for unknown /
    unseen stations so a future embedding layer can fall back gracefully
  - z-score normalized numeric columns, with the fitted scaler parameters
    persisted (scalers.json) so inference can reproduce the exact transform and
    invert the flow predictions, plus the station mapping (station_index.json)
"""

logger = logging.getLogger(__name__)

# Required: rows missing any of these are dropped (no pooled-mean fill).
CORE_REQUIRED = ['TMIN', 'TMAX', 'Min Flow', 'Max Flow']
# Optional columns that get scaled when present. SWE is slow-varying; if it's
# missing for a row, default it to 0 rather than dropping the row.
OPTIONAL_NUMERIC = ['SWE']
NUMERIC_COLUMNS = CORE_REQUIRED + OPTIONAL_NUMERIC


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

    Index 0 is reserved for unknown/unseen stations so that the embedding layer
    planned for Phase 3 can fall back to a generic representation.
    """
    unique_ids = sorted(pd.unique(pd.Series(site_ids).dropna()))
    return {site_id: idx + 1 for idx, site_id in enumerate(unique_ids)}


def fit_scalers(data, columns):
    """Fit a StandardScaler per column; return {column: {mean, scale}}."""
    scalers = {}
    for column in columns:
        scaler = StandardScaler()
        scaler.fit(data[[column]].astype('float64').values)
        scalers[column] = {
            'mean': float(scaler.mean_[0]),
            'scale': float(scaler.scale_[0]),
        }
    return scalers


def apply_scalers(data, scalers):
    """Apply persisted scaler parameters in place; returns the frame."""
    for column, params in scalers.items():
        if column in data.columns:
            data[column] = (data[column].astype('float64') - params['mean']) / params['scale']
    return data


def normalize_data(data, artifacts_dir=None):
    """
    Feature-engineer and normalize the combined dataset.

    Args:
        data: per-station daily-indexed DataFrame from combine_data, with
              columns Date, site_id, TMIN, TMAX, Min Flow, Max Flow.
        artifacts_dir: if given, scalers.json and station_index.json are written
              here so inference can reproduce the transform.

    Returns the normalized DataFrame. Date and site_id are intentionally
    preserved so Phase 3 can do chronological / per-station splitting.
    """
    try:
        data = data.copy()

        missing_core = [c for c in CORE_REQUIRED if c not in data.columns]
        if missing_core:
            raise ValueError(f"Missing required columns in the data: {missing_core}")

        # Coerce every numeric column we know about, including SWE if present.
        present_numeric = [c for c in NUMERIC_COLUMNS if c in data.columns]
        for column in present_numeric:
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # SWE is slowly varying and often legitimately zero -- any remaining
        # missing value here defaults to 0 ("no snow data") instead of forcing
        # the row out, so a station with no nearby SNOTEL still contributes.
        if 'SWE' in data.columns:
            data['SWE'] = data['SWE'].fillna(0.0)

        # combine_data owns per-station gap handling for flow + temperature;
        # anything still missing in the core columns here is dropped rather
        # than filled with a pooled (cross-station) mean.
        before = len(data)
        data = data.dropna(subset=CORE_REQUIRED).reset_index(drop=True)
        if len(data) < before:
            logger.warning("Dropped %d rows with missing core values", before - len(data))

        data = add_day_of_year_features(data)

        if 'site_id' in data.columns:
            station_index = build_station_index(data['site_id'])
            data['station_idx'] = data['site_id'].map(station_index).fillna(0).astype(int)
        else:
            station_index = {}
            logger.warning("No 'site_id' column found; station_idx not added")

        scalers = fit_scalers(data, present_numeric)
        data = apply_scalers(data, scalers)

        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            with open(os.path.join(artifacts_dir, 'scalers.json'), 'w') as f:
                json.dump(scalers, f, indent=2)
            with open(os.path.join(artifacts_dir, 'station_index.json'), 'w') as f:
                json.dump(station_index, f, indent=2)
            logger.info("Wrote scalers.json and station_index.json to %s", artifacts_dir)

        return data
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return None


if __name__ == "__main__":
    df = pd.read_csv('combined_data_all_sites.csv')
    normalized_data = normalize_data(df, artifacts_dir='.')
