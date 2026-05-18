import os
from datetime import datetime, timedelta
import data.get_flow as get_flow
import data.get_CODWR_flow as get_CODWR_flow
import data.get_noaa as get_noaa
import data.get_swe as get_swe
import normalize_data
import pandas as pd
import logging
from data.utils import data_utils
from data.utils.get_coordinates import get_usgs_coordinates, get_dwr_coordinates

"""
Combines the individual data components (flow + NOAA temperature) for each
monitoring site into a single training dataset.

Phase 2 data-spine guarantees:
  - every site is placed on a regular daily index over the training window
  - short interior gaps are interpolated time-aware and PER STATION; longer
    gaps are dropped rather than filled with a pooled (cross-station) mean
  - the per-station frames carry a clean 'site_id' column

Phase 4 (SMAP soil moisture):
  - Soil moisture is fetched per station from NASA SMAP L3 enhanced via
    nasa_moisture (lazy import; failure degrades to "no SMAP" for that site).
  - Treated like SWE in the spine: slow-varying, longer interior interpolation
    limit, missing rows default to 0 rather than being dropped.
  - Set OPENFLOW_DISABLE_SMAP=1 to run the ablation baseline without SMAP.
"""

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Numeric columns that must be present and gap-free in the final per-site frame.
CORE_COLUMNS = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX']
# Longest interior gap (in days) we are willing to interpolate across for flow
# and temperature, which can change quickly day-to-day.
MAX_GAP_DAYS = 7
# Precipitation is far less amenable to interpolation than temperature: most
# days are dry, storms are spike events, and "average two dry days around a
# storm to fill the storm day" is meaningfully wrong. Use a very short interior
# interpolation limit (carries through ~2-day missing-data sensor outages) and
# then default to 0 ("no rain") for anything still missing.
MAX_PRECIP_GAP_DAYS = 2
# SWE changes slowly (snowpack accumulates/melts over weeks), so we tolerate
# longer interior gaps in the SWE series before giving up on a value.
MAX_SWE_GAP_DAYS = 30
# SMAP has a ~1-3 day revisit cadence per pass; gaps come from RFI / dense
# vegetation / frozen ground. Soil moisture itself is slow-varying so the same
# generous interior interpolation limit as SWE is appropriate.
MAX_SM_GAP_DAYS = 30
# USDM is published weekly; ffill across 8-day gaps is the natural cadence
# for ffill-only handling. Drought intensity is even slower-varying than SWE.
MAX_DROUGHT_GAP_DAYS = 14
# Reservoir storage / release are reported daily by USBR; gaps are short
# (occasional missing days), so the regular MAX_GAP_DAYS interpolation limit
# is enough. Reservoir state is also slow-varying.
MAX_RESERVOIR_GAP_DAYS = 14


def _to_daily_series(df, value_columns, daily_index):
    """Index `df` by Date and reindex onto a regular daily DatetimeIndex."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    present = [c for c in value_columns if c in df.columns]
    if not present:
        return pd.DataFrame(index=daily_index, columns=value_columns, dtype='float64')
    # Collapse any accidental duplicate dates before reindexing.
    daily = df.groupby('Date')[present].mean()
    return daily.reindex(daily_index)


def merge_dataframes(noaa_data, flow_data, site_id, start_date, end_date,
                     swe_data=None, sm_data=None,
                     drought_data=None, reservoir_data=None,
                     huc8=None):
    """
    Merge a site's flow, NOAA temperature, SWE, SMAP soil moisture, USDM
    drought, and USBR reservoir series onto one regular daily index.

    Short interior flow/temp gaps (<= MAX_GAP_DAYS) are interpolated time-aware;
    rows still missing a core value afterwards are dropped (no pooled-mean
    fill). All Phase 4/5 auxiliary features (SWE, soil_moisture, drought_index,
    reservoir_storage, reservoir_release) are slow-varying, get longer
    interpolation limits, and missing values are imputed per the column-
    specific semantics rather than dropping the row.
    """
    if 'Date' not in noaa_data or 'Date' not in flow_data:
        raise ValueError("'Date' column missing in one of the dataframes")

    try:
        daily_index = pd.date_range(
            pd.Timestamp(start_date).normalize(),
            pd.Timestamp(end_date).normalize(),
            freq='D',
            name='Date',
        )

        flow_daily = _to_daily_series(flow_data, ['Min Flow', 'Max Flow'], daily_index)
        # NOAA fetch now returns precipitation alongside TMIN/TMAX (get_noaa.py
        # surfaces GHCND PRCP as `precipitation` in mm). The column may still
        # be absent if the NCEI station had no PRCP coverage at all.
        noaa_daily = _to_daily_series(
            noaa_data, ['TMIN', 'TMAX', 'precipitation'], daily_index)

        combined = pd.concat([flow_daily, noaa_daily], axis=1)
        for col in CORE_COLUMNS:
            if col not in combined.columns:
                combined[col] = float('nan')
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

        # Time-aware interpolation of short interior gaps only (limit_area
        # 'inside' means no edge extrapolation). `combined` is a single station,
        # so this interpolation never bleeds across station boundaries. We
        # interpolate the core flow/temp columns at MAX_GAP_DAYS and handle
        # precipitation separately just below with a much shorter limit
        # (interpolation across a missed storm day is misleading).
        combined[CORE_COLUMNS] = combined[CORE_COLUMNS].interpolate(
            method='time', limit=MAX_GAP_DAYS, limit_area='inside')
        if 'precipitation' in combined.columns:
            combined['precipitation'] = pd.to_numeric(
                combined['precipitation'], errors='coerce')
            combined['precipitation'] = combined['precipitation'].interpolate(
                method='time', limit=MAX_PRECIP_GAP_DAYS, limit_area='inside')

        # SWE: separate, slow-varying series; longer interpolation limit.
        if swe_data is not None and not swe_data.empty:
            swe_daily = _to_daily_series(swe_data, ['SWE'], daily_index)
            swe_daily['SWE'] = pd.to_numeric(swe_daily['SWE'], errors='coerce')
            swe_daily = swe_daily.interpolate(method='time', limit=MAX_SWE_GAP_DAYS, limit_area='inside')
            combined['SWE'] = swe_daily['SWE']
        else:
            combined['SWE'] = float('nan')

        # SMAP soil moisture: slow-varying surface state; generous interior
        # interpolation limit (matches SWE). Edge handling happens AFTER the
        # core-column dropna below, where we have the final row set.
        if sm_data is not None and not sm_data.empty:
            sm_daily = _to_daily_series(sm_data, ['soil_moisture'], daily_index)
            sm_daily['soil_moisture'] = pd.to_numeric(sm_daily['soil_moisture'], errors='coerce')
            sm_daily = sm_daily.interpolate(method='time', limit=MAX_SM_GAP_DAYS, limit_area='inside')
            combined['soil_moisture'] = sm_daily['soil_moisture']
        else:
            combined['soil_moisture'] = float('nan')

        # USDM drought intensity: weekly snapshots, forward-filled. ffill is
        # the right move for a step-function weekly series (interpolation
        # would smear discrete category changes into ramps).
        if drought_data is not None and not drought_data.empty:
            d_daily = _to_daily_series(drought_data, ['drought_index'], daily_index)
            d_daily['drought_index'] = pd.to_numeric(d_daily['drought_index'], errors='coerce')
            d_daily['drought_index'] = d_daily['drought_index'].ffill(limit=MAX_DROUGHT_GAP_DAYS)
            combined['drought_index'] = d_daily['drought_index']
        else:
            combined['drought_index'] = float('nan')

        # USBR reservoir storage + release: daily series, slow-varying.
        # Interpolate short interior gaps. Empty df = unregulated station
        # (no mapping entry); the column stays NaN and is handled below.
        if reservoir_data is not None and not reservoir_data.empty:
            r_daily = _to_daily_series(
                reservoir_data,
                ['reservoir_storage', 'reservoir_release'],
                daily_index,
            )
            for col in ('reservoir_storage', 'reservoir_release'):
                r_daily[col] = pd.to_numeric(r_daily[col], errors='coerce')
            r_daily = r_daily.interpolate(method='time',
                                          limit=MAX_RESERVOIR_GAP_DAYS,
                                          limit_area='inside')
            combined['reservoir_storage'] = r_daily['reservoir_storage']
            combined['reservoir_release'] = r_daily['reservoir_release']
        else:
            combined['reservoir_storage'] = float('nan')
            combined['reservoir_release'] = float('nan')

        # Drop rows still missing any core flow/temp value. SWE / soil_moisture
        # / precipitation are NOT in CORE_COLUMNS, so a missing one never
        # drops a row.
        before = len(combined)
        combined = combined.dropna(subset=CORE_COLUMNS)
        # Precipitation: 0 mm means "no measurable rain", a legitimate default
        # for any post-interpolation gap (NCEI station with no PRCP coverage,
        # short outage that exceeded MAX_PRECIP_GAP_DAYS, etc.). Most days in
        # most basins are zero anyway, so this defaults to the modal value.
        if 'precipitation' not in combined.columns:
            combined['precipitation'] = 0.0
        combined['precipitation'] = combined['precipitation'].fillna(0.0)
        # SWE: 0 means "no snow", which IS a legitimate default; keep that.
        combined['SWE'] = combined['SWE'].fillna(0.0)
        # Soil moisture: 0 means "Sahara desert", which is NOT a legitimate
        # default for a SMAP gap (RFI / frozen ground / sensor outage). We
        # carry the last interpolated value forward and backward (SM is slow-
        # varying), then fall back to the station's median observed SM where
        # ffill/bfill can't reach, and only to 0 if the station has no
        # observations at all. An sm_observed indicator (1 = real or short-
        # gap interpolated retrieval, 0 = imputed via ffill / median fallback)
        # gives the model a way to tell the truth from the imputation.
        sm_series = combined['soil_moisture']
        combined['sm_observed'] = sm_series.notna().astype('int64')
        sm_series = sm_series.ffill().bfill()
        median = sm_series.median(skipna=True)
        if pd.isna(median):
            median = 0.0
        combined['soil_moisture'] = sm_series.fillna(median)

        # Drought: 0 means "no drought anywhere in the HUC", which IS a
        # legitimate default for a missing weekly value. Fill remaining gaps
        # with 0 -- the model can treat absence as "we don't have a drought
        # signal here" (which behaves equivalently to no drought).
        combined['drought_index'] = combined['drought_index'].fillna(0.0)

        # Reservoir: 0 storage / 0 release reads as "tiny / empty reservoir",
        # which IS misleading -- the right semantic for an unregulated gauge
        # is "no upstream reservoir at all". A reservoir_observed indicator
        # (1 = mapped + retrieval succeeded for this row, 0 = unmapped / no
        # data) tells the model when the storage / release columns are real.
        # The numeric columns then default to 0 only as a placeholder; the
        # indicator carries the truth.
        res_series = combined['reservoir_storage']
        combined['reservoir_observed'] = res_series.notna().astype('int64')
        combined['reservoir_storage'] = res_series.ffill().bfill().fillna(0.0)
        rel_series = combined['reservoir_release']
        combined['reservoir_release'] = rel_series.ffill().bfill().fillna(0.0)
        logger.info(
            "Site %s: %d/%d daily rows usable after gap handling",
            site_id, len(combined), before,
        )

        combined = combined.reset_index()
        combined['site_id'] = site_id
        # Basin identity (HUC8) is constant per station and is what the Phase 3
        # basin embedding looks up. Empty string when the lookup failed; the
        # basin index treats that as "unknown".
        combined['huc8'] = huc8 or ''
        return combined
    except Exception as e:
        logger.error(f"Error merging dataframes for site {site_id}: {e}")
        return pd.DataFrame()


def _disabled(env_var):
    """Truthy-toggle env-var check, used for the per-source ablation levers."""
    return os.getenv(env_var, '').strip() in ('1', 'true', 'True')


def fetch_and_process_data(prefix, site_id, start_date, end_date, flow_data):
    """
    Resolve a site's coordinates and fetch all its auxiliary timeseries:
    NOAA temperature, NRCS SWE, SMAP soil moisture, USDM drought, USBR
    reservoir storage + release, and the enclosing HUC8 basin id.

    Returns a dict with keys:
      noaa_data, swe_data, sm_data, drought_data, reservoir_data, huc8
    Any data frame may be empty (auxiliary features degrade gracefully; only
    missing NOAA causes the caller to skip the site). huc8 may be None.

    Per-source env vars short-circuit the corresponding fetch to empty for
    ablation runs:
      OPENFLOW_DISABLE_SMAP, OPENFLOW_DISABLE_DROUGHT, OPENFLOW_DISABLE_RESERVOIR
    """
    if prefix == "USGS":
        coords_dict = get_usgs_coordinates(site_id)
    elif prefix == "DWR":
        coords_dict = get_dwr_coordinates(site_id)
    else:
        coords_dict = None

    if not coords_dict:
        logger.error(f"Could not resolve coordinates for {prefix}:{site_id}. Skipping...")
        return None

    latitude = float(coords_dict['latitude'])
    longitude = float(coords_dict['longitude'])

    # get_noaa.main expects date strings, not datetime objects.
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    closest_noaa_station, noaa_data = get_noaa.main(latitude, longitude, start_str, end_str)

    if noaa_data is None or noaa_data.empty:
        logger.warning(f"No NOAA data available for site ID {site_id}. Skipping...")
        return None

    noaa_data = noaa_data.copy()
    noaa_data['USGS_site_ID'] = site_id

    # Resolve the enclosing HUC8 for the basin embedding. Cheap (one ArcGIS
    # call) and shapely-free, so we can do it here without dragging the data
    # deps into combine_data's environment.
    try:
        huc8 = get_swe.get_huc_id(latitude, longitude, level=8)
    except Exception as e:
        logger.warning("HUC8 lookup failed for %s: %s", site_id, e)
        huc8 = None
    if not huc8:
        logger.warning("No HUC8 resolved for %s -- basin embedding will fall back", site_id)

    # SWE: graceful failure -> empty.
    try:
        swe_data = get_swe.get_swe(latitude, longitude, start_date, end_date)
    except Exception as e:
        logger.warning("SWE fetch failed for %s: %s", site_id, e)
        swe_data = pd.DataFrame(columns=['Date', 'SWE'])

    # SMAP soil moisture: lazy import (heavy deps), env-var lever, graceful.
    sm_data = pd.DataFrame(columns=['Date', 'soil_moisture'])
    if _disabled('OPENFLOW_DISABLE_SMAP'):
        logger.info("OPENFLOW_DISABLE_SMAP set -- skipping SMAP for %s", site_id)
    else:
        try:
            from data import nasa_moisture
            sm_data = nasa_moisture.main(latitude, longitude, start_date, end_date)
        except ImportError as e:
            logger.warning("Soil-moisture deps unavailable (%s); skipping SMAP for %s",
                           e, site_id)
        except Exception as e:
            logger.warning("SMAP fetch failed for %s: %s", site_id, e)

    # USDM drought: light deps, env-var lever, graceful.
    drought_data = pd.DataFrame(columns=['Date', 'drought_index'])
    if _disabled('OPENFLOW_DISABLE_DROUGHT'):
        logger.info("OPENFLOW_DISABLE_DROUGHT set -- skipping USDM for %s", site_id)
    else:
        try:
            from data import get_drought
            drought_data = get_drought.get_drought(latitude, longitude, start_date, end_date)
        except Exception as e:
            logger.warning("USDM drought fetch failed for %s: %s", site_id, e)

    # USBR reservoir: needs a station->reservoir mapping; empty when no entry.
    reservoir_data = pd.DataFrame(
        columns=['Date', 'reservoir_storage', 'reservoir_release'])
    if _disabled('OPENFLOW_DISABLE_RESERVOIR'):
        logger.info("OPENFLOW_DISABLE_RESERVOIR set -- skipping RISE for %s",
                    site_id)
    else:
        try:
            from data import get_reservoir
            reservoir_data = get_reservoir.get_reservoir(
                f"{prefix}:{site_id}", start_date, end_date)
        except Exception as e:
            logger.warning("USBR reservoir fetch failed for %s: %s", site_id, e)

    return {
        'noaa_data': noaa_data,
        'swe_data': swe_data,
        'sm_data': sm_data,
        'drought_data': drought_data,
        'reservoir_data': reservoir_data,
        'huc8': huc8,
    }


def get_site_ids(filename=None):
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.github', 'site_ids.txt')
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_combined_data(all_data, base_path):
    """Concatenate the per-site frames, normalize, and persist the dataset."""
    final_data = pd.concat(all_data.values(), ignore_index=True)
    data_utils.preview_data(final_data)

    combined_data_file_path = os.path.join(base_path, 'openFlowML', 'combined_data_all_sites.csv')
    final_data.to_csv(combined_data_file_path, index=False)

    # Normalization persists scalers.json + station_index.json to base_path so
    # inference can reproduce the transform and invert flow predictions.
    normalized_data = normalize_data.normalize_data(final_data, artifacts_dir=base_path)
    if normalized_data is None:
        logger.error("Normalization failed; no normalized dataset produced")
        return None

    normalized_data_path = os.path.join(base_path, 'openFlowML', 'normalized_data.csv')
    normalized_data.to_csv(normalized_data_path, index=False)
    return normalized_data


def get_base_path():
    """Repo root, whether running in GitHub Actions or locally."""
    if 'GITHUB_WORKSPACE' in os.environ:
        return os.environ['GITHUB_WORKSPACE']
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(training_num_years=7):
    all_data = {}
    site_ids = get_site_ids()
    base_path = get_base_path()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=training_num_years * 365)

    for site_id in site_ids:
        try:
            prefix, id = site_id.split(':')
            logger.info(f"Consuming {prefix}:{id}")
            if prefix == "DWR":
                flow_dataframe = get_CODWR_flow.main(id, start_date, end_date)
            elif prefix == "USGS":
                flow_dataframe = get_flow.main(id, start_date, end_date)
            else:
                logger.warning(f"Unrecognized prefix for site ID {site_id}. Skipping...")
                continue

            fetched = fetch_and_process_data(
                prefix, id, start_date, end_date, flow_dataframe)
            if (fetched is None
                    or fetched['noaa_data'] is None
                    or fetched['noaa_data'].empty
                    or flow_dataframe.empty):
                logger.warning(f"No usable data for site ID {site_id}. Skipping...")
                continue

            merged = merge_dataframes(
                fetched['noaa_data'], flow_dataframe, site_id, start_date, end_date,
                swe_data=fetched['swe_data'],
                sm_data=fetched['sm_data'],
                drought_data=fetched['drought_data'],
                reservoir_data=fetched['reservoir_data'],
                huc8=fetched['huc8'])
            if merged.empty:
                logger.warning(f"No usable merged data for site ID {site_id}. Skipping...")
                continue
            all_data[site_id] = merged
        except Exception as e:
            logger.error(f"An error occurred for site ID {site_id}: {e}")

    if all_data:
        return save_combined_data(all_data, base_path)

    logger.error("No combined data for all sites")
    return None


if __name__ == "__main__":
    main()
