import os
from datetime import datetime, timedelta
import data.get_flow as get_flow
import data.get_CODWR_flow as get_CODWR_flow
import data.get_noaa as get_noaa
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

TODO (later in Phase 2): wire in SWE as a history-window feature.
"""

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Numeric columns that must be present and gap-free in the final per-site frame.
CORE_COLUMNS = ['Min Flow', 'Max Flow', 'TMIN', 'TMAX']
# Longest interior gap (in days) we are willing to interpolate across.
MAX_GAP_DAYS = 7


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


def merge_dataframes(noaa_data, flow_data, site_id, start_date, end_date):
    """
    Merge a site's NOAA temperature and flow data onto one regular daily index.

    Short interior gaps (<= MAX_GAP_DAYS) are interpolated time-aware; rows that
    are still missing a core value afterwards are dropped. There is deliberately
    no pooled-mean fill -- filling a flow gap with the average of every station
    injects garbage into the series.
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
        noaa_daily = _to_daily_series(noaa_data, ['TMIN', 'TMAX'], daily_index)

        combined = pd.concat([flow_daily, noaa_daily], axis=1)
        for col in CORE_COLUMNS:
            if col not in combined.columns:
                combined[col] = float('nan')
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

        # Time-aware interpolation of short interior gaps only (limit_area
        # 'inside' means no edge extrapolation). `combined` is a single station,
        # so this interpolation never bleeds across station boundaries.
        combined = combined.interpolate(method='time', limit=MAX_GAP_DAYS, limit_area='inside')

        # Drop rows still missing any core value (long gaps, leading/trailing).
        before = len(combined)
        combined = combined.dropna(subset=CORE_COLUMNS)
        logger.info(
            "Site %s: %d/%d daily rows usable after gap handling",
            site_id, len(combined), before,
        )

        combined = combined.reset_index()
        combined['site_id'] = site_id
        return combined
    except Exception as e:
        logger.error(f"Error merging dataframes for site {site_id}: {e}")
        return pd.DataFrame()


def fetch_and_process_data(prefix, site_id, start_date, end_date, flow_data):
    """Resolve a site's coordinates and fetch its NOAA temperature series."""
    if prefix == "USGS":
        coords_dict = get_usgs_coordinates(site_id)
    elif prefix == "DWR":
        coords_dict = get_dwr_coordinates(site_id)
    else:
        coords_dict = None

    if not coords_dict:
        logger.error(f"Could not resolve coordinates for {prefix}:{site_id}. Skipping...")
        return None

    latitude = coords_dict['latitude']
    longitude = coords_dict['longitude']

    # get_noaa.main expects date strings, not datetime objects.
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    closest_noaa_station, noaa_data = get_noaa.main(latitude, longitude, start_str, end_str)

    if noaa_data is None or noaa_data.empty:
        logger.warning(f"No NOAA data available for site ID {site_id}. Skipping...")
        return None

    noaa_data = noaa_data.copy()
    noaa_data['USGS_site_ID'] = site_id
    # Gap handling and numeric coercion happen in merge_dataframes (per station,
    # on the regular daily index) -- not here, and never via a pooled mean.
    return noaa_data


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

            noaa_dataframe = fetch_and_process_data(prefix, id, start_date, end_date, flow_dataframe)
            if noaa_dataframe is None or noaa_dataframe.empty or flow_dataframe.empty:
                logger.warning(f"No usable data for site ID {site_id}. Skipping...")
                continue

            merged = merge_dataframes(noaa_dataframe, flow_dataframe, site_id, start_date, end_date)
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
