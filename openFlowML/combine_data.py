import os
from datetime import datetime, timedelta
import data.get_flow as get_flow
import data.get_CODWR_flow as get_CODWR_flow
import data.get_noaa as get_noaa
import data.get_swe as get_swe
import data.get_upstream as get_upstream
import data.get_basin_precip as get_basin_precip
import data.get_hydro_forcing as get_hydro_forcing
import data.get_gridded_swe as get_gridded_swe
import data.get_nwm_prior as get_nwm_prior
import normalize_data
import windowing
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

Upstream gauges:
  - Stations listed in data/upstream_gauges.json get an `upstream_flow`
    column: the routed daily inflow from their upstream gauges (see
    data/get_upstream.py), with an `upstream_observed` indicator and a
    per-station `upstream_known_days` (how many forecast days of upstream
    flow are known at forecast time; windowing uses it). Stations without
    upstream gauges get the 0 placeholder + indicator 0, like reservoirs.
  - Set OPENFLOW_DISABLE_UPSTREAM=1 to run the ablation without it.

Basin precipitation by travel-time zone:
  - Stations in data/basin_zones.json get rain_/snow_{local,mid,far}: gridMET
    daily rain and snowfall averaged over each river-distance zone of the
    contributing area (data/get_basin_precip.py), with basin_precip_observed.
    Rain is never interpolated across missing days (observed = 0 instead).
  - Set OPENFLOW_DISABLE_BASIN_PRECIP=1 to run the ablation without it.
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
# Upstream routed inflow behaves like flow (can change quickly), so it gets
# the same short interior interpolation limit as the core flow columns.
MAX_UPSTREAM_GAP_DAYS = MAX_GAP_DAYS


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
                     huc8=None, upstream_data=None, upstream_known_days=0,
                     basin_precip_data=None,
                     energy_data=None, hydro_data=None,
                     gridded_swe_data=None, prior_data=None):
    """
    Merge a site's flow, weather, SWE, soil moisture, USDM drought, USBR
    reservoir, upstream gauges, basin zone precip, radiation/PET/VPD energy,
    continuous soil moisture + sub-daily rainfall intensity, elevation-band
    SWE, and NWM/RFC forecast prior series onto one regular daily index.

    Short interior flow/temp gaps (<= MAX_GAP_DAYS) are interpolated time-aware;
    rows still missing a core value afterwards are dropped (no pooled-mean
    fill). Auxiliary features are imputed per their column-specific semantics
    and paired with observed indicators rather than dropping rows.
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

        # Whole-gap interpolation of short interior gaps only (<= MAX_GAP_DAYS):
        # unlike pandas' limit=n, a long outage (e.g. winter ice) stays entirely
        # missing instead of getting MAX_GAP_DAYS fake linear-ramp days at its
        # start. `combined` is a single station, so interpolation never bleeds
        # across station boundaries.
        for col in CORE_COLUMNS:
            combined[col] = get_upstream.interpolate_short_gaps(combined[col], MAX_GAP_DAYS)
        if 'precipitation' in combined.columns:
            combined['precipitation'] = pd.to_numeric(
                combined['precipitation'], errors='coerce')
            combined['precipitation'] = get_upstream.interpolate_short_gaps(
                combined['precipitation'], MAX_PRECIP_GAP_DAYS)

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

        # Upstream routed inflow (daily, already lagged + weighted onto the
        # target's dates by get_upstream). Empty df = no upstream gauges.
        if upstream_data is not None and not upstream_data.empty:
            u_daily = _to_daily_series(upstream_data, ['upstream_flow'], daily_index)
            u_daily['upstream_flow'] = pd.to_numeric(u_daily['upstream_flow'], errors='coerce')
            # Whole-gap rule (not pandas' limit=n partial fill): a long
            # upstream outage stays missing -> upstream_observed = 0.
            combined['upstream_flow'] = get_upstream.interpolate_short_gaps(
                u_daily['upstream_flow'], MAX_UPSTREAM_GAP_DAYS)
        else:
            combined['upstream_flow'] = float('nan')

        # Basin zone rain / snow (mm/day). No interpolation: a missing day
        # of rain is unknown, not the average of its neighbours.
        bp_cols = get_basin_precip.FEATURE_COLUMNS
        if basin_precip_data is not None and not basin_precip_data.empty:
            bp_daily = _to_daily_series(basin_precip_data, bp_cols, daily_index)
            for col in bp_cols:
                combined[col] = pd.to_numeric(bp_daily[col], errors='coerce')
        else:
            for col in bp_cols:
                combined[col] = float('nan')

        # Phase 1: Radiation / PET / VPD energy forcing (short interior gaps).
        en_cols = get_basin_precip.ENERGY_COLUMNS
        if energy_data is not None and not energy_data.empty:
            en_daily = _to_daily_series(energy_data, en_cols, daily_index)
            for col in en_cols:
                combined[col] = get_upstream.interpolate_short_gaps(
                    pd.to_numeric(en_daily[col], errors='coerce'), MAX_GAP_DAYS)
        else:
            for col in en_cols:
                combined[col] = float('nan')

        # Phase 1: Sub-daily rainfall intensity (no interpolation) + continuous
        # surface/root-zone soil moisture (slow-varying, long gap interp).
        int_cols = get_hydro_forcing.INTENSITY_COLUMNS
        hsm_cols = get_hydro_forcing.SOIL_COLUMNS
        if hydro_data is not None and not hydro_data.empty:
            hy_daily = _to_daily_series(hydro_data, [*int_cols, *hsm_cols], daily_index)
            for col in int_cols:
                combined[col] = pd.to_numeric(hy_daily[col], errors='coerce')
            for col in hsm_cols:
                s = pd.to_numeric(hy_daily[col], errors='coerce')
                combined[col] = s.interpolate(method='time', limit=MAX_SM_GAP_DAYS, limit_area='inside')
        else:
            for col in [*int_cols, *hsm_cols]:
                combined[col] = float('nan')

        # Phase 3: Elevation-band gridded SWE (slow-varying).
        gswe_cols = get_gridded_swe.SWE_COLUMNS
        if gridded_swe_data is not None and not gridded_swe_data.empty:
            gs_daily = _to_daily_series(gridded_swe_data, gswe_cols, daily_index)
            for col in gswe_cols:
                s = pd.to_numeric(gs_daily[col], errors='coerce')
                combined[col] = s.interpolate(method='time', limit=MAX_SWE_GAP_DAYS, limit_area='inside')
        else:
            for col in gswe_cols:
                combined[col] = float('nan')

        # Phase 3: Physics/operator prior (NWM / RFC streamflow, cfs).
        if prior_data is not None and not prior_data.empty:
            pr_daily = _to_daily_series(prior_data, ['prior_flow'], daily_index)
            combined['prior_flow'] = get_upstream.interpolate_short_gaps(
                pd.to_numeric(pr_daily['prior_flow'], errors='coerce'), MAX_GAP_DAYS)
        else:
            combined['prior_flow'] = float('nan')

        # Drop rows still missing any core flow/temp value. Auxiliary columns
        # are NOT in CORE_COLUMNS, so a missing one never drops a row.
        before = len(combined)
        combined = combined.dropna(subset=CORE_COLUMNS)
        # Precipitation: 0 mm means "no measurable rain", a legitimate default
        # for any post-interpolation gap.
        if 'precipitation' not in combined.columns:
            combined['precipitation'] = 0.0
        combined['precipitation'] = combined['precipitation'].fillna(0.0)
        # SWE: 0 means "no snow", which IS a legitimate default; keep that.
        combined['SWE'] = combined['SWE'].fillna(0.0)
        # Soil moisture: carry the last interpolated value forward/backward,
        # then fall back to station median, with sm_observed indicator.
        sm_series = combined['soil_moisture']
        combined['sm_observed'] = sm_series.notna().astype('int64')
        sm_series = sm_series.ffill().bfill()
        median = sm_series.median(skipna=True) if sm_series.notna().any() else 0.0
        combined['soil_moisture'] = sm_series.fillna(median)

        # Drought: 0 means "no drought anywhere in the HUC".
        combined['drought_index'] = combined['drought_index'].fillna(0.0)

        # Reservoir: paired with reservoir_observed indicator.
        res_series = combined['reservoir_storage']
        combined['reservoir_observed'] = res_series.notna().astype('int64')
        combined['reservoir_storage'] = res_series.ffill().bfill().fillna(0.0)
        rel_series = combined['reservoir_release']
        combined['reservoir_release'] = rel_series.ffill().bfill().fillna(0.0)

        # Upstream: no ffill/bfill across outages; 0 placeholder + indicator.
        up_series = combined['upstream_flow']
        combined['upstream_observed'] = up_series.notna().astype('int64')
        combined['upstream_flow'] = up_series.fillna(0.0)
        combined['upstream_known_days'] = (
            int(upstream_known_days) if combined['upstream_observed'].any() else 0)

        combined['basin_precip_observed'] = (
            combined[bp_cols].notna().all(axis=1).astype('int64'))
        combined[bp_cols] = combined[bp_cols].fillna(0.0)

        # Phase 1-3 indicators & post-dropna imputation.
        combined['energy_observed'] = combined[en_cols].notna().all(axis=1).astype('int64')
        combined[en_cols] = combined[en_cols].fillna(0.0)

        combined['precip_intensity_observed'] = combined[int_cols].notna().all(axis=1).astype('int64')
        combined[int_cols] = combined[int_cols].fillna(0.0)

        combined['hydro_sm_observed'] = combined[hsm_cols].notna().all(axis=1).astype('int64')
        for col in hsm_cols:
            s = combined[col].ffill().bfill()
            med = s.median(skipna=True) if s.notna().any() else 0.0
            combined[col] = s.fillna(med)

        combined['gridded_swe_observed'] = combined[gswe_cols].notna().all(axis=1).astype('int64')
        combined[gswe_cols] = combined[gswe_cols].fillna(0.0)

        pr_series = combined['prior_flow']
        combined['prior_observed'] = pr_series.notna().astype('int64')
        combined['prior_flow'] = pr_series.fillna(0.0)

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


# Weather gap fill: days the nearest NOAA station did not report take the
# gridMET cell at the gauge, shifted by the station-minus-gridMET mean
# temperature difference for that calendar month (all months pooled when a
# month has fewer than this many overlapping days).
WEATHER_FILL_MIN_OVERLAP_DAYS = 20
WEATHER_COLUMNS = ['TMIN', 'TMAX', 'precipitation']


def fill_weather_gaps(noaa_data, grid_data):
    """
    Fill missing NOAA days with gridMET point weather (both in training units).

    Returns (filled DataFrame[Date, TMIN, TMAX, precipitation], days filled).
    Days the station reported keep the station values; for a missing day
    TMIN/TMAX are gridMET plus that month's mean station-minus-gridMET offset
    and precipitation is gridMET unchanged. A missing station precipitation
    on a day with station temperatures is filled the same way. Pure function.
    """
    if grid_data is None or grid_data.empty:
        return noaa_data, 0
    grid = grid_data[['Date', *WEATHER_COLUMNS]].copy()
    grid['Date'] = pd.to_datetime(grid['Date']).dt.normalize()
    grid = grid.set_index('Date').apply(pd.to_numeric, errors='coerce')
    if noaa_data is None or noaa_data.empty:
        return grid.reset_index(), int(grid[['TMIN', 'TMAX']].notna().all(axis=1).sum())
    noaa = noaa_data.copy()
    noaa['Date'] = pd.to_datetime(noaa['Date']).dt.normalize()
    extra = [c for c in noaa.columns if c not in ('Date', *WEATHER_COLUMNS)]
    noaa = noaa.drop_duplicates('Date').set_index('Date')
    for col in WEATHER_COLUMNS:
        if col not in noaa.columns:
            noaa[col] = float('nan')
        noaa[col] = pd.to_numeric(noaa[col], errors='coerce')
    index = noaa.index.union(grid.index)
    station = noaa.reindex(index)
    g = grid.reindex(index)
    out = station[WEATHER_COLUMNS].copy()
    month = index.month
    for col in ('TMIN', 'TMAX'):
        diff = station[col] - g[col]
        pooled = diff.mean() if diff.notna().any() else 0.0
        counts = diff.groupby(month).count()
        means = diff.groupby(month).mean()
        offset = pd.Series([means[m] if counts.get(m, 0) >= WEATHER_FILL_MIN_OVERLAP_DAYS else pooled
                            for m in month], index=index)
        out[col] = station[col].fillna(g[col] + offset)
    out['precipitation'] = station['precipitation'].fillna(g['precipitation'])
    filled = int((station[['TMIN', 'TMAX']].isna().any(axis=1)
                  & out[['TMIN', 'TMAX']].notna().all(axis=1)).sum())
    for col in extra:
        out[col] = station[col]
    out.index.name = 'Date'
    return out.reset_index(), filled


DEFAULT_WEATHER_SOURCE = 'gridmet'


def weather_source(override=None) -> str:
    """Weather source for point TMIN/TMAX/precipitation: 'gridmet' (default) or 'noaa'."""
    raw = override if override is not None else os.getenv('OPENFLOW_WEATHER_SOURCE', '')
    src = (raw or DEFAULT_WEATHER_SOURCE).strip().lower()
    return src if src in ('gridmet', 'noaa') else DEFAULT_WEATHER_SOURCE


def fetch_and_process_data(prefix, site_id, start_date, end_date, flow_data, noaa_fallback=None,
                           coords=None, weather_source_override=None,
                           hydro_source_override=None, include_historical_prior=True):
    """
    Resolve a site's coordinates and fetch all its auxiliary timeseries:
    gauge-collocated weather (gridMET by default, or NOAA GHCND), NRCS SWE,
    SMAP soil moisture, USDM drought, USBR reservoir storage + release,
    upstream gauges, basin zone precip, gridMET radiation/PET/VPD energy,
    continuous soil moisture + sub-daily rainfall intensity (ERA5-Land or
    NLDAS-2), elevation-band gridded SWE, NWM retrospective prior, and HUC8.

    Returns a dict with keys:
      noaa_data, swe_data, sm_data, drought_data, reservoir_data,
      upstream_data, upstream_known_days, basin_precip_data,
      energy_data, hydro_data, gridded_swe_data, prior_data, huc8
    Any auxiliary data frame may be empty (degrades gracefully; only missing
    weather causes the caller to skip the site). huc8 may be None.

    Per-source env vars short-circuit the corresponding fetch to empty for
    ablation runs:
      OPENFLOW_DISABLE_SMAP, OPENFLOW_DISABLE_DROUGHT, OPENFLOW_DISABLE_RESERVOIR,
      OPENFLOW_DISABLE_UPSTREAM, OPENFLOW_DISABLE_BASIN_PRECIP,
      OPENFLOW_DISABLE_ENERGY, OPENFLOW_DISABLE_HYDRO_FORCING,
      OPENFLOW_DISABLE_GRIDDED_SWE, OPENFLOW_DISABLE_PRIOR
    """
    if coords is not None:   # caller already resolved them (saves a flaky second lookup)
        coords_dict = {'latitude': coords['lat'], 'longitude': coords['lon']}
    elif prefix == "USGS":
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
    wx_source = weather_source(weather_source_override)
    noaa_data = pd.DataFrame()

    if wx_source == 'gridmet' and not _disabled('OPENFLOW_DISABLE_WEATHER_FILL'):
        # Collocated 4-km gridMET at the gauge coordinates eliminates elevation
        # offsets between distant GHCND stations and live gauge-point forecasts
        # (Open-Meteo / NWS gridpoints).
        try:
            grid = get_basin_precip.gridmet_point_weather(latitude, longitude, start_date, end_date)
            if grid is not None and not grid.empty:
                noaa_data = grid
        except Exception as e:
            logger.warning("gridMET primary weather failed for %s (%s); falling back to NOAA", site_id, e)

    if noaa_data is None or noaa_data.empty:
        _, noaa_data = get_noaa.main(latitude, longitude, start_str, end_str)

        if (noaa_data is None or noaa_data.empty) and noaa_fallback is not None:
            # Inference over a short recent window: no nearby GHCND station may
            # pass the coverage gate even though one did over the training years.
            logger.info("No NOAA station for %s; using fallback weather", site_id)
            noaa_data = noaa_fallback(latitude, longitude, start_date, end_date)

        # Station outages (a GHCND station that stops reporting) otherwise drop
        # every day without temperature from training; fill them from gridMET.
        if wx_source == 'noaa' and not _disabled('OPENFLOW_DISABLE_WEATHER_FILL'):
            try:
                grid = get_basin_precip.gridmet_point_weather(latitude, longitude, start_date, end_date)
                noaa_data, n_filled = fill_weather_gaps(noaa_data, grid)
                if n_filled:
                    logger.info("Filled %d NOAA weather days for %s from gridMET", n_filled, site_id)
            except Exception as e:
                logger.warning("gridMET weather gap fill skipped for %s: %s", site_id, e)

    if noaa_data is None or noaa_data.empty:
        logger.warning(f"No weather data available for site ID {site_id}. Skipping...")
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

    # Upstream gauges (data/upstream_gauges.json): routed daily inflow. Empty
    # when the station has none; graceful on failure.
    full_id = f"{prefix}:{site_id}"
    upstream_data = pd.DataFrame(columns=['Date', 'upstream_flow'])
    upstream_known_days = 0
    try:
        upstream_data = get_upstream.get_upstream(full_id, start_date, end_date)
        if not upstream_data.empty:
            upstream_known_days = get_upstream.known_days(
                get_upstream.upstream_config(full_id), windowing.DECODER_DAYS)
    except Exception as e:
        logger.warning("Upstream fetch failed for %s: %s", site_id, e)

    # Basin precipitation by travel-time zone (gridMET over basin_zones.json).
    try:
        basin_precip_data = get_basin_precip.get_zone_precip_daily(
            full_id, start_date, end_date)
    except Exception as e:
        logger.warning("Basin precip fetch failed for %s: %s", site_id, e)
        basin_precip_data = pd.DataFrame(columns=['Date'] + get_basin_precip.FEATURE_COLUMNS)

    # Phase 1: gridMET solar radiation / PET / VPD energy forcing.
    energy_data = pd.DataFrame(columns=['Date', *get_basin_precip.ENERGY_COLUMNS])
    if not _disabled('OPENFLOW_DISABLE_ENERGY'):
        try:
            energy_data = get_basin_precip.gridmet_point_energy(
                latitude, longitude, start_date, end_date)
        except Exception as e:
            logger.warning("gridMET energy fetch failed for %s: %s", site_id, e)

    # Phase 1: Continuous surface/root-zone soil moisture + sub-daily rainfall intensity.
    hydro_data = pd.DataFrame(columns=['Date', *get_hydro_forcing.ALL_COLUMNS])
    if not _disabled('OPENFLOW_DISABLE_HYDRO_FORCING'):
        try:
            hydro_data = get_hydro_forcing.get_hydro_forcing(
                latitude, longitude, start_date, end_date,
                source_override=hydro_source_override)
        except Exception as e:
            logger.warning("Hydro forcing fetch failed for %s: %s", site_id, e)

    # Phase 3: Elevation-band gridded SWE (low / mid / high + melt rate).
    gridded_swe_data = pd.DataFrame(columns=['Date', *get_gridded_swe.SWE_COLUMNS])
    if not _disabled('OPENFLOW_DISABLE_GRIDDED_SWE'):
        try:
            gridded_swe_data = get_gridded_swe.get_gridded_swe(
                full_id, latitude, longitude, start_date, end_date)
        except Exception as e:
            logger.warning("Gridded SWE fetch failed for %s: %s", site_id, e)

    # Phase 3: Historical NWM v3.0 retrospective streamflow prior (when optional
    # Zarr deps are installed; skipped at inference where live RFC/NWM is used).
    prior_data = pd.DataFrame(columns=['Date', 'prior_flow'])
    if include_historical_prior and not _disabled('OPENFLOW_DISABLE_PRIOR'):
        try:
            prior_data = get_nwm_prior.fetch_historical_prior(
                full_id, start_date, end_date)
        except Exception as e:
            logger.warning("Historical prior fetch failed for %s: %s", site_id, e)

    return {
        'noaa_data': noaa_data,
        'swe_data': swe_data,
        'sm_data': sm_data,
        'drought_data': drought_data,
        'reservoir_data': reservoir_data,
        'basin_precip_data': basin_precip_data,
        'upstream_data': upstream_data,
        'upstream_known_days': upstream_known_days,
        'energy_data': energy_data,
        'hydro_data': hydro_data,
        'gridded_swe_data': gridded_swe_data,
        'prior_data': prior_data,
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


# Years of history to train on. Longer records span more wet and dry snow
# years, so the model learns the snowpack -> runoff link instead of the
# seasonal cycle of a few particular years. OPENFLOW_TRAINING_YEARS overrides.
TRAINING_YEARS = 25


def fetch_station_merged(site_id, start_date, end_date):
    """Fetch flow + auxiliary sources for one station and merge onto a daily index."""
    prefix, sid = site_id.split(':', 1)
    logger.info(f"Consuming {prefix}:{sid}")
    if prefix == "DWR":
        flow_dataframe = get_CODWR_flow.main(sid, start_date, end_date)
    elif prefix == "USGS":
        flow_dataframe = get_flow.main(sid, start_date, end_date)
    else:
        logger.warning(f"Unrecognized prefix for site ID {site_id}. Skipping...")
        return pd.DataFrame()

    fetched = fetch_and_process_data(
        prefix, sid, start_date, end_date, flow_dataframe)
    if (fetched is None
            or fetched['noaa_data'] is None
            or fetched['noaa_data'].empty
            or flow_dataframe.empty):
        logger.warning(f"No usable data for site ID {site_id}. Skipping...")
        return pd.DataFrame()

    return merge_dataframes(
        fetched['noaa_data'], flow_dataframe, site_id, start_date, end_date,
        swe_data=fetched['swe_data'],
        sm_data=fetched['sm_data'],
        drought_data=fetched['drought_data'],
        reservoir_data=fetched['reservoir_data'],
        huc8=fetched['huc8'],
        upstream_data=fetched.get('upstream_data'),
        upstream_known_days=fetched.get('upstream_known_days', 0),
        basin_precip_data=fetched.get('basin_precip_data'),
        energy_data=fetched.get('energy_data'),
        hydro_data=fetched.get('hydro_data'),
        gridded_swe_data=fetched.get('gridded_swe_data'),
        prior_data=fetched.get('prior_data'))


def _resolve_cache_dir(cache_dir=None):
    if cache_dir:
        return os.path.expanduser(cache_dir)
    env_dir = os.getenv('OPENFLOW_CACHE_DIR', '').strip()
    if env_dir:
        return os.path.expanduser(env_dir)
    if _disabled('OPENFLOW_USE_CACHE'):
        from data import feature_store
        return feature_store.DEFAULT_CACHE_DIR
    return None


def main(training_num_years=None, cache_dir=None):
    if training_num_years is None:
        training_num_years = int(os.getenv('OPENFLOW_TRAINING_YEARS', '') or TRAINING_YEARS)
    resolved_cache = _resolve_cache_dir(cache_dir)
    refresh_days = int(os.getenv('OPENFLOW_CACHE_REFRESH_DAYS', '') or 14)
    force_rebuild = _disabled('OPENFLOW_CACHE_FORCE_REBUILD')
    all_data = {}
    site_ids = get_site_ids()
    base_path = get_base_path()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=training_num_years * 365)

    for site_id in site_ids:
        try:
            if resolved_cache:
                from data import feature_store
                merged, status = feature_store.load_or_update_station(
                    site_id, start_date, end_date,
                    fetch_station_merged,
                    resolved_cache,
                    refresh_days=refresh_days,
                    force_rebuild=force_rebuild,
                )
                logger.info("Station %s loaded via feature store (%s, %d rows)",
                            site_id, status, len(merged))
            else:
                merged = fetch_station_merged(site_id, start_date, end_date)
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
