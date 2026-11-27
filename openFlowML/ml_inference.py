"""
Run a trained OpenFlow model for one station, now: the Python reference
implementation of docs/INFERENCE.md, and what flow_forecast.py --model-dir
uses to put the ML forecast on the weekly chart.

Inputs are assembled with the SAME code that built the training data:
  - encoder (last 60 days): combine_data.fetch_and_process_data +
    merge_dataframes (flow, NOAA GHCND weather, SWE, soil moisture, drought,
    reservoirs, upstream gauges, basin zone rain/snow), then the persisted
    scalers.json. Recency fixes so the forecast starts from today:
      * flow archives that end yesterday (CO DWR) are topped up with the
        gauge's NWS (NWPS) observations through today;
      * NOAA daily data runs ~2-4 days behind, so the days between the last
        NOAA day and the last flow day are bridged: temperature interpolated
        toward the NWS gridpoint forecast for today, precipitation from the
        observed local-zone basin rain (gridMET / NWS Stage IV);
      * when no nearby GHCND station has enough data in the short recent
        window (e.g. Cisco, UT), gridMET at the gauge stands in (same units).
  - decoder (next 14 days): weather from Open-Meteo when it answers, else the
    NWS gridpoint forecast at the gauge (~7 days), then persistence of the
    last forecast day with no rain; routed upstream flow from upstream
    observations + their NWS RFC forecasts with windowing's hold rule;
    basin zone rain/snow from get_basin_precip.inference_daily with
    windowing's 7-day mask.

Units follow training: TMIN/TMAX in tenths of deg C (raw GHCND), precipitation
in mm, flows in cfs. Output: daily Min/Max Flow (cfs) for the decoder days.
"""

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import combine_data
import normalize_data
import windowing
from data import get_basin_precip, get_CODWR_flow, get_flow, get_upstream, nwps
from data.utils.get_coordinates import get_dwr_coordinates, get_usgs_coordinates

logger = logging.getLogger(__name__)

HISTORY_PAD_DAYS = 30
JSON_ARTIFACTS = ('scalers', 'station_index', 'basin_index', 'training_config')
NWS_POINTS_URL = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"


class InferenceError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

def load_artifacts(model_dir: str, with_model: bool = True) -> dict:
    """scalers / indices / config (+ the Keras model) from a release or training dir."""
    art = {}
    for name in JSON_ARTIFACTS:
        path = os.path.join(model_dir, f'{name}.json')
        if not os.path.exists(path):
            raise InferenceError(f"{path} missing -- not a trained-model directory")
        with open(path) as f:
            art[name] = json.load(f)
    reg = os.path.join(model_dir, 'upstream_gauges.json')
    art['registry'] = get_upstream.load_registry(reg if os.path.exists(reg) else None)
    from data import build_station_attributes as bsa
    attr_path = os.path.join(model_dir, 'station_attributes.json')
    art['station_attributes'] = bsa.load_station_attributes(
        attr_path if os.path.exists(attr_path) else None)
    art['model_dir'] = model_dir
    art['version'] = None
    manifest = os.path.join(model_dir, 'manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as f:
            art['version'] = json.load(f).get('model_version')
    # Held-out verification of this model's training recipe (evaluate.py).
    quality = os.path.join(model_dir, 'quality_report.json')
    art['quality'] = None
    if os.path.exists(quality):
        with open(quality) as f:
            art['quality'] = json.load(f)
    if with_model:
        import tensorflow as tf
        import model as model_mod
        art['model'] = tf.keras.models.load_model(
            os.path.join(model_dir, 'lstm_model.h5'),
            custom_objects=model_mod.CUSTOM_OBJECTS, compile=False)
    return art


# --------------------------------------------------------------------------
# Data assembly helpers
# --------------------------------------------------------------------------

# Static fallback coordinates for registered stations when USGS/DWR site
# metadata APIs have a transient outage (e.g. Taos Box 2026-11-16).
FALLBACK_COORDS = {
    'USGS:08276500': {'lat': 36.3200333, 'lon': -105.7544444},
    'USGS:09070500': {'lat': 39.6446111, 'lon': -107.0780139},
    'USGS:09114500': {'lat': 38.5419357, 'lon': -106.9497661},
    'USGS:09163500': {'lat': 39.1327420, 'lon': -109.0270750},
    'USGS:09180000': {'lat': 38.7972081, 'lon': -109.1951142},
    'DWR:ARKCANCO':  {'lat': 38.4336240, 'lon': -105.2577840},
    'DWR:CONMOGCO':  {'lat': 37.0539540, 'lon': -106.1871450},
    'USGS:06719505': {'lat': 39.7530556, 'lon': -105.2347222},
}


def _coords(site_id: str) -> dict:
    prefix, sid = site_id.split(':', 1)
    c = get_usgs_coordinates(sid) if prefix == 'USGS' else get_dwr_coordinates(sid)
    if c:
        return {'lat': float(c['latitude']), 'lon': float(c['longitude'])}
    if site_id in FALLBACK_COORDS:
        logger.warning("Coordinate lookup failed for %s; using cached fallback coordinates", site_id)
        return dict(FALLBACK_COORDS[site_id])
    raise InferenceError(f"no coordinates for {site_id}")


MIN_FLOW_SPLICE_OVERLAP_DAYS = 3


def _flow(site_id: str, start, end, nws_lid: Optional[str] = None,
          tz: str = 'America/Denver') -> pd.DataFrame:
    """Daily Min/Max Flow; days after the archive's last day come from NWPS observations."""
    prefix, sid = site_id.split(':', 1)
    if prefix == 'DWR':
        df = get_CODWR_flow.get_historical_data(sid, start, end)
    else:
        df = get_flow.get_daily_flow_data(sid, start, end)
    if not nws_lid:
        return df
    last = pd.to_datetime(df['Date']).max() if df is not None and len(df) else pd.Timestamp(start)
    try:
        obs = nwps.parse_series(nwps.get_stageflow(nws_lid), 'observed')
    except Exception as e:
        logger.info("NWPS top-up unavailable for %s: %s", nws_lid, e)
        return df
    obs = obs.dropna(subset=['flow_cfs'])
    if obs.empty:
        return df
    local = obs.set_index('time')['flow_cfs'].tz_convert(tz)
    daily = local.groupby(local.index.date).agg(['min', 'max', 'mean'])
    daily.index = pd.DatetimeIndex(daily.index)
    if prefix == 'DWR':
        # The DWR archive the model trained on is one daily mean per day
        # (Min Flow == Max Flow); top-up days must look the same.
        daily['min'] = daily['max'] = daily['mean']
    # Rescale NWPS top-up days onto the archive's training scale whenever
    # enough overlapping days exist to measure a rating-curve shift (e.g.
    # CO DWR CONMOGCO vs NWPS MOGC2 reading +196 cfs higher in Nov 2026).
    ratio_min = ratio_max = 1.0
    if df is not None and len(df):
        arch = df.copy()
        arch['_d'] = pd.to_datetime(arch['Date']).dt.normalize()
        arch = arch.groupby('_d')[['Min Flow', 'Max Flow']].mean()
        both = arch.join(daily[['min', 'max']], how='inner').dropna()
        both = both[(both['min'] > 0) & (both['max'] > 0)].tail(SPLICE_OVERLAP_DAYS)
        if len(both) >= MIN_FLOW_SPLICE_OVERLAP_DAYS:
            ratio_min = float(np.median(both['Min Flow'] / both['min']))
            ratio_max = float(np.median(both['Max Flow'] / both['max']))
            if prefix == 'DWR':
                ratio_min = ratio_max = 0.5 * (ratio_min + ratio_max)
            if abs(ratio_max - 1.0) > 0.05:
                logger.info("Rescaling %s NWPS top-up by %.3f (%.0f overlap days)",
                            site_id, ratio_max, len(both))
    daily = daily[(daily.index > last) & (daily.index <= pd.Timestamp(end).normalize())]
    if daily.empty:
        return df
    extra = pd.DataFrame({'Date': daily.index.strftime('%Y-%m-%d'),
                          'Min Flow': daily['min'].to_numpy() * ratio_min,
                          'Max Flow': daily['max'].to_numpy() * ratio_max})
    logger.info("Topped up %s flow with %d NWPS day(s)", site_id, len(extra))
    return pd.concat([df, extra], ignore_index=True)


# Kept under its old name; lives in the data layer so training can use it too.
gridmet_point_weather = get_basin_precip.gridmet_point_weather


def nws_point_daily(lat: float, lon: float, tz: str) -> pd.DataFrame:
    """Daily TMIN/TMAX (tenths deg C) + precipitation (mm) from the NWS gridpoint forecast at a point."""
    from data.utils import data_utils
    r = data_utils.request_with_retry(NWS_POINTS_URL.format(lat=lat, lon=lon),
                                      headers=get_basin_precip.NWS_HEADERS)
    if r is None:
        return pd.DataFrame(columns=['TMIN', 'TMAX', 'precipitation'])
    p = r.json()['properties']
    g = data_utils.request_with_retry(p['forecastGridData'], headers=get_basin_precip.NWS_HEADERS)
    if g is None:
        return pd.DataFrame(columns=['TMIN', 'TMAX', 'precipitation'])
    hourly = get_basin_precip.parse_nws_gridpoint(g.json()).tz_convert(tz)
    grp = hourly.groupby(hourly.index.date)
    out = pd.DataFrame({'TMIN': grp['temp_c'].min() * 10.0,
                        'TMAX': grp['temp_c'].max() * 10.0,
                        'precipitation': grp['precip_mm'].sum(min_count=1)})
    out.index = pd.DatetimeIndex(out.index, name='Date')
    return out


def forecast_weather(lat: float, lon: float, tz: str, days: int = 16) -> pd.DataFrame:
    """
    Daily forecast weather in training units; Open-Meteo extended forecast
    (TMIN/TMAX tenths deg C, precipitation mm, srad MJ/m^2/day, pet mm/day,
    vpd kPa, precip_max1h/max3h/wet_hours) + 31-member ensemble spread when
    available, falling back cleanly to basic Open-Meteo or NWS gridpoints.
    """
    try:
        from data import get_forecast
        om = get_forecast.get_open_meteo_extended_forecast(
            lat, lon, days=days, temperature_unit='celsius', timezone=tz)
        if om is None or om.empty:
            om = get_forecast.get_open_meteo_forecast(lat, lon, days=days, temperature_unit='celsius')
        if om is not None and not om.empty:
            om = om.set_index(pd.DatetimeIndex(pd.to_datetime(om['Date']), name='Date'))
            om = om.drop(columns=['Date'], errors='ignore')
            om[['TMIN', 'TMAX']] = om[['TMIN', 'TMAX']].astype(float) * 10.0
            try:
                ens = get_forecast.get_open_meteo_ensemble(lat, lon, days=days, timezone=tz)
                if ens is not None and not ens.empty:
                    ens = ens.set_index(pd.DatetimeIndex(pd.to_datetime(ens['Date']), name='Date'))
                    for c in windowing.ENSEMBLE_SPREAD_FEATURES:
                        if c in ens.columns:
                            om[c] = ens[c].reindex(om.index)
            except Exception as e:
                logger.debug("Open-Meteo ensemble spread unavailable (%s)", e)
            return om.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        logger.info("Open-Meteo forecast unavailable (%s); using NWS", e)
    return nws_point_daily(lat, lon, tz)


SPLICE_OVERLAP_DAYS = 14


def splice_on_training_scale(archive: pd.Series, nwps_daily: pd.Series) -> pd.Series:
    """
    One daily series on the training data's scale: the archive (the source
    the model trained on) wherever it has a value, then the NWPS days after
    it ends (observations, then the RFC forecast), multiplied by the median
    archive / NWPS ratio over their last SPLICE_OVERLAP_DAYS common days.
    The two can disagree: in Oct 2026 the CO DWR Lobatos record read ~0.5x
    the USGS-based NWPS series, so letting NWPS overwrite the archive fed the
    model an upstream level it never saw in training.
    """
    archive = archive.dropna() if archive is not None else pd.Series(dtype=float)
    nwps_daily = nwps_daily.dropna() if nwps_daily is not None else pd.Series(dtype=float)
    if archive.empty:
        return nwps_daily
    if nwps_daily.empty:
        return archive
    both = pd.concat([archive, nwps_daily], axis=1, keys=['a', 'n']).dropna()
    both = both[both['n'] > 0].tail(SPLICE_OVERLAP_DAYS)
    ratio = float(np.median(both['a'] / both['n'])) if len(both) else 1.0
    tail = nwps_daily[nwps_daily.index > archive.index.max()] * ratio
    return pd.concat([archive, tail]).sort_index()


def _routed_upstream(site_id: str, cfg: Optional[dict], days: pd.DatetimeIndex,
                     history_start) -> pd.DataFrame:
    """Routed upstream_flow + observed flag for `days` from observed daily means + RFC forecasts."""
    out = pd.DataFrame({'upstream_flow': np.nan}, index=days)
    if not cfg or not cfg.get('upstream'):
        return out
    series = {}
    # Upstream gauges plus the anchor gauge, if the station has one
    # (get_upstream.anchored_daily): observed history + NWPS obs/forecast.
    for g in cfg['upstream'] + ([cfg['anchor']] if cfg.get('anchor') else []):
        hist = get_upstream.fetch_daily_mean(g['site_id'], history_start, days[-1])
        payload = nwps.get_stageflow(g['nws_lid']) if g.get('nws_lid') else None
        nwps_daily = []
        for kind in ('observed', 'forecast'):
            h = nwps.hourly(nwps.parse_series(payload, kind)) if payload else pd.Series(dtype=float)
            if len(h):
                d = h.tz_convert(cfg.get('timezone', 'UTC'))
                d = d.groupby(d.index.date).mean()
                d.index = pd.DatetimeIndex(d.index)
                nwps_daily.append(d)
        nw = pd.concat(nwps_daily) if nwps_daily else pd.Series(dtype=float)
        nw = nw[~nw.index.duplicated(keep='last')].sort_index()
        series[g['site_id']] = splice_on_training_scale(hist, nw)
    routed = get_upstream.routed_daily(series, cfg)
    out['upstream_flow'] = routed.reindex(days)
    return out


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------

def _mirror_disabled_sources(cfg_tc: dict, scalers: Optional[dict] = None) -> None:
    """
    Switch off the same data sources the model was trained without, so it
    sees the placeholders it learned. Older configs don't record this; SMAP
    then counts as off unless Earthdata credentials are present (it cannot be
    fetched without them). Also disables any optional source whose scaler is
    the constant-zero signature (mean == 0.0, scale == 1.0), and skips new
    Phase 1-3 fetches when loading a legacy release whose schema omits them.
    """
    disabled = cfg_tc.get('disabled_sources')
    if disabled is None:
        disabled = [] if os.getenv('EARTHDATA_USERNAME') else ['SMAP']
    disabled = set(disabled)
    if scalers:
        def _is_dead(col: str) -> bool:
            p = scalers.get(col)
            return bool(p and p.get('mean') == 0.0 and p.get('scale') == 1.0)
        if _is_dead('drought_index'):
            disabled.add('DROUGHT')
        if _is_dead('soil_moisture'):
            disabled.add('SMAP')
        if _is_dead('reservoir_storage') and _is_dead('reservoir_release'):
            disabled.add('RESERVOIR')
        if _is_dead('srad') and _is_dead('pet'):
            disabled.add('ENERGY')
        if _is_dead('sm_surface') and _is_dead('precip_max1h'):
            disabled.add('HYDRO_FORCING')
        if _is_dead('swe_mid') and _is_dead('swe_high'):
            disabled.add('GRIDDED_SWE')
        if _is_dead('prior_flow'):
            disabled.add('PRIOR')
    enc_cols = cfg_tc.get('encoder_features')
    dec_cols = cfg_tc.get('decoder_features')
    if enc_cols is not None or dec_cols is not None:
        all_cols = set(enc_cols or ()) | set(dec_cols or ())
        if not (all_cols & set(windowing.ENERGY_FEATURES)):
            disabled.add('ENERGY')
        if not (all_cols & (set(windowing.INTENSITY_FEATURES) | set(windowing.HYDRO_SM_FEATURES))):
            disabled.add('HYDRO_FORCING')
        if not (all_cols & set(windowing.GRIDDED_SWE_FEATURES)):
            disabled.add('GRIDDED_SWE')
        if 'prior_flow' not in all_cols:
            disabled.add('PRIOR')
    for name in sorted(disabled):
        os.environ[f'OPENFLOW_DISABLE_{name}'] = '1'


def prepare_inputs(site_id: str, art: dict, today=None, tz: Optional[str] = None) -> dict:
    """
    Build the five model inputs for `site_id` as of `today` (local date).
    Schema-driven from `art['training_config']` so both legacy releases
    (e.g. model-219) and Phase 1-3 models run seamlessly.
    Returns {'inputs', 'anchor', 'dates', 'info'}; no TensorFlow needed.
    """
    cfg_tc = art['training_config']
    scalers = art.get('scalers') or {}
    enc_cols = list(cfg_tc.get('encoder_features') or windowing.ENCODER_FEATURES)
    dec_cols = list(cfg_tc.get('decoder_features') or windowing.DECODER_FEATURES)
    target_cols = list(cfg_tc.get('target_features') or windowing.TARGET_FEATURES)
    all_model_cols = set(enc_cols) | set(dec_cols)

    # Save/restore env vars for Phase 1-3 sources so alternating legacy and
    # new models within one process never leaks disable flags across calls.
    saved_env = {f'OPENFLOW_DISABLE_{k}': os.environ.get(f'OPENFLOW_DISABLE_{k}')
                 for k in ('ENERGY', 'HYDRO_FORCING', 'GRIDDED_SWE', 'PRIOR')}
    _mirror_disabled_sources(cfg_tc, scalers)
    try:
        enc_days, dec_days = cfg_tc['encoder_days'], cfg_tc['decoder_days']
        reg = art['registry'].get(site_id)
        tz = tz or (reg or {}).get('timezone', 'America/Denver')
        now = pd.Timestamp.now(tz=tz) if today is None else pd.Timestamp(today)
        today = (now.tz_convert(tz).tz_localize(None) if now.tzinfo else now).normalize()
        start = today - pd.Timedelta(days=enc_days + HISTORY_PAD_DAYS)
        prefix, sid = site_id.split(':', 1)
        coords = _coords(site_id)

        flow = _flow(site_id, start.to_pydatetime(), today.to_pydatetime(),
                     nws_lid=(reg or {}).get('nws_lid'), tz=tz)
        if flow is None or flow.empty:
            raise InferenceError(f"no recent flow for {site_id}")
        wx_source = cfg_tc.get('weather_source', 'noaa')
        hydro_source = cfg_tc.get('hydro_source')
        fetched = combine_data.fetch_and_process_data(
            prefix, sid, start.to_pydatetime(), today.to_pydatetime(), flow,
            noaa_fallback=gridmet_point_weather, coords=coords,
            weather_source_override=wx_source,
            hydro_source_override=hydro_source,
            include_historical_prior=False)
        if not fetched or fetched.get('noaa_data') is None or fetched['noaa_data'].empty:
            raise InferenceError(f"no weather data for {site_id} (NOAA GHCND and the gridMET fallback both empty)")

        # Bridge the NOAA/gridMET reporting lag up to the last flow day.
        flow_last = pd.to_datetime(flow['Date']).max().normalize()
        noaa = fetched['noaa_data'].copy()
        noaa['Date'] = pd.to_datetime(noaa['Date'])
        noaa = noaa.groupby('Date')[['TMIN', 'TMAX', 'precipitation']].mean()
        noaa_last = noaa.dropna(subset=['TMIN', 'TMAX']).index.max()
        weather_fc = forecast_weather(coords['lat'], coords['lon'], tz)
        zones_daily = get_basin_precip.inference_daily(site_id, today=today, encoder_days=enc_days,
                                                       decoder_days=dec_days, tz=tz)
        bridged = []
        if pd.notna(noaa_last) and noaa_last < flow_last:
            gap = pd.date_range(noaa_last + pd.Timedelta(days=1), flow_last)
            target = weather_fc.reindex([today]).iloc[0] if today in weather_fc.index else noaa.loc[noaa_last]
            last = noaa.loc[noaa_last]
            for i, d in enumerate(gap, start=1):
                w = i / len(gap)
                local = zones_daily.loc[d, ['rain_local', 'snow_local']].sum() \
                    if d in zones_daily.index and zones_daily.loc[d, 'basin_precip_observed'] else 0.0
                noaa.loc[d] = [(1 - w) * last['TMIN'] + w * target['TMIN'],
                               (1 - w) * last['TMAX'] + w * target['TMAX'], float(local)]
                bridged.append(str(d.date()))
        noaa_df = noaa.reset_index()

        # Live RFC / NWM prior when the loaded model was trained with prior_flow.
        prior_live = pd.DataFrame(columns=['Date', 'prior_flow'])
        want_prior = ('prior_flow' in all_model_cols and 'prior_flow' in scalers
                      and not combine_data._disabled('OPENFLOW_DISABLE_PRIOR'))
        if want_prior:
            try:
                from data import get_nwm_prior
                prior_live = get_nwm_prior.fetch_live_prior(
                    site_id, nws_lid=(reg or {}).get('nws_lid'),
                    usgs_id=(reg or {}).get('usgs_id'), tz=tz)
            except Exception as e:
                logger.warning("Live RFC/NWM prior unavailable for %s: %s", site_id, e)

        merged = combine_data.merge_dataframes(
            noaa_df, flow, site_id, start.to_pydatetime(), today.to_pydatetime(),
            swe_data=fetched['swe_data'], sm_data=fetched['sm_data'],
            drought_data=fetched['drought_data'], reservoir_data=fetched['reservoir_data'],
            huc8=fetched['huc8'], upstream_data=fetched.get('upstream_data'),
            upstream_known_days=fetched.get('upstream_known_days', 0),
            basin_precip_data=fetched.get('basin_precip_data'),
            energy_data=fetched.get('energy_data'),
            hydro_data=fetched.get('hydro_data'),
            gridded_swe_data=fetched.get('gridded_swe_data'),
            prior_data=prior_live if not prior_live.empty else fetched.get('prior_data'))
        if merged.empty:
            raise InferenceError(f"no usable merged rows for {site_id}")
        merged['Date'] = pd.to_datetime(merged['Date'])
        merged = merged.sort_values('Date').reset_index(drop=True)
        # The most recent contiguous encoder window.
        tail = merged.iloc[-enc_days:]
        if len(tail) < enc_days or (tail['Date'].diff().dropna() != pd.Timedelta(days=1)).any():
            raise InferenceError(f"{site_id}: last {enc_days} days are not contiguous after gap handling")
        anchor = tail['Date'].iloc[-1]
        # Latest basin precip (gridMET lags; Stage IV fills) for encoder rows.
        tail = tail.copy()
        zd = zones_daily.reindex(tail['Date'])
        fill = (tail['basin_precip_observed'].to_numpy() == 0) & (zd['basin_precip_observed'].to_numpy() == 1)
        for c in get_basin_precip.FEATURE_COLUMNS:
            tail.loc[fill, c] = zd[c].to_numpy()[fill]
        tail.loc[fill, 'basin_precip_observed'] = 1

        # Decoder rows.
        dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=dec_days, name='Date')
        dec = pd.DataFrame({'Date': dates})
        wx = weather_fc.reindex(dates)
        known_wx = wx[['TMIN', 'TMAX']].notna().all(axis=1)
        last_known = wx[known_wx].iloc[-1] if known_wx.any() else tail.iloc[-1]
        dec['TMIN'] = wx['TMIN'].fillna(last_known['TMIN']).to_numpy()
        dec['TMAX'] = wx['TMAX'].fillna(last_known['TMAX']).to_numpy()
        dec['precipitation'] = wx['precipitation'].fillna(0.0).to_numpy()

        # Extended energy & sub-daily precip intensity from weather_fc.
        for group, obs_col in ((windowing.ENERGY_FEATURES, 'energy_observed'),
                               (windowing.INTENSITY_FEATURES, 'precip_intensity_observed')):
            if all(c in wx.columns for c in group):
                sub = wx[group].apply(pd.to_numeric, errors='coerce')
                ok_mask = sub.notna().all(axis=1)
                for c in group:
                    dec[c] = sub[c].fillna(0.0).to_numpy()
                dec[obs_col] = ok_mask.astype(int).to_numpy()
            else:
                for c in group:
                    dec[c] = 0.0
                dec[obs_col] = 0

        up_cfg = get_upstream.upstream_config(site_id, art['registry'])
        all_up_days = pd.DatetimeIndex(tail['Date']).union(dates)
        up_all = _routed_upstream(site_id, up_cfg, all_up_days, start)
        # Backfill missing recent encoder upstream days (e.g. anchor day when the
        # upstream archive lags behind target flow) from the spliced NWPS series so
        # enc['upstream_observed'][-1] stays valid and upstream_change doesn't zero out.
        up_tail = up_all['upstream_flow'].reindex(pd.DatetimeIndex(tail['Date'])).to_numpy()
        fill_up = (tail['upstream_observed'].to_numpy() == 0) & np.isfinite(up_tail)
        if fill_up.any():
            tail.loc[fill_up, 'upstream_flow'] = up_tail[fill_up]
            tail.loc[fill_up, 'upstream_observed'] = 1
        up = up_all.reindex(dates)
        dec['upstream_flow'] = up['upstream_flow'].to_numpy()
        dec['upstream_observed'] = up['upstream_flow'].notna().astype(int).to_numpy()

        zd = zones_daily.reindex(dates)
        for c in get_basin_precip.FEATURE_COLUMNS:
            dec[c] = zd[c].to_numpy()
        dec['basin_precip_observed'] = zd['basin_precip_observed'].fillna(0).astype(int).to_numpy()

        # Decoder physics/operator prior (RFC + NWM).
        if not prior_live.empty:
            pr_s = (prior_live.assign(Date=pd.to_datetime(prior_live['Date']).dt.normalize())
                    .groupby('Date')['prior_flow'].mean().reindex(dates))
            dec['prior_flow'] = pr_s.to_numpy(dtype=float)
            dec['prior_observed'] = pr_s.notna().astype(int).to_numpy()
        else:
            dec['prior_flow'] = np.nan
            dec['prior_observed'] = 0

        # Scale exactly like training.
        frame = pd.concat([tail, dec], ignore_index=True)
        frame = normalize_data.add_day_of_year_features(frame)
        for c in ('upstream_flow', 'prior_flow', *get_basin_precip.FEATURE_COLUMNS,
                  *windowing.ENERGY_FEATURES, *windowing.INTENSITY_FEATURES):
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors='coerce')

        from data import build_station_attributes as bsa
        station_attrs = art.get('station_attributes') or {}
        flow_units = cfg_tc.get('flow_units', 'cfs')
        area_km2 = None
        if isinstance(cfg_tc.get('station_areas_km2'), dict) and site_id in cfg_tc['station_areas_km2']:
            area_km2 = float(cfg_tc['station_areas_km2'][site_id])
        if not area_km2:
            area_km2 = bsa.station_area_km2(site_id, station_attrs)
        if flow_units == 'mm_day':
            if not area_km2 or area_km2 <= 0.0:
                raise InferenceError(f"Model expects specific discharge (mm/day) but area_km2 missing for {site_id}")
            for col in normalize_data.SPECIFIC_DISCHARGE_COLUMNS:
                if col in frame.columns:
                    frame[col] = normalize_data.cfs_to_specific(
                        pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float), area_km2)

        frame = normalize_data.apply_scalers(frame, scalers)
        # Any optional Phase 1-3 column absent from scalers stays neutral 0.0.
        for col in windowing.OPTIONAL_DATASET_FEATURES:
            if col in frame.columns and col not in scalers and not col.endswith('_observed'):
                frame[col] = 0.0
                ind = normalize_data.SCALER_FIT_MASKS.get(col)
                if ind and ind in frame.columns:
                    frame[ind] = 0
        for col, ind in normalize_data.SCALER_FIT_MASKS.items():
            if col in frame and ind in frame:
                frame.loc[frame[ind].fillna(0) == 0, col] = 0.0
        enc = frame.iloc[:enc_days]
        decf = frame.iloc[enc_days:].reset_index(drop=True)

        enc_X = enc.reindex(columns=enc_cols, fill_value=0.0).to_numpy(dtype=float)
        dec_df = decf.reindex(columns=dec_cols, fill_value=0.0).copy()
        for col in windowing.WINDOW_BUILT_DECODER_FEATURES:
            if col in dec_df.columns:
                dec_df[col] = 0.0
        dec_X = dec_df.to_numpy(dtype=float)

        t_idx = [enc_cols.index(c) for c in target_cols]
        known_days = int(cfg_tc.get('upstream_known_days', {}).get(site_id, 0))
        if 'upstream_flow' in dec_cols and 'upstream_known' in dec_cols:
            up_col = dec_cols.index('upstream_flow')
            up_known_col = dec_cols.index('upstream_known')
            dec_X[:, up_col], dec_X[:, up_known_col] = windowing.upstream_decoder_window(
                decf['upstream_flow'].to_numpy(dtype=float), decf['upstream_observed'].to_numpy(dtype=float),
                known_days, enc['upstream_flow'].iloc[-1], enc['upstream_observed'].iloc[-1])
            if 'upstream_change' in dec_cols:
                uc_mode = cfg_tc.get('upstream_change_mode', 'legacy')
                if uc_mode == 'mass_balance_hold':
                    dec_X[:, dec_cols.index('upstream_change')] = windowing.upstream_change(
                        dec_X[:, up_col], dec_X[:, up_known_col], enc['upstream_flow'].iloc[-1],
                        enc['upstream_observed'].iloc[-1], hold_beyond_known=True,
                        scalers=scalers, anchor_flow_scaled=enc_X[-1, t_idx])
                elif uc_mode == 'scaled_hold':
                    dec_X[:, dec_cols.index('upstream_change')] = windowing.upstream_change(
                        dec_X[:, up_col], dec_X[:, up_known_col], enc['upstream_flow'].iloc[-1],
                        enc['upstream_observed'].iloc[-1], hold_beyond_known=True)
                else:
                    dec_X[:, dec_cols.index('upstream_change')] = windowing.upstream_change(
                        dec_X[:, up_col], dec_X[:, up_known_col], enc['upstream_flow'].iloc[-1],
                        enc['upstream_observed'].iloc[-1], hold_beyond_known=False)

        if all(c in dec_cols for c in windowing.BASIN_PRECIP_FEATURES) and 'basin_precip_known' in dec_cols:
            bp_cols = [dec_cols.index(c) for c in windowing.BASIN_PRECIP_FEATURES]
            bp_known = dec_cols.index('basin_precip_known')
            dec_X[:, bp_cols], dec_X[:, bp_known] = windowing.basin_precip_decoder_window(
                decf[windowing.BASIN_PRECIP_FEATURES].to_numpy(dtype=float),
                decf['basin_precip_observed'].to_numpy(dtype=float),
                int(cfg_tc.get('basin_precip_known_days', windowing.BASIN_PRECIP_KNOWN_DAYS)))

        if all(c in dec_cols for c in windowing.ENERGY_FEATURES) and 'energy_known' in dec_cols:
            en_cols_idx = [dec_cols.index(c) for c in windowing.ENERGY_FEATURES]
            en_known_idx = dec_cols.index('energy_known')
            dec_X[:, en_cols_idx], dec_X[:, en_known_idx] = windowing.masked_decoder_window(
                decf[windowing.ENERGY_FEATURES].to_numpy(dtype=float),
                decf['energy_observed'].to_numpy(dtype=float),
                int(cfg_tc.get('energy_known_days', windowing.ENERGY_KNOWN_DAYS)))

        if all(c in dec_cols for c in windowing.INTENSITY_FEATURES) and 'precip_intensity_known' in dec_cols:
            int_cols_idx = [dec_cols.index(c) for c in windowing.INTENSITY_FEATURES]
            int_known_idx = dec_cols.index('precip_intensity_known')
            dec_X[:, int_cols_idx], dec_X[:, int_known_idx] = windowing.masked_decoder_window(
                decf[windowing.INTENSITY_FEATURES].to_numpy(dtype=float),
                decf['precip_intensity_observed'].to_numpy(dtype=float),
                int(cfg_tc.get('precip_intensity_known_days', windowing.PRECIP_INTENSITY_KNOWN_DAYS)))

        if all(c in dec_cols for c in windowing.ENSEMBLE_SPREAD_FEATURES) and all(c in wx.columns for c in windowing.ENSEMBLE_SPREAD_FEATURES):
            pstd_i = dec_cols.index('precip_ens_std')
            pp90_i = dec_cols.index('precip_ens_p90')
            tstd_i = dec_cols.index('temp_ens_std')
            ek_i = dec_cols.index('ens_spread_known') if 'ens_spread_known' in dec_cols else None
            for d_i in range(dec_days):
                v_pstd = wx['precip_ens_std'].iloc[d_i]
                v_pp90 = wx['precip_ens_p90'].iloc[d_i]
                v_tstd = wx['temp_ens_std'].iloc[d_i]
                if np.isfinite(v_pstd) and np.isfinite(v_pp90) and np.isfinite(v_tstd):
                    s1, s2, s3 = windowing.scale_ensemble_spread(v_pstd, v_pp90, v_tstd, scalers)
                    dec_X[d_i, pstd_i] = s1
                    dec_X[d_i, pp90_i] = s2
                    dec_X[d_i, tstd_i] = s3
                    if ek_i is not None:
                        dec_X[d_i, ek_i] = 1.0

        if 'forcing_is_forecast' in dec_cols:
            dec_X[:, dec_cols.index('forcing_is_forecast')] = 1.0
        if 'lead_day_norm' in dec_cols:
            dec_X[:, dec_cols.index('lead_day_norm')] = np.arange(1, dec_days + 1, dtype=float) / float(dec_days)

        if all(c in dec_cols for c in ('prior_flow', 'prior_known', 'prior_change')):
            pr_col = dec_cols.index('prior_flow')
            pr_known = dec_cols.index('prior_known')
            pr_chg = dec_cols.index('prior_change')
            dec_X[:, pr_col], dec_X[:, pr_known], dec_X[:, pr_chg] = windowing.prior_decoder_window(
                decf['prior_flow'].to_numpy(dtype=float),
                decf['prior_observed'].to_numpy(dtype=float),
                float(enc['prior_flow'].iloc[-1]) if 'prior_flow' in enc.columns else 0.0,
                float(enc['prior_observed'].iloc[-1]) if 'prior_observed' in enc.columns else 0.0,
                known_days=int(cfg_tc.get('prior_known_days', windowing.PRIOR_KNOWN_DAYS)))

        if not (np.isfinite(enc_X).all() and np.isfinite(dec_X).all()):
            bad = [c for c in enc_cols if c in enc.columns and not np.isfinite(enc[c].to_numpy(dtype=float)).all()]
            raise InferenceError(f"non-finite model inputs for {site_id}: {bad}")
        station_idx = int(art['station_index'].get(site_id, 0))
        basin_idx = int(art['basin_index'].get(str(fetched.get('huc8') or ''), 0))
        inputs = {
            'encoder_input': enc_X[None].astype('float32'),
            'decoder_input': dec_X[None].astype('float32'),
            'persistence_input': enc_X[-1, t_idx][None].astype('float32'),
            'station_input': np.array([station_idx], dtype='int32'),
            'basin_input': np.array([basin_idx], dtype='int32'),
        }
        static_feats = cfg_tc.get('static_features') or []
        if static_feats:
            svec = bsa.station_static_vector(
                site_id, station_attrs, cfg_tc.get('static_scalers'), static_feats)
            inputs['static_input'] = svec[None].astype('float32')
        info = {'anchor': str(anchor.date()), 'today': str(today.date()),
                'station_idx': station_idx, 'basin_idx': basin_idx,
                'trained_station': station_idx > 0, 'bridged_weather_days': bridged,
                'weather_forecast_days': int(known_wx.sum()),
                'upstream_known_days': known_days,
                'flow_units': flow_units,
                'area_km2': area_km2}
        return {'inputs': inputs, 'anchor': anchor, 'dates': dates, 'info': info}
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def band_columns(quantiles) -> list:
    """Column names predict() adds for a model with the band, e.g. 'max_q05_cfs'."""
    return [f'{side}_q{round(q * 100):02d}_cfs' for side in ('min', 'max')
            for q in quantiles if q != 0.5]


def predict(art: dict, prepared: dict) -> pd.DataFrame:
    """
    Run the model; DataFrame[Date, min_cfs, max_cfs] for the decoder days.

    A model with the probabilistic band (training_config.quantiles) adds the
    forecast quantiles of the daily low and high, e.g. max_q05_cfs ..
    max_q95_cfs (the 0.5 quantile is min_cfs / max_cfs themselves).
    """
    pred = art['model'].predict(prepared['inputs'], verbose=0)
    if isinstance(pred, dict):
        out = pred['persistence_plus_delta'][0]
        band = pred.get('flow_quantiles')
    else:
        out = pred[0]
        band = None
    tf_ = windowing.TARGET_FEATURES
    sc = art['scalers']
    tc = art.get('training_config') or {}
    flow_units = tc.get('flow_units', 'cfs')
    area_km2 = (prepared.get('info') or {}).get('area_km2')

    def inv(z, col):
        p = sc[col]
        x = z * p['scale'] + p['mean']
        val = np.expm1(x) if p.get('transform') == 'log1p' else x
        if flow_units == 'mm_day' and area_km2:
            val = normalize_data.specific_to_cfs(val, area_km2)
        return val
    lo = inv(out[:, tf_.index('Min Flow')], 'Min Flow')
    hi = inv(out[:, tf_.index('Max Flow')], 'Max Flow')
    swapped = lo > hi
    frame = pd.DataFrame({'Date': prepared['dates'], 'min_cfs': np.where(swapped, hi, lo),
                          'max_cfs': np.where(swapped, lo, hi)})
    quantiles = tc.get('quantiles')
    if band is not None and quantiles:
        min_v = inv(band[0][:, tf_.index('Min Flow'), :], 'Min Flow')
        max_v = inv(band[0][:, tf_.index('Max Flow'), :], 'Max Flow')
        side_values = {
            'min': np.where(swapped[:, None], max_v, min_v),
            'max': np.where(swapped[:, None], min_v, max_v),
        }
        for side in ('min', 'max'):
            values = side_values[side]
            for i, q in enumerate(quantiles):
                if q != 0.5:
                    frame[f'{side}_q{round(q * 100):02d}_cfs'] = values[:, i]
    return frame


def predict_from_feed(art: dict, payload: dict) -> pd.DataFrame:
    """Verify a Phase 4 Daily Feature Feed payload and run `predict()` on it."""
    import feature_feed
    feature_feed.verify_feed_payload(payload, training_config=art.get('training_config'))
    prepared = feature_feed.feed_to_prepared(payload)
    return predict(art, prepared)


def verified_skill(art: dict, site_id: str, leads=(1, 3, 7)) -> Optional[dict]:
    """
    The station's held-out scores from quality_report.json: test windows and
    period, and median absolute % error of the daily max flow on `leads` for
    the model and for persistence ("no change"). None without a report.
    """
    rep = (art.get('quality') or {}).get('stations', {}).get(site_id)
    if not rep or 'error_pct' not in rep:
        return None
    err = rep['error_pct']
    return {'windows': rep['windows'], 'issued': rep.get('issued'),
            'leads': list(leads),
            'model_error_pct': [err['model'][d - 1] for d in leads],
            'persistence_error_pct': [err['persistence'][d - 1] for d in leads],
            'rise_peak_ratio': (rep.get('rises') or {}).get('model', {}).get('peak_ratio')}


def forecast(site_id: str, model_dir: str, today=None, art: Optional[dict] = None) -> dict:
    """End to end: {'daily': DataFrame, 'info': {...}, 'version': str|None, 'verified': dict|None}."""
    art = art or load_artifacts(model_dir)
    prepared = prepare_inputs(site_id, art, today=today)
    daily = predict(art, prepared)
    return {'daily': daily, 'info': prepared['info'], 'version': art.get('version'),
            'verified': verified_skill(art, site_id)}
