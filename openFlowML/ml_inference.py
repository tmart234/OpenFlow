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
    art['model_dir'] = model_dir
    art['version'] = None
    manifest = os.path.join(model_dir, 'manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as f:
            art['version'] = json.load(f).get('model_version')
    if with_model:
        import tensorflow as tf
        import model as model_mod
        art['model'] = tf.keras.models.load_model(
            os.path.join(model_dir, 'lstm_model.h5'),
            custom_objects={'StationDropout': model_mod.StationDropout}, compile=False)
    return art


# --------------------------------------------------------------------------
# Data assembly helpers
# --------------------------------------------------------------------------

def _coords(site_id: str) -> dict:
    prefix, sid = site_id.split(':', 1)
    c = get_usgs_coordinates(sid) if prefix == 'USGS' else get_dwr_coordinates(sid)
    if not c:
        raise InferenceError(f"no coordinates for {site_id}")
    return {'lat': float(c['latitude']), 'lon': float(c['longitude'])}


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
    daily = local.groupby(local.index.date).agg(['min', 'max'])
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily[(daily.index > last) & (daily.index <= pd.Timestamp(end).normalize())]
    if daily.empty:
        return df
    extra = pd.DataFrame({'Date': daily.index.strftime('%Y-%m-%d'),
                          'Min Flow': daily['min'].to_numpy(), 'Max Flow': daily['max'].to_numpy()})
    logger.info("Topped up %s flow with %d NWPS day(s)", site_id, len(extra))
    return pd.concat([df, extra], ignore_index=True)


def gridmet_point_weather(lat: float, lon: float, start, end) -> pd.DataFrame:
    """NOAA-shaped daily weather from the gridMET cell at a point (tenths deg C, mm)."""
    pad = 0.05
    out = {}
    for key, name in (('tmmn', 'TMIN'), ('tmmx', 'TMAX'), ('pr', 'precipitation')):
        dates, la, lo, v = get_basin_precip._gridmet_subset(
            key, lat - pad, lat + pad, lon - pad, lon + pad, start, end)
        iy, ix = int(np.argmin(np.abs(la - lat))), int(np.argmin(np.abs(lo - lon)))
        series = v[:, iy, ix]
        out[name] = (series - 273.15) * 10.0 if key != 'pr' else series
        out['Date'] = dates
    return pd.DataFrame(out)[['Date', 'TMIN', 'TMAX', 'precipitation']]


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
    """Daily forecast weather in training units; Open-Meteo first, NWS gridpoint fallback."""
    try:
        from data import get_forecast
        om = get_forecast.get_open_meteo_forecast(lat, lon, days=days, temperature_unit='celsius')
        if om is not None and not om.empty:
            om = om.set_index(pd.DatetimeIndex(pd.to_datetime(om['Date']), name='Date'))
            om[['TMIN', 'TMAX']] = om[['TMIN', 'TMAX']].astype(float) * 10.0
            return om[['TMIN', 'TMAX', 'precipitation']].astype(float)
    except Exception as e:
        logger.info("Open-Meteo forecast unavailable (%s); using NWS", e)
    return nws_point_daily(lat, lon, tz)


def _routed_upstream(site_id: str, cfg: Optional[dict], days: pd.DatetimeIndex,
                     history_start) -> pd.DataFrame:
    """Routed upstream_flow + observed flag for `days` from observed daily means + RFC forecasts."""
    out = pd.DataFrame({'upstream_flow': np.nan}, index=days)
    if not cfg or not cfg.get('upstream'):
        return out
    series = {}
    for g in cfg['upstream']:
        hist = get_upstream.fetch_daily_mean(g['site_id'], history_start, days[-1])
        payload = nwps.get_stageflow(g['nws_lid']) if g.get('nws_lid') else None
        parts = [hist]
        for kind in ('observed', 'forecast'):
            h = nwps.hourly(nwps.parse_series(payload, kind)) if payload else pd.Series(dtype=float)
            if len(h):
                d = h.tz_convert(cfg.get('timezone', 'UTC'))
                d = d.groupby(d.index.date).mean()
                d.index = pd.DatetimeIndex(d.index)
                parts.append(d)
        combined = pd.concat([p for p in parts if len(p)])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        series[g['site_id']] = combined
    routed = get_upstream.routed_daily(series, cfg)
    out['upstream_flow'] = routed.reindex(days)
    return out


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------

def _mirror_disabled_sources(cfg_tc: dict) -> None:
    """
    Switch off the same data sources the model was trained without, so it
    sees the placeholders it learned. Older configs don't record this; SMAP
    then counts as off unless Earthdata credentials are present (it cannot be
    fetched without them).
    """
    disabled = cfg_tc.get('disabled_sources')
    if disabled is None:
        disabled = [] if os.getenv('EARTHDATA_USERNAME') else ['SMAP']
    for name in disabled:
        os.environ[f'OPENFLOW_DISABLE_{name}'] = '1'


def prepare_inputs(site_id: str, art: dict, today=None, tz: Optional[str] = None) -> dict:
    """
    Build the five model inputs for `site_id` as of `today` (local date).
    Returns {'inputs', 'anchor', 'dates', 'info'}; no TensorFlow needed.
    """
    cfg_tc = art['training_config']
    _mirror_disabled_sources(cfg_tc)
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
    fetched = combine_data.fetch_and_process_data(prefix, sid, start.to_pydatetime(),
                                                  today.to_pydatetime(), flow,
                                                  noaa_fallback=gridmet_point_weather, coords=coords)
    if not fetched or fetched.get('noaa_data') is None or fetched['noaa_data'].empty:
        raise InferenceError(f"no weather data for {site_id} (NOAA GHCND and the gridMET fallback both empty)")

    # Bridge the NOAA reporting lag up to the last flow day.
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
            w = i / (len(gap) + 1)
            local = zones_daily.loc[d, ['rain_local', 'snow_local']].sum() \
                if d in zones_daily.index and zones_daily.loc[d, 'basin_precip_observed'] else 0.0
            noaa.loc[d] = [(1 - w) * last['TMIN'] + w * target['TMIN'],
                           (1 - w) * last['TMAX'] + w * target['TMAX'], float(local)]
            bridged.append(str(d.date()))
    noaa_df = noaa.reset_index()

    merged = combine_data.merge_dataframes(
        noaa_df, flow, site_id, start.to_pydatetime(), today.to_pydatetime(),
        swe_data=fetched['swe_data'], sm_data=fetched['sm_data'],
        drought_data=fetched['drought_data'], reservoir_data=fetched['reservoir_data'],
        huc8=fetched['huc8'], upstream_data=fetched.get('upstream_data'),
        upstream_known_days=fetched.get('upstream_known_days', 0),
        basin_precip_data=fetched.get('basin_precip_data'))
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
    up_cfg = get_upstream.upstream_config(site_id, art['registry'])
    up = _routed_upstream(site_id, up_cfg, dates, start)
    dec['upstream_flow'] = up['upstream_flow'].to_numpy()
    dec['upstream_observed'] = up['upstream_flow'].notna().astype(int).to_numpy()
    zd = zones_daily.reindex(dates)
    for c in get_basin_precip.FEATURE_COLUMNS:
        dec[c] = zd[c].to_numpy()
    dec['basin_precip_observed'] = zd['basin_precip_observed'].fillna(0).astype(int).to_numpy()

    # Scale exactly like training.
    frame = pd.concat([tail, dec], ignore_index=True)
    frame = normalize_data.add_day_of_year_features(frame)
    for c in ('upstream_flow', *get_basin_precip.FEATURE_COLUMNS):
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    frame = normalize_data.apply_scalers(frame, art['scalers'])
    for col, ind in normalize_data.SCALER_FIT_MASKS.items():
        if col in frame and ind in frame:
            frame.loc[frame[ind].fillna(0) == 0, col] = 0.0
    enc = frame.iloc[:enc_days]
    decf = frame.iloc[enc_days:].reset_index(drop=True)

    enc_X = enc[windowing.ENCODER_FEATURES].to_numpy(dtype=float)
    dec_X = decf.reindex(columns=windowing.DECODER_FEATURES).to_numpy(dtype=float)
    up_col = windowing.DECODER_FEATURES.index('upstream_flow')
    up_known_col = windowing.DECODER_FEATURES.index('upstream_known')
    known_days = int(cfg_tc.get('upstream_known_days', {}).get(site_id, 0))
    dec_X[:, up_col], dec_X[:, up_known_col] = windowing.upstream_decoder_window(
        decf['upstream_flow'].to_numpy(dtype=float), decf['upstream_observed'].to_numpy(dtype=float),
        known_days, enc['upstream_flow'].iloc[-1], enc['upstream_observed'].iloc[-1])
    if 'upstream_change' in windowing.DECODER_FEATURES:
        dec_X[:, windowing.DECODER_FEATURES.index('upstream_change')] = windowing.upstream_change(
            dec_X[:, up_col], dec_X[:, up_known_col], enc['upstream_flow'].iloc[-1],
            enc['upstream_observed'].iloc[-1])
    bp_cols = [windowing.DECODER_FEATURES.index(c) for c in windowing.BASIN_PRECIP_FEATURES]
    bp_known = windowing.DECODER_FEATURES.index('basin_precip_known')
    dec_X[:, bp_cols], dec_X[:, bp_known] = windowing.basin_precip_decoder_window(
        decf[windowing.BASIN_PRECIP_FEATURES].to_numpy(dtype=float),
        decf['basin_precip_observed'].to_numpy(dtype=float),
        int(cfg_tc.get('basin_precip_known_days', windowing.BASIN_PRECIP_KNOWN_DAYS)))
    if not (np.isfinite(enc_X).all() and np.isfinite(dec_X).all()):
        bad = [c for c in windowing.ENCODER_FEATURES if not np.isfinite(enc[c].to_numpy(dtype=float)).all()]
        raise InferenceError(f"non-finite model inputs for {site_id}: {bad}")
    t_idx = [windowing.ENCODER_FEATURES.index(c) for c in windowing.TARGET_FEATURES]
    station_idx = int(art['station_index'].get(site_id, 0))
    basin_idx = int(art['basin_index'].get(str(fetched.get('huc8') or ''), 0))
    inputs = {
        'encoder_input': enc_X[None].astype('float32'),
        'decoder_input': dec_X[None].astype('float32'),
        'persistence_input': enc_X[-1, t_idx][None].astype('float32'),
        'station_input': np.array([station_idx], dtype='int32'),
        'basin_input': np.array([basin_idx], dtype='int32'),
    }
    info = {'anchor': str(anchor.date()), 'today': str(today.date()),
            'station_idx': station_idx, 'basin_idx': basin_idx,
            'trained_station': station_idx > 0, 'bridged_weather_days': bridged,
            'weather_forecast_days': int(known_wx.sum()),
            'upstream_known_days': known_days}
    return {'inputs': inputs, 'anchor': anchor, 'dates': dates, 'info': info}


def predict(art: dict, prepared: dict) -> pd.DataFrame:
    """Run the model; DataFrame[Date, min_cfs, max_cfs] for the decoder days."""
    out = art['model'].predict(prepared['inputs'], verbose=0)[0]
    tf_ = windowing.TARGET_FEATURES
    sc = art['scalers']

    def inv(z, col):
        p = sc[col]
        x = z * p['scale'] + p['mean']
        return np.expm1(x) if p.get('transform') == 'log1p' else x
    lo = inv(out[:, tf_.index('Min Flow')], 'Min Flow')
    hi = inv(out[:, tf_.index('Max Flow')], 'Max Flow')
    return pd.DataFrame({'Date': prepared['dates'], 'min_cfs': np.minimum(lo, hi),
                         'max_cfs': np.maximum(lo, hi)})


def forecast(site_id: str, model_dir: str, today=None, art: Optional[dict] = None) -> dict:
    """End to end: {'daily': DataFrame, 'info': {...}, 'version': str|None}."""
    art = art or load_artifacts(model_dir)
    prepared = prepare_inputs(site_id, art, today=today)
    daily = predict(art, prepared)
    return {'daily': daily, 'info': prepared['info'], 'version': art.get('version')}
