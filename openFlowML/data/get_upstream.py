"""
Upstream-gauge inputs: the water that is already in the river upstream.

A single-gauge model only sees its own flow history plus local weather, so it
cannot see a flood wave that is still 100+ miles upstream (e.g. the Rio
Grande at Alamosa / Conejos River in Colorado on its way to the Taos Box in
New Mexico). This module turns the upstream gauges listed for a station in
upstream_gauges.json into one daily ML feature:

    upstream_flow(d) = sum_i weight_i * Q_i(d - lag_days_i)

i.e. each upstream gauge's daily-mean flow, shifted by its travel time and
weighted, summed into "the water arriving at the target on day d". It is
NaN on any day where an upstream gauge has no data (after interpolating
short gaps); combine_data turns that into upstream_observed = 0.

Training history comes from each gauge's `site_id` (CO DWR daily archive or
USGS IV aggregated to daily mean). At inference the same routed sum is built
from recent observations plus the NWS RFC forecast at each gauge's `nws_lid`
(see flow_forecast.py and docs/INFERENCE.md).

Set OPENFLOW_DISABLE_UPSTREAM=1 to skip upstream fetches (ablation).
"""

import json
import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'upstream_gauges.json')
# Longest interior gap (days) interpolated in an upstream gauge's daily series
# before routing; longer gaps leave upstream_flow NaN (-> upstream_observed 0).
UPSTREAM_MAX_GAP_DAYS = 7


def load_registry(path: Optional[str] = None) -> Dict[str, dict]:
    """Load the upstream registry, dropping documentation keys ('_doc', ...)."""
    with open(path or REGISTRY_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def resolve_site(name: str, registry: Optional[Dict[str, dict]] = None) -> Tuple[str, dict]:
    """
    Resolve a user-facing station name to (site_id, config).

    Accepts a full site id ('USGS:08276500'), a bare id ('08276500'), an
    alias ('taos-box'), or the target's NWS LID ('TJBN5'); case-insensitive.
    Raises KeyError listing the known stations when nothing matches.
    """
    registry = registry if registry is not None else load_registry()
    want = name.strip().lower()
    for site_id, cfg in registry.items():
        candidates = {site_id.lower(), site_id.split(':', 1)[-1].lower(),
                      str(cfg.get('nws_lid', '')).lower()}
        candidates |= {a.lower() for a in cfg.get('aliases', [])}
        if want in candidates:
            return site_id, cfg
    known = ', '.join(f"{sid} ({', '.join(cfg.get('aliases', [])[:2])})"
                      for sid, cfg in registry.items())
    raise KeyError(f"Unknown station {name!r}. Known: {known}")


def upstream_config(site_id: str, registry: Optional[Dict[str, dict]] = None) -> Optional[dict]:
    """Registry entry for site_id, or None when it has no upstream gauges."""
    registry = registry if registry is not None else load_registry()
    cfg = registry.get(site_id)
    if not cfg or not cfg.get('upstream'):
        return None
    return cfg


def known_days(cfg: Optional[dict], decoder_days: int) -> int:
    """
    Number of leading decoder days whose upstream_flow is known at forecast
    time: the shortest travel time (those days are already observed upstream)
    plus the upstream RFC forecast length. 0 for stations without upstream
    gauges. Capped at decoder_days.
    """
    if not cfg or not cfg.get('upstream'):
        return 0
    lag = min(int(g.get('lag_days', 0)) for g in cfg['upstream'])
    return int(min(decoder_days, max(0, lag + int(cfg.get('forecast_days', 0)))))


def max_lag_days(cfg: Optional[dict]) -> int:
    if not cfg or not cfg.get('upstream'):
        return 0
    lags = [int(g.get('lag_days', 0)) for g in cfg['upstream']]
    if cfg.get('anchor'):
        lags.append(int(cfg['anchor'].get('lag_days', 0)))
    return max(lags)


def gauge_ids(cfg: Optional[dict]) -> List[str]:
    """Every gauge a station's routed upstream needs: upstream gauges, then the anchor."""
    if not cfg or not cfg.get('upstream'):
        return []
    ids = [g['site_id'] for g in cfg['upstream']]
    if cfg.get('anchor'):
        ids.append(cfg['anchor']['site_id'])
    return ids


def interpolate_short_gaps(series: pd.Series, max_gap_days: int) -> pd.Series:
    """
    Time-interpolate interior NaN runs of at most max_gap_days; leave longer
    runs (and edges) NaN. Unlike pandas' interpolate(limit=n), which fills the
    first n days of ANY gap, a long outage stays entirely missing rather than
    getting a fake partial ramp.
    """
    s = pd.Series(series, dtype='float64')
    if s.empty:
        return s
    filled = s.interpolate(method='time', limit_area='inside')
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform('sum')
    return filled.mask(isna & (run_len > max_gap_days))


def _daily(series: pd.Series) -> pd.Series:
    s = pd.Series(series).copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = pd.to_numeric(s, errors='coerce').groupby(level=0).mean().sort_index()
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq='D')
    return interpolate_short_gaps(s.reindex(full), UPSTREAM_MAX_GAP_DAYS)


def routed_daily(series_by_site: Dict[str, pd.Series], cfg: dict) -> pd.Series:
    """
    Route upstream daily-mean series to the target: shift each gauge later by
    its lag_days, scale by its weight, and sum. A day is NaN unless every
    upstream gauge contributes a value. With an `anchor` gauge in cfg the sum
    is only the supply that anchored_daily scales (see there). Pure function
    (no I/O); empty when any needed gauge has no series.
    """
    empty = pd.Series(dtype='float64', name='upstream_flow')
    parts = []
    for gauge in cfg['upstream']:
        s = series_by_site.get(gauge['site_id'])
        if s is None or len(s) == 0:
            return empty
        d = _daily(s)
        d.index = d.index + pd.Timedelta(days=int(gauge.get('lag_days', 0)))
        parts.append(d * float(gauge.get('weight', 1.0)))
    frame = pd.concat(parts, axis=1)
    out = frame.sum(axis=1, min_count=len(parts))
    if cfg.get('anchor'):
        anchor = series_by_site.get(cfg['anchor']['site_id'])
        if anchor is None or len(anchor) == 0:
            return empty
        out = anchored_daily(anchor, out, cfg['anchor'])
    out.name = 'upstream_flow'
    out.index.name = 'Date'
    return out


# Anchor gauges: the recent anchor/supply ratio that scales supply changes.
ANCHOR_RATIO_DAYS = 7
ANCHOR_RATIO_MAX = 1.5
# Diversions are capacity-limited: the ditches can take roughly the first
# DIVERT_CAPACITY_CFS of a rise (that part passes at the recent ratio); beyond
# it at least FLOOD_PASS_THROUGH of the extra water passes. In the late-October
# 2026 rain flood (irrigation season, recent ratio ~0.15) about half of a
# ~6,000 cfs rise reached Lobatos. Coarse values: big rises during irrigation
# season are rare in the record. Per-anchor overrides: `divert_capacity_cfs`,
# `flood_pass_through` in the registry entry.
DIVERT_CAPACITY_CFS = 500.0
FLOOD_PASS_THROUGH = 0.5


def passed_change(supply_change, ratio: float, anchor_cfg: Optional[dict] = None):
    """
    How much of an upstream supply change (cfs, scalar or array) reaches the
    anchor: falls and the first `divert_capacity_cfs` of a rise pass at
    `ratio`; the rest of a rise passes at max(ratio, flood_pass_through).
    """
    cfg = anchor_cfg or {}
    cap = float(cfg.get('divert_capacity_cfs', DIVERT_CAPACITY_CFS))
    flood = max(float(ratio), float(cfg.get('flood_pass_through', FLOOD_PASS_THROUGH)))
    d = np.asarray(supply_change, dtype=float)
    return ratio * np.minimum(d, cap) + flood * np.maximum(d - cap, 0.0)


def anchored_daily(anchor: pd.Series, supply: pd.Series, anchor_cfg: dict) -> pd.Series:
    """
    Upstream flow seen through an anchor gauge: a gauge below the diversions
    between the upstream gauges and the target (Lobatos for the Taos Box, below
    the San Luis Valley irrigation ditches). In irrigation season most of a
    rise at Alamosa never reaches Taos; the anchor shows what actually does.

    anchor: the anchor gauge's flow (any time step; daily-meaned and shifted
    by anchor_cfg lag_days onto target dates). supply: routed_daily of the
    upstream gauges, on target dates (observations, then RFC forecasts).

    Result: the lagged anchor wherever it has a value. Target dates after the
    last anchor value continue it with the supply's change since that date,
    passed through the diversions by passed_change at the pass-through ratio
    anchor/supply over the last ANCHOR_RATIO_DAYS days both have (clipped to
    [0, ANCHOR_RATIO_MAX]), floored at 0. Training data (anchor observed
    every day) is just the lagged anchor; the continuation is how inference
    reaches the RFC forecast days.
    """
    a = _daily(anchor) * float(anchor_cfg.get('weight', 1.0))
    a.index = a.index + pd.Timedelta(days=int(anchor_cfg.get('lag_days', 0)))
    supply = supply.copy()
    supply.index = pd.DatetimeIndex(supply.index).normalize()
    out = a.reindex(a.index.union(supply.index))
    last = a.last_valid_index()
    if last is None or pd.isna(supply.get(last)):
        return out
    both = pd.concat([a, supply], axis=1, keys=['a', 's']).dropna()
    both = both[both.index <= last].tail(ANCHOR_RATIO_DAYS)
    ratio = (float(np.clip(both['a'].mean() / both['s'].mean(), 0.0, ANCHOR_RATIO_MAX))
             if len(both) and both['s'].mean() > 0 else 0.0)
    future = supply[(supply.index > last) & supply.notna()]
    out.loc[future.index] = np.maximum(
        a[last] + passed_change((future - supply[last]).to_numpy(), ratio, anchor_cfg), 0.0)
    return out


def fetch_daily_mean(site_id: str, start_date, end_date) -> pd.Series:
    """Daily-mean flow (cfs) for a DWR:<abbrev> or USGS:<id> gauge, indexed by Date."""
    prefix, sid = site_id.split(':', 1)
    if prefix == 'DWR':
        from data import get_CODWR_flow
        df = get_CODWR_flow.get_historical_data(sid, start_date, end_date)
        # DWR's daily archive is one representative (mean) value per day,
        # surfaced as Min Flow == Max Flow.
        col = 'Min Flow'
    elif prefix == 'USGS':
        from data import get_flow
        df = get_flow.get_daily_mean_flow(sid, start_date, end_date)
        col = 'flow'
    else:
        raise ValueError(f"Unsupported upstream site id prefix: {site_id}")
    if df is None or df.empty:
        return pd.Series(dtype='float64')
    return pd.Series(pd.to_numeric(df[col], errors='coerce').to_numpy(),
                     index=pd.to_datetime(df['Date']), name=site_id)


def get_upstream(site_id: str, start_date, end_date,
                 registry: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """
    DataFrame[Date, upstream_flow] for a target station over [start, end].

    Empty when the station has no upstream gauges, the feature is disabled,
    or any upstream gauge returned no data (the feature then degrades to
    upstream_observed = 0 in combine_data rather than failing the site).
    """
    empty = pd.DataFrame(columns=['Date', 'upstream_flow'])
    if os.getenv('OPENFLOW_DISABLE_UPSTREAM', '').strip() in ('1', 'true', 'True'):
        logger.info("OPENFLOW_DISABLE_UPSTREAM set -- skipping upstream for %s", site_id)
        return empty
    cfg = upstream_config(site_id, registry)
    if cfg is None:
        return empty
    fetch_start = start_date - timedelta(days=max_lag_days(cfg) + UPSTREAM_MAX_GAP_DAYS)
    series = {}
    for gauge_id in gauge_ids(cfg):
        try:
            s = fetch_daily_mean(gauge_id, fetch_start, end_date)
        except Exception as e:
            logger.warning("Upstream fetch failed for %s (%s): %s", site_id, gauge_id, e)
            return empty
        if s.empty:
            logger.warning("No upstream data for %s from %s; upstream feature disabled",
                           site_id, gauge_id)
            return empty
        series[gauge_id] = s
    routed = routed_daily(series, cfg)
    routed = routed[(routed.index >= pd.Timestamp(start_date).normalize())
                    & (routed.index <= pd.Timestamp(end_date).normalize())]
    logger.info("Upstream for %s: %d routed days from %d gauges",
                site_id, int(routed.notna().sum()), len(series))
    return routed.reset_index()
