"""
USBR Snow-to-Flow (S2F) seasonal forecast baseline.

Reclamation publishes monthly seasonal water-supply forecasts (typically
April-July or April-September runoff *volume*) for major reservoirs in the
Upper Colorado and other basins. These are the operational statistical
baselines water managers actually use.

Public API:
    fetch(site_id, anchor_date) -> DataFrame[Date, s2f_volume_kaf]
    baseline_predictions(test_samples) -> Optional[np.ndarray]

IMPLEMENTATION STATUS + TIMESCALE CAVEAT:
    S2F predicts a *seasonal volume* (e.g. April-July total runoff into
    Reservoir X, in kAF). Our model predicts *daily* flow over a 14-day
    horizon. Comparing them apples-to-apples is awkward: you'd have to either

      (a) aggregate our 14-day prediction into a fraction-of-seasonal-volume
          contribution and compare to S2F's fractional progress estimate, or
      (b) disaggregate S2F's seasonal volume into a daily climatological
          shape and compare daily.

    Approach (b) is what this stub assumes when filled in -- given a seasonal
    forecast V_kAF and a per-site climatology of the daily distribution
    inside the forecast season, predict daily_flow[d] = V_kAF * daily_share[d].
    That requires a per-site daily-share lookup that doesn't exist in the
    repo yet; building it is part of the follow-up.

    The data fetch itself: USBR publishes S2F products at
        https://www.usbr.gov/uc/water/crsp/wsf/   (Upper Colorado WSF)
        https://www.usbr.gov/pn/hydromet/forecast.html  (Pacific Northwest)
    as month-by-month CSV / PDFs that need scraping. There's no clean REST
    archive. Until that's wired, fetch() returns empty and
    baseline_predictions() returns None so train.py skips this baseline.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Conversion factor used by disaggregate_seasonal_to_daily: 1 kAF/day in cfs.
# Derivation: 1 acre-foot = 43,560 ft^3; 1 day = 86,400 s; so 1 AF/day =
# 43,560 / 86,400 cfs = 0.50417 cfs. 1,000 AF/day = 504.17 cfs.
_CFS_PER_KAF_PER_DAY = 43_560.0 * 1_000.0 / 86_400.0


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=['Date', 's2f_volume_kaf'])


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


def fetch(site_id: str, anchor_date) -> pd.DataFrame:
    """
    S2F seasonal-volume forecast for `site_id`, as of the month containing
    `anchor_date`. Returns a single-row DataFrame [Date, s2f_volume_kaf]
    when found, or empty otherwise.

    Currently stubbed: USBR S2F doesn't expose a clean REST archive, so this
    always returns empty until the per-basin scraper is built. The function
    signature is the integration point so combine_data / baselines can adopt
    the real fetch without changing callers. When wired, baseline_predictions
    will call disaggregate_seasonal_to_daily below to produce daily-cadence
    predictions in cfs.
    """
    logger.debug("S2F fetch stubbed for %s @ %s -- archive integration pending",
                 site_id, _to_date(anchor_date))
    return _empty()


def disaggregate_seasonal_to_daily(volume_kaf: float,
                                   season_start: date,
                                   season_end: date,
                                   anchor_date,
                                   horizon_days: int,
                                   climatology: Optional[Mapping] = None) -> np.ndarray:
    """
    Convert a seasonal-volume forecast (e.g. "April-July runoff = 850 kAF")
    into a daily-cfs prediction series for the `horizon_days` after
    `anchor_date`.

    Approach: split the seasonal volume across days inside [season_start,
    season_end] using a per-day-of-year share, then convert each daily
    kAF/day allocation to cfs via _CFS_PER_KAF_PER_DAY. Days outside the
    forecast season get the climatological out-of-season flow share (default:
    zero contribution from the seasonal forecast; the persistence baseline
    already covers the rest).

    Args:
        volume_kaf:      seasonal total in thousand acre-feet
        season_start:    first day inside the forecast season
        season_end:      last day inside the forecast season (inclusive)
        anchor_date:     first day of the prediction window (forecast issue
                         date + 1, typically)
        horizon_days:    number of daily predictions to return
        climatology:     optional dict {day_of_year -> fraction_of_seasonal_volume};
                         must sum to ~1.0 over the days in the season.
                         When None, falls back to a uniform share across the
                         season (1 / season_length per in-season day) -- the
                         honest minimum-information baseline.

    Returns: ndarray of shape (horizon_days,) in cfs.
    """
    anchor = _to_date(anchor_date)
    if season_end < season_start:
        raise ValueError("season_end must be on or after season_start")
    season_length = (season_end - season_start).days + 1
    out = np.zeros(horizon_days, dtype='float32')
    for offset in range(horizon_days):
        day = anchor + timedelta(days=offset)
        if day < season_start or day > season_end:
            continue
        if climatology is None:
            share = 1.0 / season_length
        else:
            doy = day.timetuple().tm_yday
            share = float(climatology.get(doy, 1.0 / season_length))
        daily_kaf = volume_kaf * share
        out[offset] = daily_kaf * _CFS_PER_KAF_PER_DAY
    return out


def baseline_predictions(test_samples) -> Optional[np.ndarray]:
    """
    Per-sample S2F-derived daily prediction aligned with WindowedSample test
    items. Returns (N, horizon, target_features) on the raw flow scale when
    feasible, or None when the S2F archive isn't wired in (current state).

    The disaggregation math lives in disaggregate_seasonal_to_daily and is
    ready to use; the missing piece is fetch() returning real data and a
    per-site daily-share climatology lookup. Until then this is a no-op
    returning None so train.py reports "S2F: not available" and moves on.
    """
    if not test_samples:
        return None
    found = 0
    for sample in test_samples:
        if not fetch(sample.site_id, sample.anchor_date).empty:
            found += 1
    if found == 0:
        return None
    # When fetch() returns real data, the loop above would build a
    # (N, horizon, target_features) tensor by calling
    # disaggregate_seasonal_to_daily for each sample and tiling the cfs
    # value across both Min Flow + Max Flow target columns.
    return None
