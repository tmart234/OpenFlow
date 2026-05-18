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
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    the real fetch without changing callers.
    """
    logger.debug("S2F fetch stubbed for %s @ %s -- archive integration pending",
                 site_id, _to_date(anchor_date))
    return _empty()


def baseline_predictions(test_samples) -> Optional[np.ndarray]:
    """
    Per-sample S2F-derived daily prediction aligned with WindowedSample test
    items. Returns (N, horizon, target_features) on the raw flow scale when
    feasible, or None when the S2F archive isn't wired in (current state).

    See the module docstring for the daily-disaggregation approach this
    expects when fully implemented; until then it's a no-op returning None
    so train.py reports "S2F: not available" and moves on.
    """
    if not test_samples:
        return None
    found = 0
    for sample in test_samples:
        if not fetch(sample.site_id, sample.anchor_date).empty:
            found += 1
    if found == 0:
        return None
    # The disaggregation step (seasonal kAF -> daily cfs via per-site
    # climatological share) belongs here once fetch() returns real data.
    return None
