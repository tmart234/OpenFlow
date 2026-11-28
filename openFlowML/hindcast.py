"""
Hindcast charts for CI training runs.

After train.py evaluates the model on the held-out test set, this picks, per
station, the test window with the highest observed flow (the hardest, most
interesting case: usually a flood or the snowmelt peak) and charts the
model's 14-day daily min-max forecast against what actually happened, with
the forecast peak called out (charts.plot_forecast). The PNGs are uploaded as
a workflow artifact so every training run can be eyeballed, station by
station, before its mobile models ship.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import windowing

logger = logging.getLogger(__name__)

HISTORY_DAYS = 30


def invert(z: np.ndarray, params: dict) -> np.ndarray:
    """Undo normalize_data's z-score (+ log1p) for one column."""
    x = np.asarray(z, dtype=float) * params['scale'] + params['mean']
    return np.expm1(x) if params.get('transform') == 'log1p' else x


def pick_peak_windows(samples: Sequence) -> Dict[str, int]:
    """site_id -> index of that station's sample with the largest observed target (scaled)."""
    best: Dict[str, tuple] = {}
    for i, s in enumerate(samples):
        peak = float(np.max(s.target_Y))
        if s.site_id not in best or peak > best[s.site_id][0]:
            best[s.site_id] = (peak, i)
    return {site: i for site, (_, i) in best.items()}


def hindcast_series(sample, prediction: np.ndarray, scalers: dict,
                    history_days: int = HISTORY_DAYS, band: Optional[np.ndarray] = None,
                    area_km2: Optional[float] = None) -> dict:
    """
    Unscaled dates + flows (cfs) for one sample: encoder history and the
    forecast window. With `band` (H, T, Q) the series also carries the outer
    forecast quantiles: range_low (lowest quantile of the daily min) and
    range_high (highest quantile of the daily max). When `area_km2` is set
    (specific-discharge model), converts inverted mm/day values back to cfs.
    """
    enc_days = sample.encoder_X.shape[0]
    dec_days = sample.target_Y.shape[0]
    dates = pd.date_range(sample.anchor_date, periods=enc_days + dec_days, freq='D')
    i_min = windowing.ENCODER_FEATURES.index('Min Flow')
    i_max = windowing.ENCODER_FEATURES.index('Max Flow')
    t_min = windowing.TARGET_FEATURES.index('Min Flow')
    t_max = windowing.TARGET_FEATURES.index('Max Flow')
    hist = slice(enc_days - history_days, enc_days)

    scale = 1.0
    if area_km2 and float(area_km2) > 0.0:
        import normalize_data
        scale = float(area_km2) / normalize_data.CFS_TO_MM_DAY_PER_KM2

    hist_min = invert(sample.encoder_X[hist, i_min], scalers['Min Flow']) * scale
    hist_max = invert(sample.encoder_X[hist, i_max], scalers['Max Flow']) * scale
    return {
        'history_dates': dates[hist],
        'history_flow': (hist_min + hist_max) / 2,
        'future_dates': dates[enc_days:],
        'observed_min': invert(sample.target_Y[:, t_min], scalers['Min Flow']) * scale,
        'observed_max': invert(sample.target_Y[:, t_max], scalers['Max Flow']) * scale,
        'predicted_min': invert(prediction[:, t_min], scalers['Min Flow']) * scale,
        'predicted_max': invert(prediction[:, t_max], scalers['Max Flow']) * scale,
        **({} if band is None else {
            'range_low': invert(band[:, t_min, 0], scalers['Min Flow']) * scale,
            'range_high': invert(band[:, t_max, -1], scalers['Max Flow']) * scale}),
    }


def write_hindcast_charts(samples: Sequence, predictions: np.ndarray, scalers: dict,
                          out_dir: str, names: Optional[Dict[str, str]] = None,
                          model_version: Optional[str] = None,
                          band: Optional[np.ndarray] = None,
                          quantiles: Optional[Sequence[float]] = None,
                          station_areas_km2: Optional[Dict[str, float]] = None) -> List[str]:
    """
    Write <out_dir>/<site>_hindcast.png per station; returns the paths written.
    With `band` (N, H, T, Q) at `quantiles` the shaded range is the model's
    forecast range instead of its daily min-max.
    """
    import charts
    os.makedirs(out_dir, exist_ok=True)
    names = names or {}
    paths = []
    for site, i in sorted(pick_peak_windows(samples).items()):
        s = samples[i]
        area = (station_areas_km2 or {}).get(site)
        series = hindcast_series(s, predictions[i], scalers,
                                 band=None if band is None else band[i],
                                 area_km2=area)
        if band is not None and quantiles:
            pct = round((max(quantiles) - min(quantiles)) * 100)
            series['band_label'] = f'Model {pct}% forecast range'
            range_note = (f"shaded: {pct}% forecast range (quantile {min(quantiles):g} of the daily min "
                          f"to {max(quantiles):g} of the daily max)")
        else:
            range_note = "daily min–max flow"
        has_upstream = ('upstream_observed' in windowing.ENCODER_FEATURES and
                        s.encoder_X[-1, windowing.ENCODER_FEATURES.index('upstream_observed')] >= 0.5)
        issued = series['history_dates'][-1]
        path = os.path.join(out_dir, f"{site.replace(':', '_')}_hindcast.png")
        charts.plot_hindcast(
            path,
            title=f"{names.get(site, site)}: model hindcast issued {issued:%b %-d, %Y}",
            subtitle=(f"{site} · held-out test window with the highest observed flow "
                      f"· upstream inputs {'on' if has_upstream else 'off'}"),
            footnote=(f"OpenFlow seq2seq LSTM{(' ' + model_version) if model_version else ''}; "
                      f"{range_note}; window not used in training or model selection."),
            **series)
        paths.append(path)
        logger.info("Wrote hindcast chart %s", path)
    return paths
