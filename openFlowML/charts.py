"""
Forecast hydrograph charts: observed flow, forecast with an uncertainty band,
a "Now" marker, the forecast peak called out, optional official NWS forecast
and flood-category lines.

One function, `plot_forecast`, serves both producers:
  - flow_forecast.py (CLI): hourly upstream-routing forecasts
  - train.py (CI):           daily ML hindcasts on held-out test windows

matplotlib is imported lazily with the headless Agg backend, so importing
this module costs nothing and it works on CI runners without a display.
"""

from datetime import timedelta
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Brand-neutral palette (light surface). One hue for the forecast (it is one
# entity); observed is neutral ink; official NWS gets a second, distinct hue.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e6e5e1'
FORECAST = '#2a78d6'
OFFICIAL = '#eb6834'
ML = '#1baf7a'
FLOOD = {'action': '#c98500', 'minor': '#e34948', 'moderate': '#b8322f', 'major': '#7d1f1d'}
# Rain panel: zones are entities 3-5 of the figure (after forecast + official).
ZONE_COLORS = {'local': '#1baf7a', 'mid': '#eda100', 'far': '#e87ba4'}
ZONE_LABELS = {'local': 'local zone', 'mid': 'mid zone', 'far': 'far (headwaters) zone'}


def _local_naive(times, tz: Optional[str]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    if idx.tz is not None:
        idx = idx.tz_convert(tz) if tz else idx
        idx = idx.tz_localize(None)
    return idx


def _is_daily(idx: pd.DatetimeIndex) -> bool:
    if len(idx) < 2:
        return True
    return pd.Series(idx).diff().median() >= pd.Timedelta(hours=23)


def _nice_ceiling(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(v))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10):
        if step * mag >= v:
            return step * mag
    return 10 * mag


def _round_cfs(v: float) -> float:
    if v >= 1000:
        return round(v, -2)
    if v >= 100:
        return round(v, -1)
    return round(v)


def ml_band(dates, lows, highs, start_time=None, start_value=None, step_hours: int = 1):
    """
    A smooth band through a daily ML forecast, for drawing like the routing
    range. Each day's low and high are placed at local noon and joined with
    monotone cubic curves (PCHIP: no overshoot between days), ending at the
    last day's noon (no made-up flat tail). With start_time /
    start_value (the latest observation), the band starts there at zero
    width, so it grows out of the observed line.

    Returns (times, low, high): local naive DatetimeIndex and two arrays.
    """
    from scipy.interpolate import PchipInterpolator
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    lo = np.minimum(np.asarray(lows, dtype=float), np.asarray(highs, dtype=float))
    hi = np.maximum(np.asarray(lows, dtype=float), np.asarray(highs, dtype=float))
    knots = list(days + pd.Timedelta(hours=12))
    k_lo, k_hi = list(lo), list(hi)
    if start_time is not None and start_value is not None and pd.Timestamp(start_time) < knots[0]:
        knots.insert(0, pd.Timestamp(start_time))
        k_lo.insert(0, float(start_value))
        k_hi.insert(0, float(start_value))
    if len(knots) == 1:
        times = pd.date_range(days[0], days[0] + pd.Timedelta(days=1), freq=f'{step_hours}h')
        return times, np.full(len(times), k_lo[0]), np.full(len(times), k_hi[0])
    times = pd.date_range(knots[0], knots[-1], freq=f'{step_hours}h')
    x = np.asarray((pd.DatetimeIndex(knots) - knots[0]) / pd.Timedelta(hours=1), dtype=float)
    xt = np.asarray((times - knots[0]) / pd.Timedelta(hours=1), dtype=float)
    band_lo = PchipInterpolator(x, k_lo)(xt)
    band_hi = PchipInterpolator(x, k_hi)(xt)
    return times, np.minimum(band_lo, band_hi), np.maximum(band_lo, band_hi)


def plot_forecast(out_path: str, *,
                  title: str,
                  obs_times=None, obs_values=None,
                  fc_times=None, fc_central=None,
                  fc_low=None, fc_high=None,
                  now=None,
                  tz: Optional[str] = None,
                  subtitle: Optional[str] = None,
                  footnote: Optional[str] = None,
                  official_times=None, official_values=None,
                  thresholds: Optional[Dict[str, float]] = None,
                  obs_label: str = 'Observed',
                  fc_label: str = 'Forecast (best estimate)',
                  band_label: str = 'Likely range',
                  official_label: str = 'NWS official forecast',
                  units: str = 'cfs',
                  peak_callout: bool = True,
                  xlim=None,
                  rain: Optional[dict] = None,
                  ml: Optional[dict] = None) -> dict:
    """
    Render the chart to `out_path` (PNG). Times may be tz-aware (converted to
    `tz` for display) or naive (taken as already local). `obs_*` may extend
    past `now` (hindcasts: the truth over the forecast window).

    `ml` (optional) overlays a daily ML forecast: {'dates': local naive
    dates, 'min': cfs, 'max': cfs, 'label': str, 'band_label': str} drawn
    as a smooth band through the daily lows and highs (ml_band), starting at
    the latest observation, with a line along its middle.

    `rain` (optional) adds a lower panel of daily basin rain by travel-time
    zone: {'dates': local naive dates, 'zones': {zone: mm/day array},
    'forecast_from': first forecast date or None (forecast bars are faded)}.
    A separate panel, never a second y-axis.

    Returns {'peak_time', 'peak_value', 'callout'} for the called-out point
    (plus 'ml_range': (low, high) cfs when the ML forecast range was drawn)
    (local, naive): the forecast peak when it rises meaningfully above where
    the forecast starts ('peak'), otherwise the forecast's end value on a
    falling / steady river ('end'). Nones when there is no forecast.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    has_rain = rain is not None and len(rain.get('dates', [])) > 0
    if has_rain:
        fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 8.0), dpi=150, sharex=True,
                                      gridspec_kw={'height_ratios': [3.3, 1.0]})
        rax.set_facecolor(SURFACE)
    else:
        fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
        rax = None
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ymax = 0.0
    tmin, tmax = [], []
    peak_info = {'peak_time': None, 'peak_value': None, 'callout': None}
    info_ml_range = None

    if fc_times is not None and fc_central is not None and len(fc_times):
        ft = _local_naive(fc_times, tz)
        fc_central = np.asarray(fc_central, dtype=float)
        if fc_low is not None and fc_high is not None:
            lo = np.asarray(fc_low, dtype=float)
            hi = np.asarray(fc_high, dtype=float)
            ax.fill_between(ft, lo, hi, color=FORECAST, alpha=0.14, lw=0, label=band_label)
            ymax = max(ymax, np.nanmax(hi))
        ax.plot(ft, fc_central, color=FORECAST, lw=2.2, ls=(0, (5, 2.5)), label=fc_label)
        ymax = max(ymax, np.nanmax(fc_central))
        tmin.append(ft.min())
        tmax.append(ft.max())

    if obs_times is not None and obs_values is not None and len(obs_times):
        ot = _local_naive(obs_times, tz)
        ov = np.asarray(obs_values, dtype=float)
        ax.plot(ot, ov, color=INK_2, lw=2, label=obs_label, zorder=3)
        if np.isfinite(ov).any():
            ymax = max(ymax, np.nanmax(ov))
        tmin.append(ot.min())
        tmax.append(ot.max())

    if ml is not None and len(ml.get('dates', [])):
        start_t = start_v = None
        if now is not None and obs_values is not None and np.isfinite(np.asarray(obs_values, float)).any():
            start_t = _local_naive(pd.DatetimeIndex([pd.Timestamp(now)]), tz)[0]
            ov = np.asarray(obs_values, dtype=float)
            start_v = float(ov[np.isfinite(ov)][-1])
        bt, blo, bhi = ml_band(ml['dates'], ml['min'], ml['max'], start_t, start_v)
        if ml.get('range_low') is not None and ml.get('range_high') is not None:
            # The model's forecast range (e.g. 5th percentile of the daily low
            # to 95th of the daily high): a lighter band around the daily one.
            rt, rlo, rhi = ml_band(ml['dates'], ml['range_low'], ml['range_high'], start_t, start_v)
            ax.fill_between(rt, rlo, rhi, color=ML, alpha=0.09, lw=0, zorder=1.5,
                            label=ml.get('range_label', 'ML 90% forecast range'))
            for edge in (rlo, rhi):
                ax.plot(rt, edge, color=ML, lw=0.8, alpha=0.45, ls=':', zorder=1.6)
            ymax = max(ymax, np.nanmax(rhi))
            info_ml_range = (float(np.nanmin(rlo)), float(np.nanmax(rhi)))
        ax.fill_between(bt, blo, bhi, color=ML, alpha=0.20, lw=0, zorder=2,
                        label=ml.get('band_label', 'ML range (daily low–high)'))
        ax.plot(bt, (blo + bhi) / 2, color=ML, lw=2.0, zorder=4,
                label=ml.get('label', 'OpenFlow ML model'))
        ymax = max(ymax, np.nanmax(bhi))
        tmax.append(bt.max())

    if official_times is not None and official_values is not None and len(official_times):
        st = _local_naive(official_times, tz)
        sv = np.asarray(official_values, dtype=float)
        ax.plot(st, sv, color=OFFICIAL, lw=1.8, label=official_label, zorder=4)
        if np.isfinite(sv).any():
            ymax = max(ymax, np.nanmax(sv))

    if fc_times is not None and fc_central is not None and len(fc_times) and peak_callout:
        ft = _local_naive(fc_times, tz)
        fmt = '%a %b %-d' if _is_daily(ft) else '%a %b %-d, ~%-I %p'
        start_v = float(fc_central[0])
        i = int(np.nanargmax(fc_central))
        # A "peak" is only worth calling out if it is meaningfully above where
        # the forecast starts; on a falling river call out where it ends up.
        rising = len(fc_central) > 1 and float(fc_central[i]) > 1.02 * start_v
        if not rising:
            i = len(fc_central) - 1
        pt, pv = ft[i], float(fc_central[i])
        tz_abbr = ''
        ts_i = pd.DatetimeIndex(pd.to_datetime(fc_times))[i]
        if tz and ts_i.tzinfo is not None and not _is_daily(ft):
            tz_abbr = ' ' + ts_i.tz_convert(tz).strftime('%Z')
        span = (ft.max() - ft.min()) if len(ft) > 1 else timedelta(days=1)
        text_x = pt + span * 0.08
        if text_x > ft.max():
            text_x = pt - span * 0.25
        if rising:
            label = f"Peak ≈ {_round_cfs(pv):,.0f} {units}\n{pt.strftime(fmt)}{tz_abbr}"
        else:
            verb = 'Falling to' if pv < 0.95 * start_v else 'Holding near'
            # A forecast ending at local midnight ends the previous day.
            day = pt - pd.Timedelta(minutes=1) if (pt.hour, pt.minute) == (0, 0) else pt
            label = f"{verb} ≈ {_round_cfs(pv):,.0f} {units}\nby {day.strftime('%a %b %-d')}"
        label_y = pv + 0.15 * max(ymax, 1.0)
        ax.annotate(label, xy=(pt, pv), xytext=(text_x, label_y),
                    fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle='-', color=INK_2, lw=1))
        ax.scatter([pt], [pv], s=40, color=FORECAST, zorder=5, edgecolor=SURFACE, linewidth=2)
        peak_info = {'peak_time': pt, 'peak_value': pv, 'callout': 'peak' if rising else 'end'}

    top = _nice_ceiling(ymax * 1.15)
    if peak_info['peak_value'] is not None:
        # Room for the two-line callout above the peak.
        top = max(top, _nice_ceiling(label_y * 1.15))
    for name, value in sorted((thresholds or {}).items(), key=lambda kv: kv[1]):
        # Only draw categories the forecast is getting near; far-away flood
        # lines would squash the hydrograph.
        if value <= top * 1.05:
            color = FLOOD.get(name, MUTED)
            ax.axhline(value, color=color, lw=1, ls=(0, (2, 2)), zorder=1)
            ax.text(1.0, value, f" {name.title()} {value:,.0f}", transform=ax.get_yaxis_transform(),
                    color=color, fontsize=8.5, va='center', ha='left', clip_on=False)
            top = max(top, _nice_ceiling(value * 1.1))

    if now is not None:
        nt = _local_naive([now], tz)[0]
        ax.axvline(nt, color=MUTED, lw=1)
        if rax is not None:
            rax.axvline(nt, color=MUTED, lw=1)
        daily = (fc_times is None or len(fc_times) == 0
                 or _is_daily(_local_naive(fc_times, tz)))
        label = nt.strftime('%a %b %-d') if daily else nt.strftime('%a %-I %p')
        ax.text(nt, top * 0.98, f" Now  {label}", color=INK_2, fontsize=9.5, va='top')

    ax.set_ylim(0, top)
    peak_info['y_top'] = top
    if xlim is not None:
        ax.set_xlim(*[_local_naive([x], tz)[0] for x in xlim])
    elif tmin and tmax:
        ax.set_xlim(min(tmin), max(tmax))
    x0, x1 = ax.get_xlim()
    days = mdates.num2date(x1) - mdates.num2date(x0)
    if days <= timedelta(days=21):
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%b %-d'))
        for d in pd.date_range(pd.Timestamp(mdates.num2date(x0)).tz_localize(None).normalize(),
                               pd.Timestamp(mdates.num2date(x1)).tz_localize(None), freq='D'):
            ax.axvline(d, color=GRID, lw=0.8, zorder=0)
            if rax is not None:
                rax.axvline(d, color=GRID, lw=0.8, zorder=0)
    elif days <= timedelta(days=90):
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_ylabel(f"Flow ({units})", color=INK_2)
    fig.suptitle(title, x=0.06, ha='left', fontsize=13.5, color=INK, weight='bold')
    if subtitle:
        ax.set_title(subtitle, loc='left', fontsize=10, color=INK_2)
    # On the chart's own background, so a flood-threshold line running behind
    # the legend (it grows to six rows with the ML overlays) does not strike
    # through its labels.
    ax.legend(loc='upper left', frameon=True, facecolor=SURFACE, edgecolor='none',
              framealpha=0.92, fontsize=10, labelcolor=INK_2).set_zorder(10)
    if rax is not None:
        _rain_bars(rax, rain)
    bottom = 0.0
    if footnote:
        fig.text(0.06, 0.015, footnote, fontsize=8.3, color=INK_2, va='bottom')
        bottom = 0.03 + 0.022 * footnote.count('\n')
    fig.tight_layout(rect=(0, bottom + 0.02, 1, 0.97), h_pad=0.4)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    if info_ml_range is not None:
        peak_info['ml_range'] = info_ml_range
    return peak_info


def _rain_bars(rax, rain: dict) -> None:
    """Grouped daily bars, one per zone; forecast days faded."""
    from matplotlib.ticker import FuncFormatter
    dates = pd.DatetimeIndex(rain['dates'])
    fc_from = rain.get('forecast_from')
    is_fc = (dates >= pd.Timestamp(fc_from)) if fc_from is not None else np.zeros(len(dates), bool)
    zones = [z for z in ('local', 'mid', 'far') if z in rain['zones']]
    width = 0.26
    top = 0.0
    for k, z in enumerate(zones):
        vals = np.nan_to_num(np.asarray(rain['zones'][z], dtype=float))
        x = dates + pd.Timedelta(hours=12 + (k - (len(zones) - 1) / 2) * width * 24)
        rax.bar(x[~is_fc], vals[~is_fc], width=width * 0.92, color=ZONE_COLORS[z],
                label=ZONE_LABELS[z], zorder=2)
        if is_fc.any():
            rax.bar(x[is_fc], vals[is_fc], width=width * 0.92, color=ZONE_COLORS[z],
                    alpha=0.4, zorder=2)
        top = max(top, vals.max() if len(vals) else 0)
    rax.set_ylim(0, max(5.0, _nice_ceiling(top * 1.2)))
    rax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    rax.grid(axis='y', color=GRID, lw=0.8)
    rax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        rax.spines[side].set_visible(False)
    rax.spines['bottom'].set_color(MUTED)
    rax.tick_params(colors=INK_2, length=0)
    rax.set_ylabel('Rain (mm/day)', color=INK_2)
    rax.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3)
    rax.text(1.0, 1.0, 'Basin-mean rain by river-distance zone (gridMET / NWS Stage IV; NWS forecast faded)',
             transform=rax.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK_2)


def plot_hindcast(out_path: str, *, title: str, history_dates: Sequence,
                  history_flow: Sequence[float], future_dates: Sequence,
                  observed_min: Sequence[float], observed_max: Sequence[float],
                  predicted_min: Sequence[float], predicted_max: Sequence[float],
                  subtitle: Optional[str] = None, footnote: Optional[str] = None,
                  range_low: Optional[Sequence[float]] = None,
                  range_high: Optional[Sequence[float]] = None,
                  band_label: str = 'Model daily min–max') -> dict:
    """
    Daily ML hindcast chart: encoder-window history + observed truth over the
    forecast window (daily midpoint of Min/Max Flow) against the model's
    predicted daily range (band) and midpoint (dashed), with the forecast
    peak called out. The "Now" line marks the forecast issue day. With
    range_low / range_high (the probabilistic band) those bound the shading.
    """
    hist_t = pd.DatetimeIndex(pd.to_datetime(history_dates))
    fut_t = pd.DatetimeIndex(pd.to_datetime(future_dates))
    obs_mid = (np.asarray(observed_min, float) + np.asarray(observed_max, float)) / 2
    p_lo = np.asarray(predicted_min, float)
    p_hi = np.asarray(predicted_max, float)
    central = (p_lo + p_hi) / 2
    if range_low is not None and range_high is not None:
        p_lo, p_hi = np.asarray(range_low, float), np.asarray(range_high, float)
    return plot_forecast(
        out_path, title=title, subtitle=subtitle, footnote=footnote,
        obs_times=hist_t.append(fut_t),
        obs_values=np.concatenate([np.asarray(history_flow, float), obs_mid]),
        fc_times=fut_t, fc_central=central, fc_low=p_lo, fc_high=p_hi,
        now=hist_t[-1], obs_label='Observed (daily mean of min/max)',
        fc_label='Model forecast (midpoint)', band_label=band_label)
