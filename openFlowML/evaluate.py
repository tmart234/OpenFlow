"""
Forecast verification for a trained OpenFlow model.

Scores the model on held-out test windows in real flow units (cfs), per
station and lead day, against two reference forecasts any forecast must beat
to be worth using:

  persistence  tomorrow (and every later day) equals today's flow
  upstream     today's flow plus the change in routed upstream flow on the
               days it is known, then held (stations with upstream gauges;
               elsewhere identical to persistence)

What is scored (daily maximum flow, the number a paddler reads):

  error        median absolute % error, by lead day
  skill        1 - error(model) / error(persistence), by lead day
  rises        windows whose observed 7-day peak is >= 1.3x the flow at issue
               time: median forecast/observed peak ratio and median absolute
               peak timing error (days)
  recessions   windows falling at issue time (last 2 days down): median
               absolute % error on days 3 and 7

quality_gate() turns the pooled report into pass/fail checks so CI can refuse
to publish a model that is worse than "no change" or that flattens floods.
Run standalone on a trained run with:

  python evaluate.py --base-path <run dir>       # writes quality_report.*
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import windowing
from hindcast import invert

logger = logging.getLogger(__name__)

RISE_FACTOR = 1.3          # observed 7-day peak / flow at issue time
EVENT_DAYS = 7             # rise / recession look-ahead
REPORT_LEADS = (1, 2, 3, 5, 7, 10, 14)

# Quality gate thresholds (pooled over all stations).
GATE = {
    # Day-1 error may exceed persistence by at most this factor.
    'day1_max_ratio_to_persistence': 1.10,
    # From day 3 on, the model must beat persistence on median error.
    'min_skill_from_day': 3,
    # Rising events: median forecast peak / observed peak must reach this.
    'min_rise_peak_ratio': 0.80,
    # ...and the median peak timing error must be within this many days.
    'max_rise_timing_days': 1.0,
}


def forecasts(samples: Sequence, model_pred: np.ndarray, scalers: dict) -> dict:
    """
    Unscaled (cfs) daily-max forecasts and observations for each window:
    obs, model, persistence, upstream as (n, H) arrays, plus q0 (flow at issue
    time), falling (bool, issue-time trend), site and issue date per window.
    """
    x, y = windowing.stack(list(samples))
    t = windowing.TARGET_FEATURES.index('Max Flow')
    enc, dec = windowing.ENCODER_FEATURES, windowing.DECODER_FEATURES
    q = lambda z: invert(z, scalers['Max Flow'])
    obs = q(y[..., t])
    model = q(np.asarray(model_pred)[..., t])
    enc_q = q(x['encoder_input'][..., enc.index('Max Flow')])
    q0 = enc_q[:, -1]
    horizon = obs.shape[1]
    persistence = np.repeat(q0[:, None], horizon, axis=1)

    upstream = persistence.copy()
    if 'upstream_flow' in scalers and 'upstream_flow' in dec:
        up = invert(x['decoder_input'][..., dec.index('upstream_flow')], scalers['upstream_flow'])
        up0 = invert(x['encoder_input'][:, -1, enc.index('upstream_flow')], scalers['upstream_flow'])
        known = x['decoder_input'][..., dec.index('upstream_known')] > 0.5
        obs0 = x['encoder_input'][:, -1, enc.index('upstream_observed')] > 0.5
        step = np.where(known & obs0[:, None], np.maximum(q0[:, None] + up - up0[:, None], 1.0), np.nan)
        step = pd.DataFrame(step.T).ffill().to_numpy().T          # hold the last known value
        upstream = np.where(np.isfinite(step), step, persistence)

    return {
        'obs': obs, 'model': model, 'persistence': persistence, 'upstream': upstream,
        'q0': q0, 'falling': enc_q[:, -1] < enc_q[:, -3],
        'site': np.array([s.site_id for s in samples]),
        'issued': np.array([s.anchor_date + pd.Timedelta(days=x['encoder_input'].shape[1] - 1)
                            for s in samples]),
    }


def _pct_err(pred, obs):
    return np.abs(pred / np.maximum(obs, 1e-6) - 1.0)


def _score(fc: dict, mask: np.ndarray, names=('model', 'persistence', 'upstream')) -> dict:
    obs = fc['obs'][mask]
    n, horizon = obs.shape
    out = {'windows': int(n)}
    if n == 0:
        return out
    err = {k: np.median(_pct_err(fc[k][mask], obs), axis=0) for k in names}
    out['error_pct'] = {k: [round(float(v) * 100, 2) for v in err[k]] for k in names}
    out['skill'] = [round(float(1 - m / p), 3) if p > 0 else None
                    for m, p in zip(err['model'], err['persistence'])]

    look = min(EVENT_DAYS, horizon)
    peak = obs[:, :look].max(axis=1)
    rise = peak >= RISE_FACTOR * fc['q0'][mask]
    rises = {'windows': int(rise.sum())}
    if rise.any():
        t_obs = obs[rise, :look].argmax(axis=1)
        for k in names:
            p = fc[k][mask][rise, :look]
            rises[k] = {'peak_ratio': round(float(np.median(p.max(axis=1) / peak[rise])), 3),
                        'timing_days': round(float(np.median(np.abs(p.argmax(axis=1) - t_obs))), 2)}
    out['rises'] = rises

    fall = fc['falling'][mask]
    rec = {'windows': int(fall.sum())}
    if fall.any():
        for k in names:
            e = _pct_err(fc[k][mask][fall], obs[fall])
            rec[k] = {f'day{d}_error_pct': round(float(np.median(e[:, d - 1])) * 100, 2)
                      for d in (3, 7) if d <= horizon}
    out['recessions'] = rec
    return out


def build_report(fc: dict, model_version: Optional[str] = None) -> dict:
    """Pooled + per-station scores (see module docstring)."""
    report = {'model_version': model_version, 'leads': list(range(1, fc['obs'].shape[1] + 1)),
              'all': _score(fc, np.ones(len(fc['obs']), dtype=bool)), 'stations': {}}
    for site in sorted(set(fc['site'])):
        m = fc['site'] == site
        report['stations'][site] = _score(fc, m)
        report['stations'][site]['issued'] = [str(pd.Timestamp(fc['issued'][m].min()).date()),
                                              str(pd.Timestamp(fc['issued'][m].max()).date())]
    report['gate'] = quality_gate(report)
    return report


def quality_gate(report: dict, gate: Optional[dict] = None) -> dict:
    """Pass/fail checks on the pooled scores; {'passed': bool, 'checks': [...]}."""
    g = dict(GATE, **(gate or {}))
    a = report['all']
    checks = []

    def check(name, ok, detail):
        checks.append({'check': name, 'passed': bool(ok), 'detail': detail})

    if 'error_pct' in a:
        m1, p1 = a['error_pct']['model'][0], a['error_pct']['persistence'][0]
        check('day 1 close to persistence', m1 <= p1 * g['day1_max_ratio_to_persistence'],
              f'model {m1:.1f}% vs persistence {p1:.1f}% (limit x{g["day1_max_ratio_to_persistence"]})')
        start = g['min_skill_from_day']
        losing = [d for d, s in enumerate(a['skill'], start=1) if d >= start and s is not None and s <= 0]
        check(f'beats persistence from day {start}', not losing,
              'all days' if not losing else f'loses on day(s) {losing}')
    r = a.get('rises', {})
    if r.get('windows'):
        ratio, timing = r['model']['peak_ratio'], r['model']['timing_days']
        check('rising-flow peaks not flattened', ratio >= g['min_rise_peak_ratio'],
              f'median peak ratio {ratio:.2f} (min {g["min_rise_peak_ratio"]})')
        check('rising-flow peak timing', timing <= g['max_rise_timing_days'],
              f'median timing error {timing:.1f} d (max {g["max_rise_timing_days"]})')
    return {'passed': all(c['passed'] for c in checks), 'checks': checks}


def to_markdown(report: dict, names: Optional[Dict[str, str]] = None) -> str:
    """Human-readable report (release notes / CI summary)."""
    names = names or {}
    leads = [d for d in REPORT_LEADS if d <= len(report['leads'])]
    lines = [f"# OpenFlow forecast quality{' — ' + report['model_version'] if report.get('model_version') else ''}",
             '',
             'Median absolute % error of the daily maximum flow on held-out test windows '
             '(model / persistence). Lower is better.', '',
             '| Station | Windows | ' + ' | '.join(f'Day {d}' for d in leads) + ' | Rise peak ratio | Rise timing (d) |',
             '|---|---|' + '---|' * len(leads) + '---|---|']

    def row(label, s):
        if 'error_pct' not in s:
            return f'| {label} | 0 |' + ' |' * (len(leads) + 2)
        cells = [f"{s['error_pct']['model'][d - 1]:.1f} / {s['error_pct']['persistence'][d - 1]:.1f}"
                 for d in leads]
        r = s.get('rises', {})
        ratio = f"{r['model']['peak_ratio']:.2f}" if r.get('windows') else '—'
        timing = f"{r['model']['timing_days']:.1f}" if r.get('windows') else '—'
        return f"| {label} | {s['windows']} | " + ' | '.join(cells) + f' | {ratio} | {timing} |'

    for site, s in report['stations'].items():
        lines.append(row(names.get(site, site), s))
    lines.append(row('**All stations**', report['all']))
    gate = report.get('gate', {})
    lines += ['', f"**Quality gate: {'PASSED' if gate.get('passed') else 'FAILED'}**", '']
    lines += [f"- {'✅' if c['passed'] else '❌'} {c['check']}: {c['detail']}" for c in gate.get('checks', [])]
    return '\n'.join(lines) + '\n'


def write_report(report: dict, base_path: str, names: Optional[Dict[str, str]] = None) -> List[str]:
    paths = [os.path.join(base_path, 'quality_report.json'), os.path.join(base_path, 'quality_report.md')]
    with open(paths[0], 'w') as f:
        json.dump(report, f, indent=2)
    with open(paths[1], 'w') as f:
        f.write(to_markdown(report, names))
    return paths


def main(argv=None):
    """Re-score a trained run directory (model + normalized data) and write quality_report.*."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base-path', required=True, help='training run directory')
    parser.add_argument('--gate', action='store_true',
                        help='only check an existing quality_report.json; exit 1 if it failed')
    args = parser.parse_args(argv)
    if args.gate:
        with open(os.path.join(args.base_path, 'quality_report.json')) as f:
            report = json.load(f)
        print(to_markdown(report))
        return 0 if report['gate']['passed'] else 1

    import ml_inference
    art = ml_inference.load_artifacts(args.base_path, with_model=True)
    data = pd.read_csv(os.path.join(args.base_path, 'openFlowML', 'normalized_data.csv'),
                       parse_dates=['Date'])
    split = windowing.chronological_split(windowing.build_windows(data), 0.15, 0.15,
                                          windowing.DECODER_DAYS)
    x, _ = windowing.stack(split.test)
    pred = art['model'].predict(x, verbose=0, batch_size=512)
    report = build_report(forecasts(split.test, pred, art['scalers']), art.get('version'))
    for p in write_report(report, args.base_path):
        print('wrote', p)
    print(to_markdown(report))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
