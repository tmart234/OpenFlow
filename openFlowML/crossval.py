"""
Leave-years-out cross-validation of the training recipe.

The default held-out test split (train.py) is the newest ~2 years: about two
snowmelt seasons, which cannot tell model changes apart once they differ by
less than ~1% error. This scores the same recipe on every water year
instead: the years are cut into --folds contiguous blocks
(windowing.water_year_folds), each block is held out once, the networks are
trained on the rest exactly as train.py trains its members (same weights,
early stopping on the validation year, band calibration), and all held-out
forecasts are pooled into one quality report (evaluate.py format) plus a
per-fold summary.

  python openFlowML/crossval.py --base-path <dir>            # writes cv_report.{json,md}
  python openFlowML/crossval.py --compare a.json b.json      # A vs B, pooled and by fold

<dir> holds normalized_data.csv under openFlowML/ and scalers.json /
station_index.json / basin_index.json, i.e. a training run directory or what
combine_data.main() writes. --fetch runs combine_data first. Training
switches (OPENFLOW_BAND, OPENFLOW_DISABLE_*, ...) apply as in train.py, so
an A/B is two runs with different settings, then --compare.
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

import evaluate
import windowing

logger = logging.getLogger(__name__)

DEFAULT_FOLDS = 5
REPORT_NAME = 'cv_report'


def _mean_error(report: dict) -> Optional[float]:
    """Mean over lead days of the median absolute % error of the model."""
    err = report['all'].get('error_pct', {}).get('model')
    return round(float(np.mean(err)), 3) if err else None


def cross_validate(data: pd.DataFrame, scalers: dict, num_stations: int, num_basins: int,
                   folds: int = DEFAULT_FOLDS, members: int = 1) -> dict:
    """Train and score one ensemble of `members` per fold; pooled + per-fold report."""
    import model as model_mod
    import train
    samples = windowing.build_windows(data, windowing.ENCODER_DAYS, windowing.DECODER_DAYS)
    splits = windowing.water_year_folds(samples, folds)
    pooled_samples: List = []
    points, bands = [], []
    per_fold = []
    for k, split in enumerate(splits):
        test_years = sorted({windowing.water_year(s.anchor_date + pd.Timedelta(days=windowing.ENCODER_DAYS - 1))
                             for s in split.test})
        logger.info("Fold %d/%d: test water years %s; train %d, val %d, test %d windows",
                    k + 1, folds, test_years, len(split.train), len(split.val), len(split.test))
        tr_x, tr_y = windowing.stack(split.train)
        va_x, va_y = windowing.stack(split.val)
        te_x, _ = windowing.stack(split.test)
        nets, _ = train.fit_members(
            num_stations, num_basins,
            (tr_x, tr_y, train.sample_weights(tr_x, tr_y)),
            (va_x, va_y, train.sample_weights(va_x, va_y)),
            size=members, seed=train.SEED + 100 * k)
        net = model_mod.average_ensemble(nets)
        if model_mod.has_band(net):
            model_mod.calibrate_band(net, va_x, va_y, train.QUANTILES)
        pred = net.predict(te_x, verbose=0, batch_size=512)
        point, band = model_mod.point_forecast(pred), model_mod.band_forecast(pred)
        fold_report = evaluate.build_report(evaluate.forecasts(
            split.test, point, scalers, band=band, quantiles=train.QUANTILES))
        per_fold.append({'fold': k, 'test_water_years': test_years, 'windows': len(split.test),
                         'mean_error_pct': _mean_error(fold_report),
                         'error_pct': fold_report['all']['error_pct'],
                         'band': fold_report['all'].get('band')})
        logger.info("Fold %d: mean error %.2f%% (persistence %.2f%%)", k + 1,
                    per_fold[-1]['mean_error_pct'], np.mean(fold_report['all']['error_pct']['persistence']))
        pooled_samples += split.test
        points.append(point)
        if band is not None:
            bands.append(band)
    pooled = evaluate.build_report(evaluate.forecasts(
        pooled_samples, np.concatenate(points), scalers,
        band=np.concatenate(bands) if bands else None, quantiles=train.QUANTILES))
    return {'folds': per_fold, 'pooled': pooled, 'mean_error_pct': _mean_error(pooled),
            'n_folds': folds, 'members_per_fold': members, 'windows': len(pooled_samples),
            'settings': {k: v for k, v in sorted(os.environ.items()) if k.startswith('OPENFLOW_')}}


def to_markdown(report: dict, names: Optional[dict] = None) -> str:
    lines = [f"# Cross-validation: {report['n_folds']} leave-water-years-out folds",
             '',
             f"Every water year held out once; {report['windows']} forecasts pooled. "
             f"Mean over lead days of the median absolute % error of the daily maximum: "
             f"**{report['mean_error_pct']:.2f}%**.",
             '',
             '| Fold | Test water years | Windows | Mean error % |',
             '|---|---|---|---|']
    for f in report['folds']:
        years = f['test_water_years']
        lines.append(f"| {f['fold'] + 1} | {years[0]}–{years[-1]} | {f['windows']} | {f['mean_error_pct']:.2f} |")
    if report.get('settings'):
        lines += ['', 'Settings: ' + ', '.join(f'`{k}={v}`' for k, v in report['settings'].items())]
    return '\n'.join(lines) + '\n\n' + evaluate.to_markdown(report['pooled'], names)


def compare(path_a: str, path_b: str) -> str:
    """Markdown: B minus A, pooled by lead day and fold by fold (negative = B better)."""
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    ea, eb = a['pooled']['all']['error_pct']['model'], b['pooled']['all']['error_pct']['model']
    leads = [d for d in evaluate.REPORT_LEADS if d <= len(ea)]
    lines = [f'# Cross-validation: B vs A', '', f'A: `{path_a}`  ', f'B: `{path_b}`', '',
             '| | ' + ' | '.join(f'Day {d}' for d in leads) + ' | Mean |', '|---|' + '---|' * (len(leads) + 1)]
    for label, e, m in (('A', ea, a['mean_error_pct']), ('B', eb, b['mean_error_pct'])):
        lines.append(f'| {label} | ' + ' | '.join(f'{e[d - 1]:.2f}' for d in leads) + f' | {m:.2f} |')
    lines.append('| B − A | ' + ' | '.join(f'{eb[d - 1] - ea[d - 1]:+.2f}' for d in leads)
                 + f" | {b['mean_error_pct'] - a['mean_error_pct']:+.2f} |")
    lines += ['', '| Fold | Test water years | A | B | B − A |', '|---|---|---|---|---|']
    better = 0
    for fa, fb in zip(a['folds'], b['folds']):
        d = fb['mean_error_pct'] - fa['mean_error_pct']
        better += d < 0
        years = fa['test_water_years']
        lines.append(f"| {fa['fold'] + 1} | {years[0]}–{years[-1]} | {fa['mean_error_pct']:.2f} | "
                     f"{fb['mean_error_pct']:.2f} | {d:+.2f} |")
    lines += ['', f"B is better in {better} of {len(a['folds'])} folds (mean error, lower is better)."]
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base-path', help='directory with openFlowML/normalized_data.csv and the index JSONs')
    parser.add_argument('--folds', type=int, default=DEFAULT_FOLDS)
    parser.add_argument('--members', type=int, default=1, help='networks per fold (train.py ships 3)')
    parser.add_argument('--fetch', action='store_true', help='run combine_data first (needs network)')
    parser.add_argument('--compare', nargs=2, metavar=('A', 'B'), help='compare two cv_report.json files')
    args = parser.parse_args(argv)
    if args.compare:
        print(compare(*args.compare))
        return 0
    if not args.base_path:
        parser.error('--base-path is required unless --compare is given')
    if args.fetch:
        import combine_data
        os.environ.setdefault('GITHUB_WORKSPACE', args.base_path)
        combine_data.main()
    with open(os.path.join(args.base_path, 'scalers.json')) as f:
        scalers = json.load(f)
    with open(os.path.join(args.base_path, 'station_index.json')) as f:
        num_stations = len(json.load(f))
    with open(os.path.join(args.base_path, 'basin_index.json')) as f:
        num_basins = len(json.load(f))
    data = pd.read_csv(os.path.join(args.base_path, 'openFlowML', 'normalized_data.csv'), parse_dates=['Date'])
    report = cross_validate(data, scalers, num_stations, num_basins, args.folds, args.members)
    names = {}
    try:
        from data import get_upstream
        names = {sid: cfg.get('short_name') or cfg.get('name', sid)
                 for sid, cfg in get_upstream.load_registry().items()}
    except Exception:
        pass
    with open(os.path.join(args.base_path, f'{REPORT_NAME}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    md = to_markdown(report, names)
    with open(os.path.join(args.base_path, f'{REPORT_NAME}.md'), 'w') as f:
        f.write(md)
    print(md)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
