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


def _concat_forecasts(parts: List[dict]) -> dict:
    """Concatenate per-fold evaluate.forecasts() dicts (already in real cfs units)."""
    out = {}
    for k in parts[0]:
        if k == 'quantiles':
            out[k] = parts[0][k]
        else:
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def cross_validate(data: pd.DataFrame, scalers: dict, num_stations: int, num_basins: int,
                   folds: int = DEFAULT_FOLDS, members: int = 1,
                   raw_data: Optional[pd.DataFrame] = None,
                   mode: str = 'water-year') -> dict:
    """
    Train and score one ensemble of `members` per fold; pooled + per-fold report.

    `mode='water-year'` (default): leave-water-years-out CV across all stations.
    `mode='leave-basin-out'`: holds out entire HUC8 basins / stations with
    `station_idx=0` at test time to measure zero-shot transfer accuracy.
    """
    import tempfile
    import combine_data
    import model as model_mod
    import normalize_data
    import train
    from data import build_station_attributes as bsa

    if mode not in ('water-year', 'leave-basin-out'):
        raise ValueError(f"Unsupported CV mode {mode!r}; expected 'water-year' or 'leave-basin-out'")

    station_attrs = bsa.load_station_attributes()
    use_static = bool(train.USE_STATIC_ATTRIBUTES and any(
        isinstance(station_attrs.get(s), dict) for s in data['site_id'].unique()))
    static_feature_names = list(bsa.STATIC_FEATURE_NAMES) if use_static else None

    station_areas_km2 = None
    if train.SPECIFIC_DISCHARGE:
        all_sites = sorted(data['site_id'].dropna().unique())
        areas = bsa.station_areas_map(all_sites, station_attrs)
        if len(areas) == len(all_sites) and areas:
            station_areas_km2 = areas
            logger.info("CV using specific discharge (mm/day) across %d sites", len(areas))
        else:
            logger.warning("OPENFLOW_SPECIFIC_DISCHARGE=1 requested in CV but area_km2 missing; using cfs")

    forecast_table = None
    if train.TRAIN_REAL_FORECASTS:
        fc_dir = os.getenv('OPENFLOW_REFORECAST_DIR', '').strip() or combine_data._resolve_cache_dir()
        if fc_dir:
            try:
                from data import get_gefs_reforecast
                ft = get_gefs_reforecast.load_reforecast_table(fc_dir)
                if ft is not None and not ft.empty:
                    forecast_table = ft
            except Exception as e:
                logger.warning("Reforecast table skipped in CV: %s", e)

    all_sites = sorted(data['site_id'].dropna().unique())
    base_static = None
    if use_static:
        base_sc = bsa.fit_static_scalers(station_attrs, all_sites, static_feature_names)
        base_static = bsa.build_static_vectors(all_sites, station_attrs, base_sc, static_feature_names)

    base_samples = windowing.build_windows(
        data, windowing.ENCODER_DAYS, windowing.DECODER_DAYS,
        scalers=scalers, forecast_table=forecast_table, static_vectors=base_static)
    if mode == 'leave-basin-out':
        base_splits = windowing.leave_basin_out_folds(base_samples, n_folds=folds)
    else:
        base_splits = windowing.water_year_folds(base_samples, folds)

    raw_wy = (pd.to_datetime(raw_data['Date']).map(windowing.water_year)
              if raw_data is not None and not raw_data.empty else None)
    fold_fcs: List[dict] = []
    per_fold = []
    total_windows = 0
    n_static = len(static_feature_names) if static_feature_names else 0

    for k, split in enumerate(base_splits):
        fold_scalers = scalers
        fold_split = split
        test_sites = sorted({s.site_id for s in split.test})
        test_years = sorted({windowing.water_year(s.anchor_date + pd.Timedelta(days=windowing.ENCODER_DAYS - 1))
                             for s in split.test})
        val_years = sorted({windowing.water_year(s.anchor_date + pd.Timedelta(days=windowing.ENCODER_DAYS - 1))
                            for s in split.val})

        fold_static = base_static
        if use_static:
            fit_sites = [s for s in all_sites if s not in set(test_sites)] if mode == 'leave-basin-out' else all_sites
            fold_st_sc = bsa.fit_static_scalers(station_attrs, fit_sites, static_feature_names)
            fold_static = bsa.build_static_vectors(all_sites, station_attrs, fold_st_sc, static_feature_names)

        if raw_data is not None and not raw_data.empty:
            if mode == 'leave-basin-out':
                fit_mask = (~raw_data['site_id'].isin(set(test_sites))).to_numpy()
            else:
                fit_mask = (~raw_wy.isin(set(test_years) | set(val_years))).to_numpy()
            with tempfile.TemporaryDirectory() as td:
                fold_df = normalize_data.normalize_data(
                    raw_data, artifacts_dir=td, fit_mask=fit_mask,
                    station_areas_km2=station_areas_km2)
                with open(os.path.join(td, 'scalers.json')) as f:
                    fold_scalers = json.load(f)
            fold_samples = windowing.build_windows(
                fold_df, windowing.ENCODER_DAYS, windowing.DECODER_DAYS,
                scalers=fold_scalers, forecast_table=forecast_table,
                static_vectors=fold_static)
            fold_split = (windowing.leave_basin_out_folds(fold_samples, n_folds=folds)[k]
                          if mode == 'leave-basin-out'
                          else windowing.water_year_folds(fold_samples, folds)[k])
        elif mode == 'leave-basin-out' and use_static and fold_static is not None:
            fold_samples = windowing.build_windows(
                data, windowing.ENCODER_DAYS, windowing.DECODER_DAYS,
                scalers=fold_scalers, forecast_table=forecast_table,
                static_vectors=fold_static)
            fold_split = windowing.leave_basin_out_folds(fold_samples, n_folds=folds)[k]

        if mode == 'leave-basin-out':
            logger.info("Fold %d/%d (LBO): test sites %s; train %d, val %d, test %d windows",
                        k + 1, len(base_splits), test_sites,
                        len(fold_split.train), len(fold_split.val), len(fold_split.test))
        else:
            logger.info("Fold %d/%d: test water years %s; train %d, val %d, test %d windows",
                        k + 1, len(base_splits), test_years,
                        len(fold_split.train), len(fold_split.val), len(fold_split.test))

        tr_x, tr_y = windowing.stack(fold_split.train)
        va_x, va_y = windowing.stack(fold_split.val)
        te_x, _ = windowing.stack(fold_split.test)
        nets, _ = train.fit_members(
            num_stations, num_basins,
            (tr_x, tr_y, train.sample_weights(tr_x, tr_y)),
            (va_x, va_y, train.sample_weights(va_x, va_y)),
            size=members, seed=train.SEED + 100 * k,
            static_features=n_static)
        net = model_mod.average_ensemble(nets)
        if model_mod.has_band(net):
            model_mod.calibrate_band(net, va_x, va_y, train.QUANTILES)
        pred = net.predict(te_x, verbose=0, batch_size=512)
        point, band = model_mod.point_forecast(pred), model_mod.band_forecast(pred)
        fc = evaluate.forecasts(
            fold_split.test, point, fold_scalers, band=band, quantiles=train.QUANTILES,
            station_areas_km2=station_areas_km2)
        fold_report = evaluate.build_report(fc)
        entry = {
            'fold': k,
            'windows': len(fold_split.test),
            'mean_error_pct': _mean_error(fold_report),
            'error_pct': fold_report['all']['error_pct'],
            'band': fold_report['all'].get('band'),
        }
        if mode == 'leave-basin-out':
            entry['test_sites'] = test_sites
        else:
            entry['test_water_years'] = test_years
        per_fold.append(entry)
        logger.info("Fold %d: mean error %.2f%% (persistence %.2f%%)", k + 1,
                    per_fold[-1]['mean_error_pct'], np.mean(fold_report['all']['error_pct']['persistence']))
        fold_fcs.append(fc)
        total_windows += len(fold_split.test)
    pooled = evaluate.build_report(_concat_forecasts(fold_fcs))
    return {'mode': mode, 'folds': per_fold, 'pooled': pooled, 'mean_error_pct': _mean_error(pooled),
            'n_folds': len(base_splits), 'members_per_fold': members, 'windows': total_windows,
            'flow_units': 'mm_day' if station_areas_km2 else 'cfs',
            'static_attributes': use_static,
            'settings': {k: v for k, v in sorted(os.environ.items()) if k.startswith('OPENFLOW_')}}


def _fold_label(f: dict) -> str:
    if 'test_sites' in f:
        return ', '.join(f['test_sites'])
    years = f.get('test_water_years') or []
    return f"{years[0]}–{years[-1]}" if years else '—'


def to_markdown(report: dict, names: Optional[dict] = None) -> str:
    mode = report.get('mode', 'water-year')
    if mode == 'leave-basin-out':
        title = f"# Cross-validation: {report['n_folds']} leave-basin-out folds (zero-shot transfer)"
        subtitle = (f"Every river/basin held out unseen (`station_idx=0`); {report['windows']} forecasts pooled. "
                    f"Mean over lead days of the median absolute % error of the daily maximum: "
                    f"**{report['mean_error_pct']:.2f}%**.")
        col_hdr = 'Held-out sites'
    else:
        title = f"# Cross-validation: {report['n_folds']} leave-water-years-out folds"
        subtitle = (f"Every water year held out once; {report['windows']} forecasts pooled. "
                    f"Mean over lead days of the median absolute % error of the daily maximum: "
                    f"**{report['mean_error_pct']:.2f}%**.")
        col_hdr = 'Test water years'
    lines = [title, '', subtitle, '',
             f'| Fold | {col_hdr} | Windows | Mean error % |',
             '|---|---|---|---|']
    for f in report['folds']:
        lines.append(f"| {f['fold'] + 1} | {_fold_label(f)} | {f['windows']} | {f['mean_error_pct']:.2f} |")
    if report.get('settings'):
        lines += ['', 'Settings: ' + ', '.join(f'`{k}={v}`' for k, v in report['settings'].items())]
    return '\n'.join(lines) + '\n\n' + evaluate.to_markdown(report['pooled'], names)


def compare(path_a: str, path_b: str) -> str:
    """Markdown: B minus A, pooled by lead day and fold by fold (negative = B better)."""
    a, b = (json.load(open(p)) for p in (path_a, path_b))
    ea, eb = a['pooled']['all']['error_pct']['model'], b['pooled']['all']['error_pct']['model']
    leads = [d for d in evaluate.REPORT_LEADS if d <= len(ea)]
    col_hdr = 'Held-out sites' if a.get('mode') == 'leave-basin-out' else 'Test water years'
    lines = ['# Cross-validation: B vs A', '', f'A: `{path_a}`  ', f'B: `{path_b}`', '',
             '| | ' + ' | '.join(f'Day {d}' for d in leads) + ' | Mean |', '|---|' + '---|' * (len(leads) + 1)]
    for label, e, m in (('A', ea, a['mean_error_pct']), ('B', eb, b['mean_error_pct'])):
        lines.append(f'| {label} | ' + ' | '.join(f'{e[d - 1]:.2f}' for d in leads) + f' | {m:.2f} |')
    lines.append('| B − A | ' + ' | '.join(f'{eb[d - 1] - ea[d - 1]:+.2f}' for d in leads)
                 + f" | {b['mean_error_pct'] - a['mean_error_pct']:+.2f} |")
    lines += ['', f'| Fold | {col_hdr} | A | B | B − A |', '|---|---|---|---|---|']
    better = 0
    for fa, fb in zip(a['folds'], b['folds']):
        d = fb['mean_error_pct'] - fa['mean_error_pct']
        better += d < 0
        lines.append(f"| {fa['fold'] + 1} | {_fold_label(fa)} | {fa['mean_error_pct']:.2f} | "
                     f"{fb['mean_error_pct']:.2f} | {d:+.2f} |")
    lines += ['', f"B is better in {better} of {len(a['folds'])} folds (mean error, lower is better)."]
    return '\n'.join(lines) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base-path', help='directory with openFlowML/normalized_data.csv and the index JSONs')
    parser.add_argument('--mode', choices=('water-year', 'leave-basin-out'), default='water-year',
                        help='CV split strategy (water-year or leave-basin-out zero-shot transfer)')
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
    raw_path = os.path.join(args.base_path, 'openFlowML', 'combined_data_all_sites.csv')
    raw_data = pd.read_csv(raw_path, parse_dates=['Date']) if os.path.exists(raw_path) else None
    report = cross_validate(
        data, scalers, num_stations, num_basins, args.folds, args.members,
        raw_data=raw_data, mode=args.mode)
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
