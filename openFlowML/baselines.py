import argparse
import logging
import os

import numpy as np
import pandas as pd

"""
Naive forecast baselines: the bar the LSTM has to clear to be worth anything.

  - persistence: predict flow[t+k] = flow[t]. Error grows with horizon k.
  - climatology: predict flow[t+k] = the per-station day-of-year mean flow.

Reported as MAE in real cfs, computed per station then pooled. Phase 3 will
compare the trained model against these numbers.
"""

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

FLOW_COLUMNS = ['Min Flow', 'Max Flow']
DEFAULT_HORIZON = 14


def persistence_baseline(df, horizon=DEFAULT_HORIZON):
    """
    MAE of predicting flow[t+k] = flow[t], for k in 1..horizon, computed per
    station and pooled. Returns {column: {'overall': float, 'per_horizon': [float, ...]}}.
    """
    result = {col: {'per_horizon': [], 'overall': float('nan')} for col in FLOW_COLUMNS}
    for col in FLOW_COLUMNS:
        per_horizon_errors = [[] for _ in range(horizon)]
        for _, station in df.groupby('site_id'):
            station = station.sort_values('Date')
            series = station[col].to_numpy(dtype=float)
            for k in range(1, horizon + 1):
                if len(series) > k:
                    per_horizon_errors[k - 1].append(np.abs(series[k:] - series[:-k]))
        per_horizon = [float(np.concatenate(e).mean()) if e else float('nan')
                       for e in per_horizon_errors]
        all_errors = [arr for e in per_horizon_errors for arr in e]
        overall = float(np.concatenate(all_errors).mean()) if all_errors else float('nan')
        result[col] = {'per_horizon': per_horizon, 'overall': overall}
    return result


def climatology_baseline(df):
    """
    MAE of predicting flow[d] = the per-station mean flow for that day-of-year.
    Horizon-independent (the climatological prediction for any date is just the
    DOY mean), so reported as a single MAE per flow column.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['doy'] = df['Date'].dt.dayofyear
    result = {}
    for col in FLOW_COLUMNS:
        errors = []
        for _, station in df.groupby('site_id'):
            doy_mean = station.groupby('doy')[col].transform('mean').to_numpy(float)
            actual = station[col].to_numpy(float)
            errors.append(np.abs(actual - doy_mean))
        result[col] = float(np.concatenate(errors).mean()) if errors else float('nan')
    return result


def evaluate_baselines(df, horizon=DEFAULT_HORIZON):
    """Compute both baselines and return a structured report dict."""
    return {
        'persistence': persistence_baseline(df, horizon=horizon),
        'climatology': climatology_baseline(df),
        'horizon': horizon,
        'sites': sorted(df['site_id'].unique().tolist()) if 'site_id' in df.columns else [],
    }


def format_report(report):
    """Render a baseline report as a short human-readable summary."""
    lines = [
        f"Baselines (horizon = {report['horizon']} days, sites = {len(report['sites'])})",
        "  persistence MAE (cfs):",
    ]
    for col, stats in report['persistence'].items():
        lines.append(f"    {col:9s} overall={stats['overall']:.1f}   "
                     f"day1={stats['per_horizon'][0]:.1f}   "
                     f"day14={stats['per_horizon'][-1]:.1f}")
    lines.append("  climatology MAE (cfs):")
    for col, mae in report['climatology'].items():
        lines.append(f"    {col:9s} {mae:.1f}")
    return "\n".join(lines)


def main(csv_path=None, horizon=DEFAULT_HORIZON):
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'combined_data_all_sites.csv')
    df = pd.read_csv(csv_path)
    report = evaluate_baselines(df, horizon=horizon)
    print(format_report(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute persistence + climatology baselines.')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to combined_data_all_sites.csv (default: alongside this script)')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON, help='Forecast horizon in days')
    args = parser.parse_args()
    main(args.csv, args.horizon)
