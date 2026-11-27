"""
Phase 4: Daily Mobile Feature Feed producer & verifier.

Generates compact, schema-versioned, SHA-256-checksummed model input payloads
(`inputs/<slug>.json` + `catalog.json`) using `ml_inference.prepare_inputs()`,
so iOS CoreML and Android TFLite clients get 100% feature parity with server-
side Python inference (upstream gauges, basin zones, energy, soil moisture,
elevation SWE, ensemble spread, RFC/NWM prior, and optional static attributes)
with ZERO API credentials or multi-source data-fetching code on device.

Usage:
    python openFlowML/feature_feed.py --model-dir model_release --out-dir forecast_out/feed
    python openFlowML/feature_feed.py --model-dir model_release --out-dir forecast_out/feed \\
        --sites taos-box,royal-gorge,conejos,dolores,westwater,clear-creek
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ml_inference
from data import get_upstream

logger = logging.getLogger(__name__)

FEED_SCHEMA_VERSION = 1
DEFAULT_DECIMALS = 5
FLOAT_INPUT_KEYS = ('encoder_input', 'decoder_input', 'persistence_input', 'static_input')
INT_INPUT_KEYS = ('station_input', 'basin_input')
REQUIRED_INPUT_KEYS = ('encoder_input', 'decoder_input', 'persistence_input',
                       'station_input', 'basin_input')


def _canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')


def schema_digest(training_config: Optional[dict]) -> str:
    """
    Deterministic SHA-256 fingerprint of the model input/output schema.
    Mobile apps compare `feed.schema_sha256` against their downloaded model's
    `training_config` / `manifest.schema` before invoking CoreML or TFLite.
    """
    cfg = training_config or {}
    canonical = {
        'encoder_days': cfg.get('encoder_days'),
        'decoder_days': cfg.get('decoder_days'),
        'encoder_features': cfg.get('encoder_features'),
        'decoder_features': cfg.get('decoder_features'),
        'target_features': cfg.get('target_features'),
        'static_features': cfg.get('static_features'),
        'flow_units': cfg.get('flow_units', 'cfs'),
        'outputs': cfg.get('outputs'),
        'quantiles': cfg.get('quantiles'),
    }
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def tensor_payload_digest(inputs_json: dict) -> str:
    """SHA-256 digest of the serialized tensor dictionary."""
    return hashlib.sha256(_canonical_json_bytes(inputs_json)).hexdigest()


def _site_slug(site_id: str, registry: Optional[dict] = None) -> str:
    entry = (registry or {}).get(site_id)
    if isinstance(entry, dict) and entry.get('aliases'):
        slug = str(entry['aliases'][0]).strip().lower()
    else:
        slug = site_id.lower()
    return re.sub(r'[^a-z0-9_-]+', '_', slug).strip('_') or 'station'


def _serialize_inputs(inputs: Dict[str, np.ndarray], decimals: int = DEFAULT_DECIMALS) -> tuple:
    """Convert numpy tensor dict into compact rounded JSON lists + shape map."""
    serialized = {}
    shapes = {}
    for key in (*FLOAT_INPUT_KEYS, *INT_INPUT_KEYS):
        if key not in inputs:
            continue
        arr = np.asarray(inputs[key])
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in input tensor {key!r}")
        shapes[key] = [int(d) for d in arr.shape]
        if key in INT_INPUT_KEYS:
            serialized[key] = arr.astype('int32').tolist()
        else:
            serialized[key] = np.round(arr.astype('float64'), decimals).tolist()
    return serialized, shapes


def _merged_registry(art: dict) -> dict:
    """Combine repo default registry with the model's registry (model wins on overlap)."""
    merged = dict(get_upstream.load_registry() or {})
    if isinstance(art.get('registry'), dict):
        merged.update(art['registry'])
    return merged


def build_station_feed(site_id: str, art: dict, *,
                       today=None,
                       tz: Optional[str] = None,
                       prepared: Optional[dict] = None,
                       decimals: int = DEFAULT_DECIMALS) -> dict:
    """
    Build a schema-versioned, checksummed Daily Feature Feed payload for `site_id`.
    If `prepared` (from `ml_inference.prepare_inputs`) is already available, reuse
    it directly to avoid duplicate upstream API calls.
    """
    if prepared is None:
        prepared = ml_inference.prepare_inputs(site_id, art, today=today, tz=tz)

    tc = art.get('training_config') or {}
    reg = _merged_registry(art)
    reg_entry = reg.get(site_id) if isinstance(reg, dict) else None
    info = dict(prepared.get('info') or {})
    serialized_inputs, shapes = _serialize_inputs(prepared['inputs'], decimals=decimals)
    forecast_dates = [str(pd.Timestamp(d).date()) for d in prepared['dates']]
    anchor_date = str(pd.Timestamp(prepared['anchor']).date())
    slug = _site_slug(site_id, reg)

    payload = {
        'feed_schema_version': FEED_SCHEMA_VERSION,
        'site_id': site_id,
        'slug': slug,
        'name': ((reg_entry or {}).get('short_name')
                 or (reg_entry or {}).get('name') or site_id),
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model_version': art.get('version'),
        'schema_sha256': schema_digest(tc),
        'anchor_date': anchor_date,
        'forecast_dates': forecast_dates,
        'flow_units': info.get('flow_units', tc.get('flow_units', 'cfs')),
        'area_km2': info.get('area_km2'),
        'info': info,
        'shapes': shapes,
        'inputs': serialized_inputs,
        'inputs_sha256': tensor_payload_digest(serialized_inputs),
    }
    return payload


def verify_feed_payload(payload: dict, training_config: Optional[dict] = None) -> dict:
    """
    Validate a feature feed payload's version, integrity checksum, shapes, and
    finiteness. When `training_config` is provided, also verifies exact
    `schema_sha256` and tensor shape compatibility. Raises ValueError on error.
    """
    if not isinstance(payload, dict):
        raise ValueError("Feature feed payload must be a dict")
    if payload.get('feed_schema_version') != FEED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feed_schema_version {payload.get('feed_schema_version')!r} "
            f"(expected {FEED_SCHEMA_VERSION})")
    inputs = payload.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError("Feature feed missing 'inputs' dict")
    missing = [k for k in REQUIRED_INPUT_KEYS if k not in inputs]
    if missing:
        raise ValueError(f"Feature feed missing required input tensors: {missing}")

    expected_sha = payload.get('inputs_sha256')
    actual_sha = tensor_payload_digest(inputs)
    if expected_sha != actual_sha:
        raise ValueError(
            f"Feature feed inputs_sha256 mismatch: expected {expected_sha}, got {actual_sha}")

    shapes = payload.get('shapes') or {}
    for key, raw_val in inputs.items():
        dtype = 'int32' if key in INT_INPUT_KEYS else 'float32'
        arr = np.asarray(raw_val, dtype=dtype)
        if not np.isfinite(arr).all():
            raise ValueError(f"Feature feed tensor {key!r} contains non-finite values")
        if key in shapes and list(arr.shape) != list(shapes[key]):
            raise ValueError(
                f"Feature feed tensor {key!r} shape {list(arr.shape)} != declared {shapes[key]}")

    if training_config is not None:
        want_schema = schema_digest(training_config)
        got_schema = payload.get('schema_sha256')
        if got_schema != want_schema:
            raise ValueError(
                f"Feature feed schema_sha256 {got_schema} does not match model schema {want_schema}")
        enc_shape = [1, int(training_config['encoder_days']), len(training_config['encoder_features'])]
        dec_shape = [1, int(training_config['decoder_days']), len(training_config['decoder_features'])]
        pers_shape = [1, len(training_config['target_features'])]
        if list(np.asarray(inputs['encoder_input']).shape) != enc_shape:
            raise ValueError(f"encoder_input shape mismatch vs training_config {enc_shape}")
        if list(np.asarray(inputs['decoder_input']).shape) != dec_shape:
            raise ValueError(f"decoder_input shape mismatch vs training_config {dec_shape}")
        if list(np.asarray(inputs['persistence_input']).shape) != pers_shape:
            raise ValueError(f"persistence_input shape mismatch vs training_config {pers_shape}")
        static_feats = training_config.get('static_features') or []
        if static_feats:
            if 'static_input' not in inputs:
                raise ValueError("Model expects static_input but feature feed omits it")
            if list(np.asarray(inputs['static_input']).shape) != [1, len(static_feats)]:
                raise ValueError("static_input shape mismatch vs training_config")
    return payload


def feed_to_prepared(payload: dict) -> dict:
    """Convert a verified feature feed JSON payload into `ml_inference.predict`'s prepared dict."""
    inputs_raw = payload['inputs']
    inputs: Dict[str, np.ndarray] = {}
    for key, val in inputs_raw.items():
        dtype = 'int32' if key in INT_INPUT_KEYS else 'float32'
        inputs[key] = np.asarray(val, dtype=dtype)
    info = dict(payload.get('info') or {})
    if 'flow_units' not in info and payload.get('flow_units'):
        info['flow_units'] = payload['flow_units']
    if 'area_km2' not in info and payload.get('area_km2') is not None:
        info['area_km2'] = payload['area_km2']
    return {
        'inputs': inputs,
        'anchor': pd.Timestamp(payload['anchor_date']),
        'dates': pd.DatetimeIndex(pd.to_datetime(payload['forecast_dates']), name='Date'),
        'info': info,
    }


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run_feed(sites: Sequence[str], model_dir: str, out_dir: str, *,
             today=None, art: Optional[dict] = None,
             decimals: int = DEFAULT_DECIMALS) -> dict:
    """
    Build Daily Feature Feed JSONs for `sites` and write:
      - `<out_dir>/inputs/<slug>.json` per station
      - `<out_dir>/catalog.json` indexing all generated payloads & checksums
    Individual station errors are logged and recorded under `errors` without
    aborting healthy stations.
    """
    art = art or ml_inference.load_artifacts(model_dir, with_model=False)
    reg = _merged_registry(art)
    inputs_dir = os.path.join(out_dir, 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)

    stations_meta: List[dict] = []
    errors: Dict[str, str] = {}
    for raw_site in sites:
        token = str(raw_site).strip()
        if not token:
            continue
        try:
            site_id, _ = get_upstream.resolve_site(token, reg)
        except Exception:
            site_id = token
        try:
            payload = build_station_feed(site_id, art, today=today, decimals=decimals)
            verify_feed_payload(payload, training_config=art.get('training_config'))
            rel_path = f"inputs/{payload['slug']}.json"
            full_path = os.path.join(out_dir, rel_path)
            with open(full_path, 'w') as f:
                json.dump(payload, f, separators=(',', ':'))
            size_bytes = os.path.getsize(full_path)
            stations_meta.append({
                'site_id': payload['site_id'],
                'slug': payload['slug'],
                'name': payload['name'],
                'anchor_date': payload['anchor_date'],
                'forecast_start': payload['forecast_dates'][0] if payload['forecast_dates'] else None,
                'forecast_end': payload['forecast_dates'][-1] if payload['forecast_dates'] else None,
                'trained_station': bool(payload['info'].get('trained_station')),
                'flow_units': payload['flow_units'],
                'path': rel_path,
                'bytes': size_bytes,
                'sha256': _file_sha256(full_path),
                'inputs_sha256': payload['inputs_sha256'],
            })
            logger.info("Wrote feature feed %s -> %s (%d bytes, anchor=%s)",
                        site_id, rel_path, size_bytes, payload['anchor_date'])
        except Exception as e:
            logger.error("Feature feed failed for %s: %s", site_id, e)
            errors[site_id] = str(e)

    catalog = {
        'feed_schema_version': FEED_SCHEMA_VERSION,
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model_version': art.get('version'),
        'schema_sha256': schema_digest(art.get('training_config')),
        'flow_units': (art.get('training_config') or {}).get('flow_units', 'cfs'),
        'stations': stations_meta,
        'errors': errors,
    }
    catalog_path = os.path.join(out_dir, 'catalog.json')
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    logger.info("Wrote feature feed catalog -> %s (%d station(s), %d error(s))",
                catalog_path, len(stations_meta), len(errors))
    return catalog


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model-dir', required=True,
                        help='Directory with training_config.json, scalers.json, station_index.json, basin_index.json')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for catalog.json and inputs/<slug>.json')
    parser.add_argument('--sites', default=None,
                        help='Comma-separated stations (alias, site_id, or NWS LID). '
                             'Defaults to all stations in upstream_gauges.json.')
    parser.add_argument('--today', default=None, help='Override local anchor date (YYYY-MM-DD)')
    parser.add_argument('--decimals', type=int, default=DEFAULT_DECIMALS,
                        help='Decimal rounding precision for float tensors (default: 5)')
    args = parser.parse_args(argv)

    art = ml_inference.load_artifacts(args.model_dir, with_model=False)
    if args.sites:
        sites = [s.strip() for s in args.sites.split(',') if s.strip()]
    else:
        sites = sorted(k for k in (art.get('registry') or {}) if not k.startswith('_'))
    catalog = run_feed(sites, args.model_dir, args.out_dir,
                       today=args.today, art=art, decimals=args.decimals)
    if not catalog['stations']:
        logger.error("Feature feed produced zero station payloads")
        return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    raise SystemExit(main())
