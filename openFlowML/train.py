"""
Phase 3 training driver.

Reads the normalized per-station daily-indexed dataset produced by
combine_data + normalize_data, builds station-aware encoder-decoder windows
with the no-flow-leakage invariant, splits chronologically with an embargo
gap, fits the seq2seq model, and persists everything inference needs
(model + scalers + station_index + basin_index + training_config).

Also logs the persistence baseline MAE on the test set so we can immediately
tell whether the model is doing anything useful (i.e. better than "tomorrow
== today").
"""

import json
import logging
import os
import shutil

import numpy as np
import tensorflow as tf

import combine_data
import normalize_data
import baselines
import windowing
import model as model_mod
from data import basin_zones, get_upstream

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ENCODER_DAYS = windowing.ENCODER_DAYS
DECODER_DAYS = windowing.DECODER_DAYS

EPOCHS = 50               # early stopping decides the actual count
# Sample weight = 1 + EVENT_WEIGHT * mean |target - persistence| (scaled log
# flow), renormalised to mean 1. Most windows are steady, so an unweighted
# loss rewards near-flat forecasts; this puts the rises and recessions --
# the windows a forecast exists for -- back in charge.
EVENT_WEIGHT = 2.0
BATCH_SIZE = 64
PATIENCE = 6              # epochs without val improvement before stopping
LR_PATIENCE = 3           # epochs before ReduceLROnPlateau halves LR
# Per-lead loss weight (1 / persistence MAE on that lead) ** LEAD_WEIGHT_POWER,
# mean 1. Day-1 errors are small next to day-14 errors, so an unweighted loss
# barely cares about tomorrow; 0 switches this off.
LEAD_WEIGHT_POWER = 0.0
# Members of the averaged ensemble (model.average_ensemble), each from its own
# random seed. Averaging removes much of one network's day-to-day noise.
ENSEMBLE_SIZE = int(os.getenv('OPENFLOW_ENSEMBLE_SIZE', '') or 3)
SEED = 20261103
# After the held-out evaluation, retrain every member on ALL windows (train +
# val + test) for the epochs and learning-rate schedule early stopping chose,
# so the published model has also learned from the most recent years (the
# test split is the newest ~2 years, floods included). quality_report.* still
# describes the held-out run of the same recipe. OPENFLOW_FINAL_REFIT=0 ships
# the evaluated ensemble instead.
FINAL_REFIT = os.getenv('OPENFLOW_FINAL_REFIT', '1').strip() not in ('0', 'false', 'False')


def _save_keras_model(model_obj, base_path):
    """
    Save the trained model.

    Keras 2.13's preferred format is still .h5; the .h5 is the canonical
    training output and the source of truth for Phase 6 mobile conversion
    (CoreML + TFLite). See docs/INFERENCE.md.
    """
    h5_path = os.path.join(base_path, 'lstm_model.h5')
    model_obj.save(h5_path)
    return h5_path


def _summarize_per_horizon(y_true, y_pred):
    """Per-horizon (day 1..14) MAE on the scaled target; both arrays (N, H, 2)."""
    err = np.abs(y_true - y_pred)
    return err.mean(axis=(0, 2)).tolist()


def _persistence_pred_for_samples(test_samples):
    """
    Persistence prediction for the test set, in the scaled target space:
    every forecast day's flow equals the last encoder-day flow, broadcast
    across the horizon. windowing.WindowedSample already stores this value
    as `persistence_anchor` so it's a straight lift.
    """
    horizon = test_samples[0].target_Y.shape[0]
    preds = [np.tile(s.persistence_anchor, (horizon, 1)) for s in test_samples]
    return np.stack(preds).astype('float32')


def event_weights(inputs, targets, event_weight: float = EVENT_WEIGHT) -> np.ndarray:
    """Per-window weights emphasising change from persistence (mean 1)."""
    change = np.abs(targets - inputs['persistence_input'][:, None, :]).mean(axis=(1, 2))
    w = 1.0 + event_weight * change
    return (w / w.mean()).astype('float32')


def lead_weights(inputs, targets, power: float = LEAD_WEIGHT_POWER) -> np.ndarray:
    """Per-lead-day weights (1 / persistence MAE) ** power, mean 1; shape (H,)."""
    mae = np.abs(targets - inputs['persistence_input'][:, None, :]).mean(axis=(0, 2))
    w = (1.0 / np.maximum(mae, 1e-6)) ** power
    return (w / w.mean()).astype('float32')


def sample_weights(inputs, targets, lead_power: float = None) -> np.ndarray:
    """Event weights per window, times lead weights per day when enabled: (N,) or (N, H)."""
    w = event_weights(inputs, targets)
    power = LEAD_WEIGHT_POWER if lead_power is None else lead_power
    if not power:
        return w
    return (w[:, None] * lead_weights(inputs, targets, power)[None, :]).astype('float32')


def _build_member(num_stations, num_basins):
    return model_mod.build_encoder_decoder(
        num_stations=num_stations,
        num_basins=num_basins,
        encoder_features=len(windowing.ENCODER_FEATURES),
        decoder_features=len(windowing.DECODER_FEATURES),
        target_features=len(windowing.TARGET_FEATURES),
        encoder_days=ENCODER_DAYS,
        decoder_days=DECODER_DAYS,
    )


def fit_members(num_stations, num_basins, train_data, val_data, size=None, seed=SEED):
    """
    Fit `size` members with early stopping on val. Returns (members, schedules):
    schedules[i] is the per-epoch learning rate member i used up to its best
    epoch, which a refit replays.
    """
    size = ENSEMBLE_SIZE if size is None else size
    members, schedules = [], []
    for i in range(size):
        tf.keras.utils.set_random_seed(seed + i)
        net = _build_member(num_stations, num_basins)
        if i == 0:
            net.summary(print_fn=logger.info)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=PATIENCE,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=LR_PATIENCE,
                min_lr=1e-5, verbose=1),
        ]
        hist = net.fit(train_data[0], train_data[1], sample_weight=train_data[2],
                       validation_data=val_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       callbacks=callbacks, verbose=2)
        best = int(np.argmin(hist.history['val_loss'])) + 1
        lrs = [float(v) for v in hist.history.get('lr', [])][:best] or [1e-3] * best
        logger.info("Member %d/%d: best epoch %d of %d", i + 1, size, best, len(hist.history['val_loss']))
        members.append(net)
        schedules.append(lrs)
    return members, schedules


def refit_members(num_stations, num_basins, all_data, schedules, seed=SEED):
    """Retrain each member from scratch on all windows, replaying its LR schedule."""
    members = []
    for i, lrs in enumerate(schedules):
        tf.keras.utils.set_random_seed(seed + 1000 + i)
        net = _build_member(num_stations, num_basins)
        sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr, lrs=lrs: lrs[min(epoch, len(lrs) - 1)])
        net.fit(all_data[0], all_data[1], sample_weight=all_data[2], epochs=len(lrs),
                batch_size=BATCH_SIZE, callbacks=[sched], verbose=2)
        logger.info("Refit member %d/%d on all windows for %d epochs", i + 1, len(schedules), len(lrs))
        members.append(net)
    return members


def _concat_inputs(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    base_path = combine_data.get_base_path()

    # 1. Pull and normalize the data spine. (Combine_data is responsible for
    #    fetch + daily indexing + gap handling + SWE wiring + HUC8 tagging;
    #    save_combined_data inside combine_data writes the normalized CSV plus
    #    scalers/station_index/basin_index JSON.)
    # OPENFLOW_REUSE_COMBINED=1 re-normalizes the combined CSV a previous run
    # wrote instead of re-fetching every source (model iteration; not for CI).
    cached = os.path.join(base_path, 'openFlowML', 'combined_data_all_sites.csv')
    if os.getenv('OPENFLOW_REUSE_COMBINED', '').strip() in ('1', 'true', 'True') and os.path.exists(cached):
        import pandas as pd
        logger.info("Reusing %s (OPENFLOW_REUSE_COMBINED)", cached)
        data = normalize_data.normalize_data(pd.read_csv(cached), artifacts_dir=base_path)
    else:
        data = combine_data.main()
    if data is None or data.empty:
        raise ValueError("combine_data returned no usable data")

    # 2. Sanity-check spine outputs the rest of training relies on.
    required = {'station_idx', 'basin_idx', 'site_id', 'Date',
                'Min Flow', 'Max Flow', 'TMIN', 'TMAX',
                'precipitation',
                'SWE', 'soil_moisture', 'sm_observed',
                'drought_index',
                'reservoir_storage', 'reservoir_release', 'reservoir_observed',
                'upstream_flow', 'upstream_observed',
                *windowing.BASIN_PRECIP_FEATURES, 'basin_precip_observed',
                'doy_sin', 'doy_cos'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Normalized dataset is missing expected columns: {sorted(missing)}")

    # 3. Build station-aware windows with no flow leakage in the decoder.
    samples = windowing.build_windows(data, ENCODER_DAYS, DECODER_DAYS)
    if not samples:
        raise ValueError("No usable training windows were produced from the dataset")
    logger.info("Built %d total windows across %d sites",
                len(samples), len({s.site_id for s in samples}))

    # 4. Chronological split with an embargo gap == forecast horizon.
    splits = windowing.chronological_split(
        samples, val_frac=0.15, test_frac=0.15, embargo_days=DECODER_DAYS)
    if not splits.train or not splits.val:
        raise ValueError("Chronological split left train or val empty -- not enough history")
    logger.info("Split sizes: train=%d  val=%d  test=%d",
                len(splits.train), len(splits.val), len(splits.test))

    train_inputs, train_targets = windowing.stack(splits.train)
    val_inputs, val_targets = windowing.stack(splits.val)
    test_inputs, test_targets = windowing.stack(splits.test) if splits.test else (None, None)

    # 5. Build the encoder-decoder. Vocabulary sizes come from the persisted
    #    index files so the embedding indices line up with what inference will
    #    look up.
    with open(os.path.join(base_path, 'station_index.json')) as f:
        station_index = json.load(f)
    with open(os.path.join(base_path, 'basin_index.json')) as f:
        basin_index = json.load(f)

    # 6. Fit an averaged ensemble of members (each early-stopped on val).
    train_w = sample_weights(train_inputs, train_targets)
    val_w = sample_weights(val_inputs, val_targets)
    members, schedules = fit_members(
        len(station_index), len(basin_index),
        (train_inputs, train_targets, train_w), (val_inputs, val_targets, val_w))
    net = model_mod.average_ensemble(members)
    logger.info("Ensemble of %d member(s)", len(members))

    # 7. Evaluate against persistence + any external operational baselines
    #    on the held-out test set. External baselines (CBRFC, S2F) return
    #    None when their archive integration isn't wired in yet, in which
    #    case they're silently skipped.
    if test_inputs is not None and len(splits.test) > 0:
        model_pred = net.predict(test_inputs, verbose=0)
        model_mae_per_h = _summarize_per_horizon(test_targets, model_pred)
        persistence_pred = _persistence_pred_for_samples(splits.test)
        persistence_mae_per_h = _summarize_per_horizon(test_targets, persistence_pred)

        external_mae_per_h = {}
        try:
            from data import get_cbrfc
            cbrfc_pred = get_cbrfc.baseline_predictions(splits.test)
            if cbrfc_pred is not None:
                external_mae_per_h['cbrfc'] = _summarize_per_horizon(
                    test_targets, cbrfc_pred)
        except Exception as e:
            logger.warning("CBRFC baseline skipped: %s", e)
        try:
            from data import get_s2f
            s2f_pred = get_s2f.baseline_predictions(splits.test)
            if s2f_pred is not None:
                external_mae_per_h['s2f'] = _summarize_per_horizon(
                    test_targets, s2f_pred)
        except Exception as e:
            logger.warning("S2F baseline skipped: %s", e)

        logger.info("Test MAE (scaled space) per horizon day:")
        for k in range(len(model_mae_per_h)):
            m = model_mae_per_h[k]
            p = persistence_mae_per_h[k]
            verdict = "BEATS" if m < p else "LOSES_TO"
            extras = "".join(
                f"  {name}={vals[k]:.4f}"
                for name, vals in external_mae_per_h.items())
            logger.info("  day %2d: model=%.4f  persistence=%.4f  (%s persistence)%s",
                        k + 1, m, p, verdict, extras)
        for name in external_mae_per_h:
            logger.info("External baseline available: %s", name)

        with open(os.path.join(base_path, 'scalers.json')) as f:
            scalers = json.load(f)
        names = {sid: cfg.get('short_name') or cfg.get('name', sid)
                 for sid, cfg in get_upstream.load_registry().items()}

        # Verification in real units (cfs) per station and lead day, against
        # persistence and the upstream-change reference, plus the quality
        # gate CI checks before publishing (evaluate.py). Written before the
        # charts so a plotting problem cannot lose it.
        try:
            import evaluate
            report = evaluate.build_report(
                evaluate.forecasts(splits.test, model_pred, scalers),
                os.environ.get('OPENFLOW_MODEL_VERSION'))
            evaluate.write_report(report, base_path, names)
            logger.info("Forecast quality (test split):\n%s", evaluate.to_markdown(report, names))
        except Exception as e:
            logger.warning("Quality report skipped: %s", e)

        # Visual QA: one hindcast chart per station (its highest-flow test
        # window), uploaded as a CI artifact. Best-effort -- a plotting
        # problem must never lose a trained model.
        try:
            import hindcast
            hindcast.write_hindcast_charts(
                splits.test, model_pred, scalers, os.path.join(base_path, 'charts'),
                names=names, model_version=os.environ.get('OPENFLOW_MODEL_VERSION'))
        except Exception as e:
            logger.warning("Hindcast charts skipped: %s", e)

    # 7b. Final refit on every window (see FINAL_REFIT). Sanity check: the
    #     refit must fit the (now in-sample) test windows at least as well as
    #     the held-out ensemble did; otherwise keep the evaluated model.
    refit_done = False
    if FINAL_REFIT and test_inputs is not None and len(splits.test) > 0:
        all_inputs = _concat_inputs([train_inputs, val_inputs, test_inputs])
        all_targets = np.concatenate([train_targets, val_targets, test_targets])
        refit = model_mod.average_ensemble(refit_members(
            len(station_index), len(basin_index),
            (all_inputs, all_targets, sample_weights(all_inputs, all_targets)), schedules))
        held_out = float(np.mean(_summarize_per_horizon(test_targets, model_pred)))
        in_sample = float(np.mean(_summarize_per_horizon(
            test_targets, refit.predict(test_inputs, verbose=0))))
        logger.info("Refit check: test MAE held-out %.4f, refit (in-sample) %.4f", held_out, in_sample)
        if in_sample <= held_out:
            net, refit_done = refit, True
        else:
            logger.warning("Refit fits the test windows worse than the held-out model; keeping the evaluated model")

    # 8. Persist the model + training config alongside the scaler/index JSON
    #    that combine_data already wrote. These five artifacts together are
    #    everything an inference pipeline needs.
    h5_path = _save_keras_model(net, base_path)
    # The upstream registry travels with the model: inference needs each
    # station's upstream gauges, lags and weights to rebuild upstream_flow.
    shutil.copyfile(get_upstream.REGISTRY_PATH, os.path.join(base_path, 'upstream_gauges.json'))
    # Same for the basin zones: inference needs each station's zone sample
    # points (and their NWS gridpoints) to build the rain/snow inputs.
    shutil.copyfile(basin_zones.ZONES_PATH, os.path.join(base_path, 'basin_zones.json'))
    known = (data.groupby('site_id')[windowing.UPSTREAM_KNOWN_DAYS_COLUMN].first()
             .astype(int).to_dict()
             if windowing.UPSTREAM_KNOWN_DAYS_COLUMN in data.columns else {})
    config = {
        'encoder_days': ENCODER_DAYS,
        'decoder_days': DECODER_DAYS,
        'encoder_features': windowing.ENCODER_FEATURES,
        'decoder_features': windowing.DECODER_FEATURES,
        'target_features': windowing.TARGET_FEATURES,
        'num_stations': len(station_index),
        'num_basins': len(basin_index),
        # Decoder columns built per forecast (not read from data) and the
        # per-station known-days horizon for the upstream hold rule; see
        # windowing.upstream_decoder_window and docs/INFERENCE.md.
        'window_built_decoder_features': windowing.WINDOW_BUILT_DECODER_FEATURES,
        'upstream_known_days': {k: int(v) for k, v in known.items()},
        'upstream_registry': 'upstream_gauges.json',
        'basin_precip_known_days': windowing.BASIN_PRECIP_KNOWN_DAYS,
        'basin_zones': 'basin_zones.json',
        'event_weight': EVENT_WEIGHT,
        'lead_weight_power': LEAD_WEIGHT_POWER,
        'linear_skip': True,
        'ensemble_size': len(members),
        'final_refit': refit_done,
        'training_years': int(os.getenv('OPENFLOW_TRAINING_YEARS', '') or combine_data.TRAINING_YEARS),
        # Sources switched off for this run (ablation levers / missing
        # credentials). Inference must switch off the same ones so it sees
        # the placeholders the model was trained on.
        'disabled_sources': sorted(
            name for name in ('SMAP', 'DROUGHT', 'RESERVOIR', 'UPSTREAM', 'BASIN_PRECIP')
            if os.getenv(f'OPENFLOW_DISABLE_{name}', '').strip() in ('1', 'true', 'True')),
    }
    with open(os.path.join(base_path, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("Saved model -> %s, training_config.json alongside scalers/index JSON",
                h5_path)
    # Phase 6 mobile conversion (CoreML + TFLite + manifest) is run as a
    # separate workflow step against this same .h5 so that a conversion
    # regression cannot lose the trained model. See export_mobile.py and
    # the "Export mobile artifacts" step in ml_training.yml.


if __name__ == '__main__':
    main()
