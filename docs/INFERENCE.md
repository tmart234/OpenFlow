# OpenFlow Inference Contract

This document specifies what the [OpenFlowMobile](https://github.com/tmart234/OpenFlowMobile) app (or any other downstream consumer) needs to know in order to run the trained model end-to-end. If you find yourself reading `train.py` to figure out a tensor shape, this doc has failed — please open an issue.

## Release artifacts

Every successful weekly training run on `main` produces a GitHub Release tagged `model-YYYY.MM.DD` (with a `-r2`, `-r3`, ... suffix on same-day re-runs). The release attaches:

| File | Purpose |
| --- | --- |
| `lstm_model.h5` | Canonical Keras model. The source of truth; everything else is derived. |
| `lstm_model.mlpackage.zip` | CoreML mlprogram for iOS (iOS 15+). Unzip and pass to `MLModel(contentsOf:)`. |
| `lstm_model.tflite` | TFLite (float32) for Android. May require the Flex delegate — see TFLite note below. |
| `lstm_model_int8.tflite` | Optional dynamic-range int8 TFLite. Smaller, slightly less accurate. Only present when it passed the parity gate; see `manifest.tflite_int8_max_abs_diff`. |
| `scalers.json` | Per-column scaler params (mean, scale, transform). Inputs and outputs are in scaled space; you invert with this. |
| `station_index.json` | Map `site_id` → embedding index. Index `0` is reserved for "unseen". |
| `basin_index.json` | Map HUC8 → embedding index. Index `0` is reserved for "unseen". |
| `training_config.json` | Schema: encoder/decoder window sizes, feature names, vocab sizes. |
| `manifest.json` | sha256 + byte size per artifact, plus tool versions and the schema block. Verify on download. |

### Verifying a downloaded release

```python
import hashlib, json
with open("manifest.json") as f:
    manifest = json.load(f)
for entry in manifest["files"]:
    h = hashlib.sha256()
    with open(entry["name"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == entry["sha256"], entry["name"]
```

## Input tensors

The model takes a dict of 5 inputs. Names are stable across exports:

| Name | dtype | Shape | Source |
| --- | --- | --- | --- |
| `encoder_input` | float32 | `[1, 60, len(encoder_features)]` | Last 60 days of features, columns in `training_config.encoder_features` order, pre-scaled. Includes observed precipitation (`precipitation`, mm/day from GHCND PRCP). |
| `decoder_input` | float32 | `[1, 14, len(decoder_features)]` | Next 14 days of forecast-time-available features, columns in `training_config.decoder_features` order, pre-scaled. Includes forecast precipitation — pull `precipitation_sum` from Open-Meteo's daily forecast (`data/get_forecast.py` does this server-side; mobile clients can call Open-Meteo directly). |
| `persistence_input` | float32 | `[1, len(target_features)]` | Last encoder-day flow values (the `Min Flow` / `Max Flow` columns from the last encoder row), in scaled space. |
| `station_input` | int32 | `[1]` | `station_index.json[site_id]`, or `0` if the site isn't in the map. |
| `basin_input` | int32 | `[1]` | `basin_index.json[huc8]`, or `0` if the HUC isn't in the map. |

`encoder_days = 60`, `decoder_days = 14`. Both are recorded in `training_config.json` so a future model with a different window size doesn't silently break clients.

### Scaling inputs

Every numeric column in `encoder_input` and `decoder_input` is z-scored; the flow columns (`Min Flow`, `Max Flow`) are additionally `log1p`-transformed before scaling. Apply scalers per-column:

```python
# scalers.json shape:
#   {"Min Flow": {"mean": ..., "scale": ..., "transform": "log1p"}, ...}
def scale(raw, params):
    x = math.log1p(raw) if params["transform"] == "log1p" else raw
    return (x - params["mean"]) / params["scale"]
```

Indicator columns (`sm_observed`, `reservoir_observed`) are 0/1 and are **not** scaled — they don't appear in `scalers.json`.

## Output tensor

One output, name `persistence_plus_delta`, shape `[1, 14, len(target_features)]` (= `[1, 14, 2]`), dtype `float32`. The two target columns are `Min Flow`, `Max Flow` in that order — same as `training_config.target_features`.

The output is in **scaled** space. To get cfs, invert per column:

```python
def invert(z, params):
    x = z * params["scale"] + params["mean"]
    return math.expm1(x) if params["transform"] == "log1p" else x
```

## Residual structure

The model is a residual forecaster: `output = persistence_input + delta`, where `delta` is what the network actually learns. If the network outputs zero, predictions equal the persistence baseline — that's the worst case, not a failure mode. There's no special-casing for this in the consumer; just decode the output as above.

## TFLite Flex delegate

The graph contains two `Embedding(mask_zero=True)` layers and an LSTM with `initial_state`. The exporter tries three TFLite conversion paths and records which one was used in `manifest.json.tflite_mode`:

- `builtins` — strict TFLITE_BUILTINS. No Flex needed on the consumer side.
- `select_tf_ops` — needed the Flex delegate. Android consumers must add `org.tensorflow:tensorflow-lite-select-tf-ops` to their `app/build.gradle`. iOS uses CoreML, not TFLite.
- `rebuilt_no_mask` — exporter rebuilt the model with `mask_zero=False` and copied weights. Functionally identical at inference (the mask only affected training-time padding behavior, which the model doesn't use). No Flex needed.

Check `manifest.tflite_mode` on download and warn the user if the Flex delegate isn't bundled.

## Unseen stations and basins

Both embedding layers reserve index `0` for unknown. If the user picks a site not in `station_index.json`, pass `station_input = 0` — the model is trained with `StationDropout` to handle this via basin-only inference. Same for `basin_input` when the HUC8 isn't known.

## Reference fixture

The export parity tests in `tests/test_export_parity.py` generate synthetic inputs, run them through Keras / CoreML / TFLite, and assert they agree within `1e-3`. They're the executable spec of this document — when in doubt, read that file.

## Publishing the first release

The weekly cron at `.github/workflows/ml_training.yml` produces a Release tagged `model-YYYY.MM.DD` whenever it runs on `main`. To create the first release before the next Monday cron fires, after this branch is merged:

```bash
gh workflow run ml_training.yml --ref main
gh run watch                  # follow until done
```

Verify the release appeared with all expected files (8 if int8 shipped, 7 otherwise) and a valid manifest:

```bash
gh release view "model-$(date -u +%Y.%m.%d)" --json assets,name
```
