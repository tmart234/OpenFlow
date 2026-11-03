# OpenFlow Inference Contract

This document specifies what the [OpenFlowMobile](https://github.com/tmart234/OpenFlowMobile) app (or any other downstream consumer) needs to know in order to run the trained model end-to-end. If you find yourself reading `train.py` to figure out a tensor shape, this doc has failed — please open an issue.

## Release artifacts

Every successful training run on `dev` or `main` is published by the **Release Model** workflow (`.github/workflows/release_model.yml`) as a GitHub Release tagged `model-YYYY.MM.DD`, with a `-r2`, `-r3`, ... suffix on same-day releases. A run is published only if:
- the manifest checksums match;
- a mobile model was exported;
- the quality gate passed (`openFlowML/release_check.py`).

The release notes carry the quality report and link the training run and commit. The release attaches:

| File | Purpose |
| --- | --- |
| `lstm_model.h5` | Canonical Keras model. The source of truth; everything else is derived. |
| `lstm_model.mlpackage.zip` | CoreML mlprogram for iOS (iOS 15+). Unzip and pass to `MLModel(contentsOf:)`. |
| `lstm_model.tflite` | TFLite (float32) for Android. May require the Flex delegate — see TFLite note below. |
| `lstm_model_int8.tflite` | Optional dynamic-range int8 TFLite. Smaller, slightly less accurate. Only present when it passed the parity gate; see `manifest.tflite_int8_max_abs_diff`. |
| `scalers.json` | Per-column scaler params (mean, scale, transform). Inputs and outputs are in scaled space; you invert with this. |
| `station_index.json` | Map `site_id` → embedding index. Index `0` is reserved for "unseen". |
| `basin_index.json` | Map HUC8 → embedding index. Index `0` is reserved for "unseen". |
| `training_config.json` | Schema: encoder/decoder window sizes, feature names, vocab sizes, per-station `upstream_known_days`. |
| `upstream_gauges.json` | Upstream-gauge registry: for each station, the upstream gauges (training `site_id`, NWS `nws_lid`), `lag_days`, `weight`, `forecast_days`, and an optional `anchor` gauge. Needed to build the `upstream_flow` inputs. |
| `basin_zones.json` | Per-station travel-time precipitation zones: zone polygons (local / mid / far by river distance), the ungauged area, and area-weighted sample `points` with their NWS gridpoint (`WFO/x,y`). Needed to build the basin rain/snow inputs. |
| `manifest.json` | sha256 + byte size per artifact, plus tool versions and the schema block. Verify on download. |
| `quality_report.json` / `.md` | Held-out verification of this training recipe (`openFlowML/evaluate.py`). Per station and lead day: median absolute % error of the daily maximum flow vs persistence, flood-peak ratio and timing, and recession error, plus the quality gate result. CI publishes a release only when the gate passes. |

The model is an **average of several networks** (`training_config.ensemble_size`), each trained from a different random seed, wrapped in one Keras graph (`model.average_ensemble`). Inputs and outputs are unchanged, so consumers need do nothing. With `training_config.final_refit = true`, the published networks were retrained on every window, including the test years, after `quality_report` was measured on the held-out run.

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

## Python reference implementation

`openFlowML/ml_inference.py` runs a trained model for a station as of today. It is the executable version of this document:
- **Encoder (last 60 days):** built with the same code as training (`combine_data`). Gaps are bridged as described below.
- **Decoder (next 14 days):** forecast weather, routed upstream flow and basin rain/snow, using the hold and mask rules from `windowing`.
- **Output:** daily min/max cfs, inverted with `scalers.json`.

Run it from the CLI with `python openFlowML/flow_forecast.py --site taos-box --model-dir <release dir>`. That draws the model's daily min–max next to the routing or NWS forecast. The **Flow Forecast** workflow does the same automatically with the newest `model-*` release.

**Units.** Training used raw NOAA GHCND daily summaries:
- `TMIN` / `TMAX` in **tenths of °C**;
- `precipitation` in mm.

Forecast temperatures must be converted to these units (°C × 10). (`data/get_forecast.py` returns °F, which does not match.)

**Gaps between the data sources:**
- **Station outages:** days the nearest NOAA station did not report take the gridMET cell at the gauge (`get_basin_precip.gridmet_point_weather`). Temperatures are shifted by that calendar month's mean station-minus-gridMET difference (`combine_data.fill_weather_gaps`). Training does the same, so a station that stops reporting no longer removes those days from the training data.
- **NOAA lag:** NOAA daily data runs 2–4 days behind. The reference bridges those days by interpolating temperature toward the NWS forecast for today, with precipitation from the observed local-zone basin rain.
- **Disabled sources:** `training_config.disabled_sources` lists data sources that training switched off (e.g. `SMAP` without Earthdata credentials). Inference switches the same ones off, so the model sees the placeholders it was trained on.

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

## Upstream-gauge inputs

Stations listed in `upstream_gauges.json` (e.g. `USGS:08276500`, the Rio Grande below Taos Junction Bridge / Lower Taos Box) get two extra encoder columns and two extra decoder columns. They let the model see water that is already in the river upstream: a flood wave in Colorado that will reach the Taos Box days later, which local weather and the station's own history cannot show.

**`upstream_flow` (encoder and decoder)** is the routed daily inflow:

```
upstream_flow(day d) = sum over upstream gauges i of  weight_i * Q_i(d - lag_days_i)
```

where `Q_i` is gauge *i*'s daily-mean flow in cfs. Scale it like flow, with the `upstream_flow` entry in `scalers.json` (log1p, then z-score).

- **Encoder days:** use observed daily means. Training used each gauge's `site_id`: the CO DWR daily archive for `DWR:` ids, USGS IV averaged to daily for `USGS:` ids. NWPS (`https://api.water.noaa.gov/nwps/v1/gauges/{nws_lid}/stageflow`, `observed`; flow is `secondary` in kcfs) covers about the last 30 days.
- **Decoder days:** for days `d < lag_days_i`, `Q_i(d - lag_days_i)` is still an observation. After that, use the daily mean of the NWS River Forecast Center forecast at `nws_lid` (the `forecast` half of the same NWPS payload).

**Anchor gauge (stations with an `anchor` entry; currently the Taos Box, anchored at Rio Grande near Lobatos).** Some rivers lose most of an upstream rise to diversions before the target. Between Alamosa / Mogote and the Taos Box, the San Luis Valley irrigation ditches take most of it from April to October. For such a station, `upstream_flow` is the anchor gauge below the diversions instead of the sum above:

```
supply(d)         = sum over upstream gauges i of weight_i * Q_i(d - lag_days_i)       (as above)
upstream_flow(d)  = weight_a * Q_anchor(d - lag_days_a)                       while the anchor has a value
                  = max(0, A_last + passed(supply(d) - supply(d_last)))        after its last value
r                 = clip(mean(A) / mean(supply), 0, 1.5) over the last 7 days both exist up to d_last
passed(dS)        = r * min(dS, C) + max(r, p) * max(dS - C, 0)
```

`A_last` is the anchor value on `d_last`, its last routed date. Training only ever sees the first line, because the anchor is observed every day. The continuation carries the upstream RFC forecasts through the diversions:
- Falls, and the first `C` cfs of a rise, pass at the current ratio `r`.
- Ditch capacity is limited, so the rest of a big rise passes at `max(r, p)`.
- Defaults are `C = 500` cfs (`divert_capacity_cfs`) and `p = 0.5` (`flood_pass_through`); an anchor entry can override them. In the late-October 2026 rain flood about half of a ~6,000 cfs rise reached Lobatos while `r` was about 0.15.

`get_upstream.anchored_daily` and `get_upstream.passed_change` are the reference implementations. The anchor counts toward "every gauge has a value" for `upstream_observed`.

The routing CLI (`flow_forecast.py`) uses the same anchor, hourly (`routing.anchored_input`). It routes Lobatos to the Taos Box, after its latest reading continues it with the Alamosa/Mogote observations and RFC forecasts by the same rule, and falls back to routing the upstream gauges directly when Lobatos has no recent data.

**`upstream_observed` (encoder only, 0/1).** Set it to 1 when every upstream gauge has a value for that day. Otherwise, and for every station without upstream gauges, set `upstream_flow` to `0.0` *in scaled space* and `upstream_observed` to 0.

**`upstream_known` (decoder only, 0/1)** is built per forecast, not read from data. Let `K = training_config.upstream_known_days[site_id]` (0 when the site is absent). This equals `min(lag_days) + forecast_days`, capped at `decoder_days`. For decoder day `d` (0-based):

- `d < K` and the routed value exists: use it, `upstream_known = 1`.
- otherwise: **hold** the last known value (starting from the last encoder day's `upstream_flow`) and set `upstream_known = 0`.

**`upstream_change` (decoder only)** is `upstream_flow` (decoder day, scaled) minus the last encoder day's `upstream_flow` (scaled) on days with `upstream_known = 1`. It is 0 on other days, and on every day when the last encoder day's `upstream_observed` is 0. The model's linear skip head reads it directly, so an incoming upstream rise moves the forecast (`windowing.upstream_change`).

`windowing.upstream_decoder_window` is the single reference implementation; training calls it for every window. In training, the observed upstream flow stood in for the RFC forecast on known days. This is the same "perfect forecast" assumption the weather columns make, so expect somewhat lower skill on known days when the RFC forecast misses.

`openFlowML/flow_forecast.py` builds these same upstream inputs from live NWPS data. It routes them hourly, without the ML model, to produce the CLI and CI forecast charts.

## Basin rain / snow inputs (travel-time zones)

Every station in `basin_zones.json` has its contributing area (from the USGS NLDI network; closed basins excluded) split into three zones by river distance to the gauge. The default split is `zone_km = [100, 250]`:

- `local` (100 river-km or less)
- `mid` (100–250 river-km)
- `far` (over 250 river-km)

The model gets one rain and one snow column per zone:

- encoder and decoder: `rain_local`, `rain_mid`, `rain_far`, `snow_local`, `snow_mid`, `snow_far`
- encoder only: `basin_precip_observed`
- decoder only: `basin_precip_known`

**Units and scaling.** Values are the zone-mean daily precipitation in mm, split into rain and snow (water equivalent). Scale each column with its `scalers.json` entry (log1p, then z-score).

**Rain vs snow** uses one rule everywhere: the liquid fraction is `clip((T + 1) / 4, 0, 1)` with `T` the daily mean air temperature in °C (`basin_zones.liquid_fraction`). Rain is `precip × fraction`; snow is the rest.

**Encoder days.** Training used gridMET daily grids (`pr`, `tmmn`, `tmmx`) averaged over each zone polygon. At inference:

- Use gridMET (about 2 days behind real time).
- Fill the remaining days with the NWS Stage IV daily analysis: `nws_precip_1day_YYYYMMDD_conus.nc`, covering the 24 h ending at 12Z on that date, mapped to the previous local day. Split rain/snow using the zone's latest gridMET mean temperature.
- `basin_precip_observed` = 1 when the day has a value. Otherwise set all six columns to `0.0` *in scaled space* and the indicator to 0.

**Decoder days.** Use NWS gridpoint forecasts (`https://api.weather.gov/gridpoints/{nws}` for each sample point):

- `quantitativePrecipitation` is 6-hourly mm, spread evenly over its hours; `temperature` is hourly.
- Sum to local calendar days per point, split rain/snow on that day's mean temperature, then take the area-weighted mean over the zone's points. Weights are `w`, normalised within the zone.
- Decoder day `d` (0-based) keeps its values with `basin_precip_known = 1` when `d < training_config.basin_precip_known_days` (7) and the forecast exists. Later days are `0.0` with `basin_precip_known = 0`.

`windowing.basin_precip_decoder_window` implements this rule. `data/get_basin_precip.inference_daily` is the reference for assembling the daily values. The point `precipitation` column (nearest rain gauge plus point forecast) is unchanged and still required.

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

## Training and publishing a model

1. **Train.** `.github/workflows/ml_training.yml` runs:
   - every Monday on `dev`, and on demand (`gh workflow run ml_training.yml --ref <branch>`);
   - on any push to a `train/**` branch (`git push origin HEAD:train/<name>`).

   It uploads the model, the mobile exports and `quality_report.*` as the `lstm_model_keras_release` artifact. The run goes red if the quality gate or the mobile export failed.
2. **Release.** `.github/workflows/release_model.yml` publishes that artifact:
   - automatically after a successful run on `dev` or `main`;
   - for any other run on demand: `gh workflow run release_model.yml -f run_id=<training run ID>`, with `-f prerelease=true` for a test release that does not become "latest". The Flow Forecast workflow ignores pre-releases.

Verify a release has every expected file and a valid manifest:

```bash
gh release view "model-$(date -u +%Y.%m.%d)" --json assets,name
```
