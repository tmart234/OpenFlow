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
| `station_attributes.json` | Optional Phase 5 static physical catchment attributes (EPA StreamCat watershed metrics: area, elevation, relief, climate normals, BFI, soil permeability/texture, bedrock depth, forest/shrub cover, wetness index, dam storage, ungauged fraction). Zero runtime API calls needed. |
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
- **Decoder (next 14 days):** forecast weather, routed upstream flow, basin rain/snow, energy/intensity, ensemble spread, and RFC/NWM prior using the hold and mask rules from `windowing`.
- **Output:** daily min/max cfs, inverted with `scalers.json` (and multiplied by `area_km2 / 2.446575545` when `training_config.flow_units == "mm_day"`).

Run it from the CLI with `python openFlowML/flow_forecast.py --site taos-box --model-dir <release dir>`. That draws the model's daily min–max next to the routing or NWS forecast (and the blended routing+ML curve from `openFlowML/fusion.py`). The **Flow Forecast** workflow does the same automatically with the newest `model-*` release and publishes the Daily Mobile Feature Feed (`openFlowML/feature_feed.py`).

**Units & weather source.** Point weather (`TMIN` / `TMAX` in **tenths of °C**, `precipitation` in mm) defaults to gauge-collocated 4-km gridMET (`training_config.weather_source = "gridmet"`), matching gauge-point Open-Meteo / NWS forecasts without nearby-GHCND elevation offsets. Legacy releases (without `weather_source`, or `"noaa"`) used raw NOAA GHCND daily summaries filled from gridMET.

Forecast temperatures must be converted to these units (°C × 10). (`data/get_forecast.py` returns °F by default when `temperature_unit='fahrenheit'`, which does not match; pass `temperature_unit='celsius'` and multiply by 10.)

**Gaps between the data sources:**
- **Mixed upstream sources:** for CO DWR upstream and anchor gauges, the daily archive the model trained on wins wherever it has a value. NWPS observations and RFC forecasts are appended only after the archive ends, multiplied by the median archive/NWPS ratio over their last 14 common days (`ml_inference.splice_on_training_scale`). In Oct 2026 the DWR Lobatos record read about half the USGS-based NWPS series; letting NWPS overwrite the archive fed the model an upstream level it never saw in training.
- **Station outages:** in legacy `"noaa"` mode, days the nearest NOAA station did not report take the gridMET cell at the gauge (`get_basin_precip.gridmet_point_weather`), shifted by that calendar month's mean station-minus-gridMET difference (`combine_data.fill_weather_gaps`).
- **Weather reporting lag:** gridMET / NOAA daily observations run ~2–4 days behind real time. The reference bridges those days by interpolating temperature toward today's gauge-point forecast, with precipitation from the observed local-zone basin rain.
- **Disabled sources:** `training_config.disabled_sources` lists data sources that training switched off (e.g. `SMAP` without Earthdata credentials). Inference switches the same ones off, so the model sees the placeholders it was trained on.

## Input tensors

The model takes a dict of 5 base inputs, plus an optional 6th `static_input` when `training_config.static_features` is non-empty. Names are stable across exports:

| Name | dtype | Shape | Source |
| --- | --- | --- | --- |
| `encoder_input` | float32 | `[1, 60, len(encoder_features)]` | Last 60 days of features, columns in `training_config.encoder_features` order, pre-scaled. |
| `decoder_input` | float32 | `[1, 14, len(decoder_features)]` | Next 14 days of forecast-time-available features, columns in `training_config.decoder_features` order, pre-scaled. |
| `persistence_input` | float32 | `[1, len(target_features)]` | Last encoder-day flow values (the `Min Flow` / `Max Flow` columns from the last encoder row), in scaled space. |
| `station_input` | int32 | `[1]` | `station_index.json[site_id]`, or `0` if the site isn't in the map. |
| `basin_input` | int32 | `[1]` | `basin_index.json[huc8]`, or `0` if the HUC isn't in the map. |
| `static_input` *(optional)* | float32 | `[1, len(static_features)]` | Present only when `training_config.static_features` is set. Z-scored EPA StreamCat physical attributes from `station_attributes.json` + `training_config.static_scalers` (zeros fallback for unknown sites). |

`encoder_days = 60`, `decoder_days = 14`. Both are recorded in `training_config.json` so a future model with a different window size doesn't silently break clients.

### Scaling inputs

Every numeric column in `encoder_input` and `decoder_input` is z-scored; heavy-tailed flow and precipitation columns (`Min Flow`, `Max Flow`, `precipitation`, `upstream_flow`, `rain_*`, `snow_*`, `precip_max1h`, `precip_max3h`, `prior_flow`) are additionally `log1p`-transformed before scaling. When `training_config.flow_units == "mm_day"` (Phase 5 opt-in specific discharge), divide flow inputs (`Min Flow`, `Max Flow`, `upstream_flow`, `prior_flow`) by `area_km2 / 2.446575545` (yielding mm/day) *before* `log1p` + z-score, and multiply inverted outputs back by `area_km2 / 2.446575545` to recover cfs.

```python
# scalers.json shape:
#   {"Min Flow": {"mean": ..., "scale": ..., "transform": "log1p"}, ...}
def scale(raw, params):
    x = math.log1p(raw) if params["transform"] == "log1p" else raw
    return (x - params["mean"]) / params["scale"]
```

Indicator columns (`*_observed`, `*_known`, `forcing_is_forecast`) and `lead_day_norm` are **not** z-scored — they don't appear in `scalers.json`.

## Upstream-gauge inputs

Stations listed in `upstream_gauges.json` (e.g. `USGS:08276500`, the Rio Grande below Taos Junction Bridge / Lower Taos Box) get two extra encoder columns and two extra decoder columns. They let the model see water that is already in the river upstream: a flood wave in Colorado that will reach the Taos Box days later, which local weather and the station's own history cannot show.

Registered stations:
- Taos Box, Royal Gorge, Conejos and Dolores (since the upstream inputs were added).
- Since 2026-11: the Colorado near Dotsero (Kremmling + Eagle), the Gunnison near Gunnison (Taylor + East at Almont) and the Colorado near the state line (Cameo + Gunnison near Grand Junction). All three use CBRFC forecasts at the upstream gauges (`forecast_days` 5).

An app that does not yet build the upstream inputs must still send the placeholders described below for every registered station.

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

The routing CLI (`flow_forecast.py`) uses the same anchor, hourly (`routing.anchored_input`), continued after Lobatos's latest reading with the Alamosa/Mogote observations and RFC forecasts by the same rule. For the Taos Box it then goes one gauge closer (registry `routing.anchor`, routing CLI only). It routes the Rio Grande near Cerro, which crests a median 11 h before Taos Junction Bridge. After Cerro's latest reading it continues Cerro with Lobatos routed to Cerro's clock (`from_anchor`, changes passed 1:1). Both reaches use a flow-dependent travel time (`celerity`: lag_ref · (q / q_ref)^−exponent, `routing.variable_shift`), because a flood wave moves faster than low water. Gauged tributaries below Cerro (`routing.tributaries`: Rio Pueblo de Taos, Red River) are added by mass balance at their measured travel times. Without Cerro data it routes Lobatos; without Lobatos data it routes the upstream gauges directly. The ML model is unaffected: it keeps the daily Lobatos anchor it was trained on.

**`upstream_observed` (encoder only, 0/1).** Set it to 1 when every upstream gauge has a value for that day. Otherwise, and for every station without upstream gauges, set `upstream_flow` to `0.0` *in scaled space* and `upstream_observed` to 0.

**`upstream_known` (decoder only, 0/1)** is built per forecast, not read from data. Let `K = training_config.upstream_known_days[site_id]` (0 when the site is absent). This equals `min(lag_days) + forecast_days`, capped at `decoder_days`. For decoder day `d` (0-based):

- `d < K` and the routed value exists: use it, `upstream_known = 1`.
- otherwise: **hold** the last known value (starting from the last encoder day's `upstream_flow`) and set `upstream_known = 0`.

**`upstream_change` (decoder only)** depends on `training_config.upstream_change_mode`:
- `"mass_balance_hold"` (current): physically consistent additive mass-balance log-flow delta `(log1p(max(q0_mid + u_dec - u0, 0)) - log1p(q0_mid)) / flow_scale` in cfs inverted via `scalers.json`, **held** past `K` using the held `upstream_flow` (eliminating the Day `K+1` cliff). Zero on every day when the last encoder day's `upstream_observed` is 0.
- `"legacy"` (omitted in older releases): `upstream_flow` (decoder, scaled) minus the last encoder day's `upstream_flow` (scaled) on days with `upstream_known = 1`, and 0 on later days.

The model's linear skip head reads `upstream_change` directly, so an incoming upstream rise moves the forecast (`windowing.upstream_change`). The skip has a shared coefficient plus a learned per-station gain (`training_config.river_upstream_gain`). Rivers whose upstream gauges carry almost all of their flow and rivers where they carry a fraction can respond differently. Clients need do nothing: the gain is looked up inside the model from `station_input`.

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

## Extended physical forcing & operational prior (Phases 1–3)

Models trained on the expanded feature schema (`training_config.encoder_features` / `decoder_features`) include five additional physically grounded blocks, each paired with an unscaled 0/1 indicator (`*_observed` in the encoder, `*_known` in the decoder):

- **Radiation & evaporative demand (`srad`, `pet`, `vpd`):** daily shortwave radiation (MJ/m²/day), reference ET (mm/day), and vapor pressure deficit (kPa) from gridMET (encoder) and Open-Meteo extended forecast (decoder, `energy_known_days = 14`).
- **Sub-daily rainfall intensity (`precip_max1h`, `precip_max3h`, `precip_wet_hours`):** hourly convective intensity & wet-hour duration from ERA5-Land + GFS (or NLDAS-2 when `hydro_source = "nldas"`) in the encoder and Open-Meteo hourly forecast in the decoder (`precip_intensity_known_days = 7`).
- **Continuous soil moisture (`sm_surface`, `sm_rootzone` + `hydro_sm_observed`, encoder only):** daily surface and root-zone volumetric soil moisture from ERA5-Land + bias-aligned GFS bridge (or NLDAS-2).
- **Elevation-band gridded SWE (`swe_low`, `swe_mid`, `swe_high`, `delta_swe_mid`, `delta_swe_high` + `gridded_swe_observed`, encoder only):** basin snow water equivalent and 1-day melt rate across three elevation bands.
- **Real-forecast provenance & weather ensemble spread (decoder only):** `precip_ens_std`, `precip_ens_p90`, `temp_ens_std` (`ens_spread_known`), `forcing_is_forecast` (`1.0` at live inference, `0.0` on observed-weather training windows), and `lead_day_norm` (`(d + 1) / decoder_days`).
- **Operational streamflow prior (`prior_flow`, `prior_observed` in encoder; `prior_flow`, `prior_known`, `prior_change` in decoder):** NWS River Forecast Center forecast extended by NOAA National Water Model v3 medium-range blend (`data/get_nwm_prior.py`, `prior_known_days = 10`).

Because assembling all of these sources identically on iOS/Android is complex (and optional NASA Earthdata sources require credentials that must never live in a mobile binary), mobile clients should consume the **Daily Mobile Feature Feed** below rather than reimplementing each upstream scraper on device.

## Daily Mobile Feature Feed (Phase 4)

The daily **Flow Forecast** workflow (`.github/workflows/flow_forecast.yml`) runs `openFlowML/feature_feed.py` against the latest published `model-*` release and publishes compact, pre-assembled input tensors (~15 KB per station) to the `forecast-archive` branch under `feed/`:

- **Catalog:** `https://raw.githubusercontent.com/tmart234/OpenFlow/forecast-archive/feed/catalog.json`
- **Per-station payload:** `https://raw.githubusercontent.com/tmart234/OpenFlow/forecast-archive/feed/inputs/<slug>.json`

Each station JSON contains:
- `feed_schema_version` (`1`), `site_id`, `slug`, `name`, `generated_utc`, `model_version`
- `schema_sha256` — SHA-256 fingerprint of the model input/output schema (`feature_feed.schema_digest(training_config)`). Compare against the downloaded model bundle before running CoreML / TFLite.
- `anchor_date`, `forecast_dates` (14 ISO dates), `flow_units` (`"cfs"` or `"mm_day"`), `area_km2`
- `shapes` and `inputs` — ready-to-run float32 / int32 arrays (`encoder_input`, `decoder_input`, `persistence_input`, `station_input`, `basin_input`, and optional `static_input`), rounded to 5 decimal places (`< 1e-5` parity vs live Python inference)
- `inputs_sha256` — canonical JSON SHA-256 integrity checksum over `inputs` (`feature_feed.verify_feed_payload`)

Mobile apps verify `sha256` (from `catalog.json`), `inputs_sha256`, and `schema_sha256`, pass `inputs` straight into CoreML / TFLite, and invert the outputs with `scalers.json` — achieving **100% feature parity** with server-side inference with **zero API credentials on device**. In Python, `ml_inference.predict_from_feed(art, payload)` is the reference consumer.

## Output tensors

**`persistence_plus_delta`** is the point forecast: shape `[1, 14, len(target_features)]` (= `[1, 14, 2]`), dtype `float32`. The two target columns are `Min Flow`, `Max Flow` in that order — same as `training_config.target_features`. Every release has it.

**`flow_quantiles`** is the forecast range: shape `[1, 14, len(target_features), len(quantiles)]` (= `[1, 14, 2, 5]`), dtype `float32`.
- Present when `training_config.quantiles` is set (also `manifest.schema.quantiles`; `outputs` lists both names). Older releases do not have it.
- The last axis follows `training_config.quantiles`, `[0.05, 0.25, 0.5, 0.75, 0.95]`. Example: `[0, d, 1, 4]` is the 95th-percentile daily maximum on decoder day `d`.
- The 0.5 slot equals `persistence_plus_delta`.
- Quantiles never cross: each is at least the one before it.

Both outputs are in **scaled** space. To get cfs, invert per column; the transform is monotonic, so quantiles stay quantiles:

```python
def invert(z, params, flow_units="cfs", area_km2=None):
    x = z * params["scale"] + params["mean"]
    val = math.expm1(x) if params["transform"] == "log1p" else x
    if flow_units == "mm_day" and area_km2:
        val = val * area_km2 / 2.446575545
    return val
```

### Reading the forecast range

The 5–95% pair is a 90% range and the 25–75% pair a 50% range. On the held-out years they contain about 90% and 50% of what the river did.
- The spread is set per lead day on the validation years (split-conformal, `training_config.band_calibration`), refined per station (`band_station_calibration`), and widened selectively on predicted rising limbs (`band_rise_calibration`, baked into `QuantileBand.rise_upper_scale`) so flood peaks reach nominal 90% coverage without widening steady days.
- `quality_report` shows the coverage achieved on the test years.

**Caveat.** Even with archived real weather forecasts injected where available (`real_forecast_training = true`), live ranges beyond day 3 remain sensitive to storm track misses. The daily forecast archive (`score_archive.py`) is where live coverage is tracked continuously.

`ml_inference.predict` returns the range as `min_q05_cfs` … `max_q95_cfs` columns, and `flow_forecast.py` JSON as the same keys per day.

## Residual structure

The model is a residual forecaster: `persistence_plus_delta = persistence_input + delta`, where `delta` is what the network actually learns. If the network outputs zero, predictions equal the persistence baseline — that's the worst case, not a failure mode. There's no special-casing for this in the consumer; just decode the output as above.

## Mobile model files

Both mobile files keep the input names above and name their output `persistence_plus_delta`:
- **TFLite:** it is the signature output key, so run it with `interpreter.getSignatureRunner("serving_default")`.
- **CoreML:** read it with `featureValue(for: "persistence_plus_delta")`.

Releases before this contract was enforced exposed `output_0` (TFLite) and shipped no CoreML file.

The CoreML file is converted from a copy of the model with its LSTMs unrolled over the fixed 60 + 14 days. The weights and outputs are identical. Unrolling is needed because coremltools cannot convert the Keras LSTM's while-loop.

## TFLite Flex delegate

The graph contains two embedding lookups and an LSTM with `initial_state`. The exporter tries three TFLite conversion paths and records which one was used in `manifest.json.tflite_mode`:

- `builtins` — strict TFLITE_BUILTINS. No Flex needed on the consumer side. Current models always take this path.
- `select_tf_ops` — needed the Flex delegate. Android consumers must add `org.tensorflow:tensorflow-lite-select-tf-ops` to their `app/build.gradle`. iOS uses CoreML, not TFLite.
- `rebuilt_no_mask` — only for models trained before the embeddings dropped `mask_zero=True`. The exporter cloned the model with `mask_zero=False`, which is functionally identical because the mask never reached the LSTMs. No Flex needed.

Check `manifest.tflite_mode` on download and warn the user if the Flex delegate isn't bundled.

## Unseen stations and basins

Both embedding layers reserve index `0` for unknown. If the user picks a site not in `station_index.json`, pass `station_input = 0` — the model is trained with `StationDropout` to handle this via basin-only inference. Same for `basin_input` when the HUC8 isn't known. When `training_config.static_features` is enabled (Phase 5), also pass `static_input` built from `station_attributes.json` + `training_config.static_scalers` (or zeros if the river is absent), which conditions both the LSTM context and the attribute-gated upstream skip for zero-shot transfer.

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
