# OpenFlow ML model: design record

Why the forecast model is built the way it is, how it compares with current practice in ML streamflow forecasting, and what is still open. Reviewed 2026-11-04 against the 15-year, 7-station training set. `docs/INFERENCE.md` is the input/output contract; `openFlowML/model.py` is the code.

## Model family: still an LSTM

LSTMs remain the standard model for daily streamflow:
- They set the large-sample benchmarks (Kratzert et al. 2019).
- They are the core of Google's operational global flood forecasting (Nearing et al. 2024).
- Transformer and state-space alternatives have not shown consistent gains on daily river flow.

The structure that matters is the *hindcast/forecast* split, which OpenFlow already has:
- The encoder LSTM reads the past 60 days.
- Its state seeds a decoder LSTM.
- The decoder walks the 14 forecast days on forecast-time inputs only.

Keep the LSTM.

## The design, piece by piece

| Choice | Status | Why |
|---|---|---|
| Encoder (60 d history) → decoder (14 d) LSTM, state handoff | current practice | Same structure as operational hindcast/forecast LSTMs. Gives a clean split between past inputs and forecast-time inputs (`windowing.py` enforces no flow in the decoder). |
| Observed flow in the encoder, output = persistence + learned delta | current practice for gauged forecasting | Feeding recent observations (autoregression / data assimilation) is the strongest short-lead skill source (Nearing et al. 2022). The residual head makes "no change" the worst case. |
| log1p + z-score on flow and precipitation | current practice | Flow is roughly log-normal; errors become relative, so small and large rivers count alike. |
| Huber loss (δ = 0.05 in scaled log flow, ≈ absolute error) | kept | Median-seeking, which matches the reported metric (median absolute % error). |
| Event and lead-day sample weights | kept | Puts rises, recessions and day 1 back in charge of a loss that steady days otherwise dominate. |
| Station + basin embeddings, StationDropout | kept, cleaned | Lets one network learn all 7 rivers. Index 0 is a trained "unknown station" vector. The `mask_zero=True` on the embeddings did nothing (the mask stops at `RepeatVector`), so it was removed. |
| Seed ensemble (3), chronological split with embargo, early stopping, refit on all years | current practice | Averaging seeds removes one network's noise. The split and embargo prevent look-ahead. |
| **Probabilistic range** (`flow_quantiles`) | **added** | Operational flood forecasts are probabilistic (distributional heads: Klotz et al. 2022, Nearing et al. 2024). See below. |
| Static catchment attributes instead of IDs | not yet | Matters for ungauged rivers and many stations; see next steps. |
| Training on archived weather *forecasts* | not yet | The biggest remaining gap; see next steps. |

## Experiments (held-out test years, 2026-11-04)

One ensemble member per run, two seeds each, same data, same issue dates scored. Metrics:
- **Test MAE:** mean absolute error in scaled log flow. Lower is better; the seed-to-seed spread is about ±0.0005.
- **Error:** median absolute % error of the daily maximum on days 1 / 7 / 14.

| Variant | What changed | Test MAE | Error d1 / d7 / d14 | Rise peak ratio | Verdict |
|---|---|---|---|---|---|
| base | current recipe | **0.1095** | 3.2 / 8.7 / 12.2 | 0.80 | — |
| fgb3 | LSTM forget-gate bias initialised at 3 (NeuralHydrology default) | 0.1113 | 3.4 / 8.5 / 12.3 | 0.80 | worse overall; not adopted |
| nhdrop | dropout on the LSTM output instead of input/recurrent dropout | 0.1102 | 3.4 / 8.8 / 12.7 | 0.81 | worse at day 1 and day 14; not adopted |
| h128 | 128-unit LSTMs instead of 64 | 0.1109 | 3.2 / 8.6 / 12.2 | 0.79 | worse overall; not adopted |
| enc180 | 180-day encoder instead of 60 (one seed) | 0.1130 | 3.2 / 8.6 / 12.6 | 0.81 | worse, 3× the training time, and 13% fewer training windows; not adopted |
| q_cal | + calibrated probabilistic range | **0.1096** | 3.2 / 8.6 / 12.5 | 0.81 | point forecast unchanged, range calibrated; adopted |

The current hyperparameters hold up against the literature defaults: a few variants win one metric, none wins overall. A plausible reason a wider or longer-memory network does not help is the small sample (7 rivers): the recent-flow, SWE and soil-moisture inputs already carry most of a basin's storage state.

The harness trained one member per run, as `train.fit_members` does (same weights, callbacks and early stopping). The split was by issue date, using the per-station cutoffs of the 60-day chronological split, so every variant was scored on the same 5,443 forecasts. One gauge (USGS:09163500) lost about 4 years of history to USGS timeouts during this fetch, the same for every variant, so absolute numbers differ slightly from CI's.

## Probabilistic range

**Outputs.** `flow_quantiles` gives the 5 / 25 / 50 / 75 / 95 % quantiles of daily min and max flow for each forecast day.

**Construction** (`model.QuantileBand`):
- The 50% quantile is the point forecast.
- The other quantiles are non-negative gaps away from it, so they cannot cross.
- The gap head reads the decoder state through a stop-gradient. Its pinball loss cannot change the point forecast, and early stopping follows the point loss.
- After training, the gaps are scaled per lead day so the outer quantiles hit 5% / 95% on the validation years. This is normalised split-conformal calibration (Lei et al. 2018; Romano et al. 2019).

**Calibration on the test years** (single members, two seeds):
- The 90% range held **86–89%** of observed daily maxima at every lead day (uncalibrated: 82–94%).
- The 50% range held **45–50%**.
- In the full production recipe (3-member ensemble), the 90% range held 85–88% and the 50% range 46–49%.
- The 90% range widens from 18% of the forecast on day 1 to 67% on day 14.
- The point forecast matched the published model-214 to within ±0.1 pt at every reported lead day.

**Known weak spot.** On rising events only about 69% of observed 7-day peaks fall under the forecast 95th percentile. Floods are rarer than the ordinary days that dominate calibration, so the range is still too narrow on big rises.

## Mobile exports (fixed 2026-11-04)

Recent releases could not run on either app:
- **CoreML:** conversion failed on the Keras LSTM's while-loop, so no `.mlpackage` shipped.
- **TFLite:** the signature output was named `output_0`, but both apps read `persistence_plus_delta`.

The exporter now converts an unrolled copy for CoreML and names the outputs for both formats. Parity tests drive the exports the way the apps do: by name, on the ensemble CI ships.

## Next steps, by expected value

1. **Train the decoder on archived forecasts, not observed weather.** Training and the held-out report use the observed weather, basin rain/snow and upstream flow as a "perfect forecast". The model therefore learns to trust forecasts more than it should, and the held-out skill and range are optimistic beyond about day 3. Operational systems train the forecast LSTM on archived forecast products. Until then, `score_archive.py` scores the live forecasts and live range coverage.
2. **More rivers.** An LSTM learns hydrology from the variety of basins it sees (Kratzert et al. 2024). Seven gauges is small. Adding Colorado/New Mexico gauges, even ones the app does not show, should help every station.
3. **Static catchment attributes** (area, elevation, snow fraction, aridity) alongside or instead of the station embedding. This is what makes predictions at unseen gauges work.
4. **Flood peaks.** The point forecast still under-calls rising-event peaks (median peak ratio ≈ 0.80, at the gate's limit). Options worth testing: regime-aware calibration of the range (e.g. separate factors when upstream is rising or rain is forecast), and an upper-quantile alert in the app.

## References

- Kratzert, F. et al. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS* 23.
- Kratzert, F. et al. (2024). Never train a Long Short-Term Memory (LSTM) network on a single basin. *HESS* 28.
- Klotz, D. et al. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. *HESS* 26.
- Nearing, G. et al. (2022). Data assimilation and autoregression for using near-real-time streamflow observations in long short-term memory networks. *HESS* 26.
- Nearing, G. et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature* 627.
- Lei, J. et al. (2018). Distribution-free predictive inference for regression. *JASA* 113.
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
