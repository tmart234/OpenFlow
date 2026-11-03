# Forecast verification

How OpenFlow checks its forecasts, what the checks found, and how far to trust each river's forecast. Three checks run, each against a baseline:

| Check | What it scores | Runs | Code |
|---|---|---|---|
| ML quality report | The ML model's daily maximum flow on held-out years, by station and lead day, vs a "no change" forecast | Every training run; gates the release | `openFlowML/evaluate.py`, `quality_report.md` in each `model-*` release |
| Routing crest backtest | When and how high the hourly routing forecast puts past flood crests, vs "it crests now" | PRs that touch routing, monthly, on demand (**Routing Crest Backtest** workflow) | `openFlowML/crest_backtest.py` |
| Forecast archive | OpenFlow routing, the NWS official forecast and the ML model, scored side by side on what the rivers then did | Daily (**Flow Forecast** workflow → `forecast-archive` branch) | `openFlowML/score_archive.py` |

## ML model (daily)

From the `model-2026.11.03` quality report: median absolute % error of the daily maximum flow, model / no-change forecast, on held-out test windows.

| Station | Day 1 | Day 3 | Day 5 | Day 7 | Rise peak ratio |
|---|---|---|---|---|---|
| Royal Gorge | 1.9 / 3.4 | 5.3 / 7.0 | 7.1 / 9.2 | 8.2 / 10.6 | 0.80 |
| Conejos River | 5.3 / 6.2 | 9.9 / 14.2 | 13.0 / 21.5 | 14.6 / 26.9 | 0.78 |
| Lower Taos Box | 2.7 / 2.8 | 3.6 / 6.3 | 3.8 / 8.5 | 4.5 / 10.0 | 0.74 |
| Dolores (Gateway–Dewey) | 4.8 / 8.2 | 7.2 / 19.5 | 8.4 / 27.2 | 11.1 / 30.4 | 0.79 |
| All 7 stations | 2.9 / 3.8 | 5.5 / 8.2 | 7.1 / 11.5 | 8.3 / 13.6 | 0.80 |

The model beats "no change" at every station from day 2 on. On rising water it under-calls the peak by about 20% (rise peak ratio 0.74–0.80) and can be a day off on its timing. It is a daily model, so it cannot say what hour a crest arrives. Use it for the week's trend; use routing or NWS for the crest.

## Routing crest backtest (hourly)

For every sharp crest at a routed station since 2015, `crest_backtest.py` issues routing forecasts 12–72 h before the crest. Each is calibrated on the 28 days before it, like the live forecast. A sharp crest is at least 1.5× and 150 cfs above the 72-h minimum and falls back below 75% within 72 h. Upstream gauges get their observed future as a perfect RFC forecast. Anchor and tributary gauges have no RFC forecast and are known only up to the issue time. Rain is observed gridMET rain, a perfect rain forecast. The score is:

- **timing error**: forecast crest time minus observed crest time (+ = late);
- **crest ratio**: forecast crest / observed crest.

The naive forecast "it crests now" has a timing error equal to the lead time. The CI gate fails a station whose median timing error is no better than that from 12 h of lead on, or whose median crest ratio leaves 0.7–1.3. Crests with at least 10 mm of rain on the ungauged area in the 72 h before them count as *rain-driven*.

### Results (2015 – Nov 2026, median timing error)

| River (crests) | 12 h ahead | 24 h ahead | 48 h ahead | 72 h ahead | Within ±6 h (12 h ahead) | Crest ratio |
|---|---|---|---|---|---|---|
| Royal Gorge (35) | 7 h | 7 h | 7.5 h | — | 43% | 0.75–0.82 |
| Dolores, Gateway–Dewey (60) | 6 h | 8.5 h | 8 h | — | 52% | 0.86–0.91 |
| Lower Taos Box (54) | 8 h | 14 h | 25.5 h | 41 h* | 41% | 0.83–1.00 |

\* Not gated. At 72 h ahead about 40% of Taos forecasts, in either version, put their highest flow at the very start or end of the window, i.e. they do not see that crest coming. Many of these are winter ice spikes and storms that have not formed yet, so the median mostly measures noise. CI gates every river at 12, 24 and 48 h.

By crest type:

| River | Rain-driven crests: timing (12 / 24 / 48 h), ratio | Other crests: timing, ratio |
|---|---|---|
| Royal Gorge | 24 crests: 5.5 / 5.5 / 6 h, 0.83–0.85 | 11 crests: 11 / 17 / 21 h, 0.72–0.75 |
| Dolores | 37 crests: 7 / 9 / 9 h, late, 0.79–0.86 | 23 crests: 6 / 6 / 6 h, 0.92–0.94 |
| Lower Taos Box | 11 crests: 10 / 23 / 35 h, 0.63–0.66 | 43 crests: 7 / 12 / 23 h, 0.86–1.00 |

The forecast charts quote these numbers (from `openFlowML/data/routing_skill.json`). When 10 mm or more of rain falls on the ungauged area around the forecast time, they quote the rain-driven ones and point to the NWS forecast where one is shown.

### What changed because of it

The first backtest exposed three problems, now fixed.

| Change | Effect (median crest timing error) |
|---|---|
| Linear-reservoir K capped at 12 h, lags searched in 2-h steps, bias toward the latest reading decays over 24 h (was 48 h). The wider grid let the 28-day calibration pick slow reservoirs that put crests late. | Dolores 12 h ahead 11 → 6 h, 24 h ahead 11 → 8.5 h; Taos 48 h ahead 31 → 25.5 h; Royal Gorge unchanged (7 h) |
| Taos routed from the Rio Grande near Cerro, NM (NWS CERN5, USGS 08263500) instead of Lobatos, plus the Rio Pueblo de Taos and Red River gauges (see below). | Taos 12 h ahead 11 → 8 h (within ±6 h: 28% → 41%), 24 h ahead 21.5 → 14 h (13% → 28%), crest ratio 0.91 → 1.00. 48 h ahead unchanged (25.5 h). Worse at 72 h (32.5 → 41 h, crest ratio 0.79 → 0.69) and on rain-driven crest height (0.76 → 0.65): the tributaries' storm runoff after their latest reading is not forecast. |
| Colorado DWR hourly timestamps read as local clock time (MST/MDT), not MST. | Removes a 1-h error for half the year at every DWR gauge |

Tried and not adopted: weighting crest timing in the calibration score (no gain at 24 h+); fitting travel times on each river's past floods instead of the last 28 days (worse out of sample); a faster rain response (Royal Gorge no change; Dolores storm crests 3 h better but spring crests 1 h worse).

### Taos: why Cerro

Measured on the hourly records 2015–2026:

| Reach | Crest travel time, median (25–75%) | Notes |
|---|---|---|
| Lobatos → Taos Junction Bridge | 29–35 h (noisy) | Lobatos (DWR) is ice-affected in winter and reads up to 2× Cerro in Jan |
| Lobatos → Cerro | 20 h (16–31 h) | Refitted each run: 2 h + 12-h reservoir in the Oct 2026 flood |
| Cerro → Taos Junction Bridge | 11 h (9–12.5 h); 7 h above 1,000 cfs, 4 h above 2,000 cfs | Cerro's rise is most of the Taos rise in nearly every flood |
| Rio Pueblo de Taos (Los Cordovas) → TJB | 2 h | Monsoon storms: e.g. Aug 23 2022, 1,000 cfs of a 1,190-cfs crest |
| Red River (fish hatchery) → TJB | 5 h | Oct 16 2025: 530 cfs of the rise |

Cerro + Red River + Rio Pueblo de Taos + Rio Hondo carry about 92% of the flow at Taos Junction Bridge; the rest is springs in the gorge. The ML model keeps the daily Lobatos anchor it was trained on; only the hourly routing uses Cerro.

### Limits

- **Rain storms**: rain input is daily (gridMET, then Stage IV), spread evenly over each day. The rain term cannot place a storm's runoff better than about ±12 h, and it under-calls storm crests by 15–25%. During a storm, prefer the NWS forecast where NWS forecasts the gauge, because NWS models storm runoff hour by hour.
- **Upstream forecasts**: the backtest gives upstream gauges a perfect forecast. Live, beyond an upstream gauge's travel time, the routing depends on the NWS RFC forecast at that gauge, so errors there carry straight through.
- **Crest height**: every routed river under-calls crests by 10–25% (crest ratio below 1). Treat the routing peak as a floor in a flood.

## Forecast archive: routing vs NWS vs ML

NWS keeps no archive of its past river forecasts. So the daily **Flow Forecast** workflow commits every forecast it makes (CSV + JSON: routing hourly, NWS official hourly where NWS forecasts the gauge, ML daily) to the `forecast-archive` branch under `archive/YYYY-MM-DD/`. It then runs `score_archive.py` on the last 60 days and posts to the job summary:

- per source, river and lead day: the median absolute % error of the daily mean flow;
- per crest observed inside a forecast's window: timing error (hourly sources) and crest ratio.

Run it locally with `python openFlowML/score_archive.py --archive <checkout of forecast-archive>`. It needs a few weeks of archive before it says anything.

## Which forecast to trust, per river

| River | Forecast point | Use |
|---|---|---|
| Lower Taos Box | USGS 08276500 / NWS TJBN5, Rio Grande below Taos Junction Bridge | Routing for timing: Cerro sees a crest about 11 h ahead (less in big floods); beyond that it follows Lobatos and the Alamosa/Mogote RFC forecasts. NWS (WGRFC) forecasts TJBN5 only during floods. When it does, compare; in rain storms prefer it. |
| Royal Gorge | CO DWR ARKCANCO = USGS 07096000 = NWS CNCC2, Arkansas River at Cañon City (downtown, below the take-out), **year-round** | NWS publishes no forecast for CNCC2, so routing from Wellsville is the only hourly forecast. Its timing is good to about ±7 h, but its crests run about 20% low. The Parkdale gauge at the put-in (PDLC2) reports only in season and is shown for context when it does. |
| Dolores, Gateway–Dewey | USGS 09180000 / NWS DOLU1, Dolores River near Cisco | NWS (CBRFC) forecasts DOLU1 routinely. Routing is good for snowmelt and dam-release crests (±6 h). In rain storms it runs late (+7–9 h) and low, so use NWS. |
| Conejos River | DWR CONMOGCO / NWS MOGC2 | No upstream gauges: NWS RFC forecast, then recession; ML for the week's trend. |
