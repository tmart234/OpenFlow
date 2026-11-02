### DATA MODULES

Each data module is designed to be a stand-alone module which can run on it's own given a set of input args

### Upstream gauges (`get_upstream.py`, `upstream_gauges.json`, `nwps.py`)

`upstream_gauges.json` lists, per target station, the upstream gauges whose flow arrives there later (with travel-time lags and weights). `get_upstream.get_upstream(site_id, start, end)` returns the routed daily `upstream_flow` feature used in training. `nwps.py` is a small client for the NOAA NWPS API (recent observations and the current RFC forecast at any NWS gauge id). The upstream-routing forecast CLI and the ML inference path both use it.

To add a station: add an entry keyed by its OpenFlow site id with the target's `nws_lid`, a `timezone`, `forecast_days` (how far ahead the RFC forecasts the upstream gauges, usually 5), and one record per upstream gauge with:
- `site_id`: the training history source (`DWR:<abbrev>` or `USGS:<id>`);
- `nws_lid`;
- `lag_days` and `weight` for the daily ML feature;
- `lag_hours` and `k_hours`: hourly routing defaults, used only when the recent calibration window is too quiet to fit them.

Then run `python openFlowML/flow_forecast.py --site <alias>` to check it.
