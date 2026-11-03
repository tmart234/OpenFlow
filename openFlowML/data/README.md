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

### Basin rain by travel-time zone (`basin_zones.py`, `basin_zones.json`, `build_basin_zones.py`, `get_basin_precip.py`)

Rain that floods a river falls on its **contributing area**, often far from the gauge. A HUC level by itself is the wrong unit:
- HUC2 13 covers the whole Rio Grande to the Gulf.
- HUC4 1301 includes the San Luis Valley's Closed Basin, which never drains to the river.
- A gauge's own HUC8 extends downstream of it.

`build_basin_zones.py` delineates each station's contributing area from the USGS NLDI stream network (upstream HUC12s; closed basins excluded automatically). It splits that area by river distance to the gauge:
- `local`: up to 100 river-km;
- `mid`: 100–250 river-km;
- `far`: beyond 250 river-km.

It also marks the **ungauged** area, the part not upstream of the station's `upstream_gauges.json` gauges. It writes simplified zone polygons plus about 36 area-weighted sample points (each with its NWS gridpoint) to `basin_zones.json`. Rebuild with `python openFlowML/data/build_basin_zones.py [--site ID]`; this needs shapely and network access. Training and inference only read the JSON.

`get_basin_precip.py` turns zones into daily rain and snow (water equivalent), split by mean air temperature with one shared rule:
- **gridMET** (daily 4-km grids, about 2 days behind): history and training.
- **NWS Stage IV** (RFC multi-sensor analysis, out within hours): the last 1–2 days. Needs h5py.
- **NWS gridpoint forecast rain** (about 7 days): forecasts.

The ML model gets `rain_/snow_{local,mid,far}`. The routing CLI turns rain on the ungauged area into a calibrated local-inflow term and draws daily zone rain under the hydrograph.
