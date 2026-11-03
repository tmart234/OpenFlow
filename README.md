# OpenFlow
New here? See our wiki 
[wiki](https://github.com/tmart234/OpenFlow/wiki) to learn more


![Diagram 1](https://raw.githubusercontent.com/tmart234/OpenFlow/dev/assets/OpenFlowML_diagram.svg "ML workflow")

- [x] Automated dataset creation based on a USGS and CODWR station
- [x] GitHub actions ML training
- [x] Mobile artifacts published per training run (CoreML + TFLite + manifest) — see [docs/INFERENCE.md](docs/INFERENCE.md)
- [x] Upstream-gauge inputs: the model sees flood waves still upstream (e.g. Colorado's San Luis Valley -> Lower Taos Box) — see `openFlowML/data/upstream_gauges.json`
- [x] Basin rain by travel-time zone: gridMET / NWS Stage IV / NWS forecast rain and snow averaged over the river-distance zones of each gauge's contributing area (not a single point, not a fixed HUC level) — see `openFlowML/data/README.md`
- [x] Forecast verification on every training run: per-station, per-lead-day error in cfs against a "no change" forecast, flood-peak and timing checks, and a quality gate that blocks publishing a worse model — see `openFlowML/evaluate.py` and `quality_report.md` in each release
- [x] Forecast CLI with charts (no trained model needed): `python openFlowML/flow_forecast.py --site taos-box` (also `royal-gorge`, `conejos`, `dolores`; add `--model-dir <release>` to overlay the trained ML model) writes a CSV, JSON and PNG with the peak called out — upstream routing where upstream gauges are registered, else the official NWS forecast for the gauge; also runnable from the **Flow Forecast** GitHub Action
