# OpenFlow
New here? See our wiki 
[wiki](https://github.com/tmart234/OpenFlow/wiki) to learn more


![Diagram 1](https://raw.githubusercontent.com/tmart234/OpenFlow/dev/assets/OpenFlowML_diagram.svg "ML workflow")

- [x] Automated dataset creation based on a USGS and CODWR station
- [x] GitHub actions ML training
- [x] Mobile artifacts published per training run (CoreML + TFLite + manifest) — see [docs/INFERENCE.md](docs/INFERENCE.md)
- [x] Upstream-gauge inputs: the model sees flood waves still upstream (e.g. Colorado's San Luis Valley -> Lower Taos Box) — see `openFlowML/data/upstream_gauges.json`
- [x] Basin rain by travel-time zone: gridMET / NWS Stage IV / NWS forecast rain and snow averaged over the river-distance zones of each gauge's contributing area (not a single point, not a fixed HUC level) — see `openFlowML/data/README.md`
- [x] Forecast CLI with charts (no trained model needed): `python openFlowML/flow_forecast.py --site taos-box` writes a CSV, JSON and PNG with the peak called out; also runnable from the **Flow Forecast** GitHub Action
