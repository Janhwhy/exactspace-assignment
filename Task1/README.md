# Task 1 — Cyclone sensor analysis

This is an analysis of 3.6 years of 5-minute cyclone data (2017-01-01 → 2020-08-07, six tags). It covers cleaning, shutdown detection, operating states, contextual anomalies with root-cause hypotheses, and a 1-hour forecast of inlet gas temperature.

> **What counts as a shutdown.**
> - **Definition:** a shutdown is the line going cold. Inlet gas temperature falls below 477 °C, a threshold found from the data by an Otsu split of its two modes. It counts as running again only above 577 °C (hysteresis), restarts shorter than 1 h are merged, and stops shorter than 30 min are dropped. This gives **39 shutdowns**.
> - **ID-fan trips are not counted as stops.** In a trip the fan stops while the gas stays hot: draft goes to ~0 Pa, and the line is back at temperature within about an hour. They are reported as **incidents** in `anomalous_periods.csv`, and the line never went cold.
> - **Other definitions give more events.** Counting every fan trip with draft ≈ 0 Pa as a stop gives **196 events (5.0×)**. Counting only fan trips of an hour or more gives **105 (2.7×)**. Both numbers are computed in the notebook.

## How to run

The raw data is bundled at `Task1/data/data.xlsx`, so the notebook runs as it is. From the repository root:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: uv sync
cd Task1
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace task1_analysis.ipynb
# or open it: ../.venv/bin/jupyter lab task1_analysis.ipynb, then Restart & Run All
```

To use another copy of the data, set `CYCLONE_DATA=/path/to/data.xlsx`. If the file is missing, the notebook stops with one message naming the path it expected.

A full run takes about 2–3 minutes. The first run parses the xlsx (~15 s) and caches it as `cache/raw.parquet`. Every CSV and every PNG is regenerated on each run, and all randomness is seeded. The notebook is saved with all outputs, so it can be read without running anything.

## What's here

```
Task1/
├── README.md                    this file
├── task1_analysis.ipynb         the analysis, executed, with the reasoning behind every decision
├── shutdown_periods.csv         ┐
├── anomalous_periods.csv        │ the four required outputs
├── clusters_summary.csv         │
├── forecasts.csv                ┘
├── outputs/supporting/          extra detail behind the required outputs
│   ├── fan_off_periods.csv      every fan-off period while hot, classified (trip / coast-down / start-up)
│   ├── data_quality.csv
│   ├── state_episodes.csv
│   └── forecast_metrics.csv
├── plots/                       19 figures, one subfolder per section of the brief (index below)
├── cyclone/                     the code the notebook calls, one module per step
└── data/data.xlsx               the raw export
```

| module | what it does |
|---|---|
| `cyclone/data.py` | loading, status-code handling, physical validity rules, strict 5-min grid, short-gap interpolation |
| `cyclone/shutdowns.py` | Otsu threshold + hysteresis + merge/min-length rules, threshold sensitivity, and the fan-off layer (fan stopped while hot) |
| `cyclone/states.py` | features, RobustScaler → PCA → KMeans, rule-based state naming, minimum-dwell smoothing, episode stats, stability checks |
| `cyclone/anomalies.py` | per-state Isolation Forest + robust-z attribution, fan-trip events, event building |
| `cyclone/forecasting.py` | persistence / ARIMA / gradient boosting, rolling-origin evaluation by regime |
| `cyclone/plots.py` | all figures |

### Output CSVs

| file | one row per | key columns |
|---|---|---|
| **`shutdown_periods.csv`** | shutdown (39) | start, end, duration_h, missing_pct |
| **`anomalous_periods.csv`** | anomalous event (411) | start, end, duration_min, detector (`isolation_forest` / `transient_fan_trip` / `fan_trip_extended`), state, top_variables (tag ± robust σ), peak_time, minutes_since_restart, minutes_to_next_shutdown |
| **`clusters_summary.csv`** | operating state (4) | description, mean / std / p05 / p50 / p95 of every tag and derived feature, share of running time, episode count and duration stats |
| **`forecasts.csv`** | (origin, horizon): 5,262 hourly origins × 12 steps | target_time, regime, y_true, pred_persistence, pred_arima, pred_gbm |
| `outputs/supporting/fan_off_periods.csv` | fan-off period while hot (229) | start, end, duration_min, kind, inlet temperature before / minimum / dip |
| `outputs/supporting/forecast_metrics.csv` | (regime, model) | n, RMSE, MAE |
| `outputs/supporting/data_quality.csv` | sensor | counts of each status code, invalid, dropout, stale, interpolated and still-missing samples |
| `outputs/supporting/state_episodes.csv` | state episode (4,891) | state, start, end, duration_h |

### Plot index (`plots/`)

The file numbers follow the notebook order; the folders follow the brief's sections.

| file | shows | brief section |
|---|---|---|
| `1_eda/01_distributions.png` | value distributions of all six tags (log count): every tag is bimodal, off vs running | 1. EDA |
| `1_eda/02_corr_all.png`, `1_eda/02_corr_running.png` | correlation matrix, all rows and running only | 1. EDA |
| `1_eda/03_week.png`, `1_eda/03_year.png` | representative one-week and one-year slices | 1. EDA |
| `2_shutdowns/04_shutdowns_2018.png` | one full year with every shutdown shaded | 2. Shutdowns |
| `2_shutdowns/04_shutdown_durations.png` | shutdown duration distribution (short stops vs long outages) | 2. Shutdowns |
| `3_states/05_states_scatter.png` | the four states in temperature-vs-draft space and in across-cyclone differentials (ΔT vs Δdraft) | 3. States |
| `3_states/05_state_mix_monthly.png` | share of running time in each state, month by month | 3. States |
| `3_states/06_temp_drop_drift.png` | inlet − outlet temperature over time: the thermocouple drift behind the "Degraded" state | 3. States |
| `4_anomalies/07_anomaly_timeline.png` | every anomalous event on the timeline, by detector | 4. Anomalies |
| `4_anomalies/08_rca_20170614_1130.png` | case A: transient fan trip with recovery | 4. Root cause |
| `4_anomalies/08_rca_20171227_1745.png` | case B: repeated draft disturbances and temperature overshoot | 4. Root cause |
| `4_anomalies/08_rca_20170909_0450.png` | case C: extended fan trip, 8 h with the fan off | 4. Root cause |
| `4_anomalies/08_rca_20180621_1530.png` | case D: start-up idle (a former false positive) and an intermittent outlet thermocouple | 4. Root cause |
| `4_anomalies/08_rca_drift_2019.png` | case E: slow drift of the outlet gas thermocouple, Feb–Apr 2019 | 4. Root cause |
| `5_forecast/09_fc_horizon_steady.png`, `5_forecast/09_fc_horizon_disturbed.png` | forecast error vs horizon for each model, steady and disturbed running | 5. Forecast |
| `5_forecast/09_fc_example.png` | 30-minute-ahead forecasts from all three models against the actual signal through a disturbance (19 Jul 2020) | 5. Forecast |

## Approach in one table

| step | choice | why |
|---|---|---|
| cleaning | remove only values that can't be real (status codes, range caps, dropouts, stale values); interpolate gaps ≤ 30 min only | statistical outliers are what step 4 needs to find; long gaps would be invented data |
| shutdowns | Otsu threshold on inlet gas temp (477 °C), hysteresis, 1 h merge, 30 min minimum | the signal is cleanly bimodal; a data-derived threshold beats a guessed one, and ±50 °C changes downtime by < 2% |
| idle (fan off) | median draft > −60 Pa while hot, ≥ 10 min: classified as coast-down into a stop, start-up idle, or fan trip | no gas is pulled through the cyclone, so it's neither running nor a cold stop; the brief asks for idle periods to be excluded before clustering |
| states | 4 KMeans clusters on fan-on running data, scaled + PCA features including across-cyclone ΔT and Δdraft | running data is a continuum (HDBSCAN finds 100% noise), so KMeans cuts it into bands with distinct physical signatures. Silhouette's k = 3 is one 93% blob |
| anomalies | per-state Isolation Forest (pooled 99.5th percentile) + every fan trip as an incident | contextual by design; trips are found by the fan-off layer, so they can't hide as a "normal" cluster |
| forecast | persistence vs ARIMA(2,1,2) vs direct gradient boosting, hourly rolling origins on Jan–Aug 2020 | errors are reported per regime, because the overall average mostly reflects how much of the test set is shut down |

**Are the states stable?** They're checked three ways: refitting on bootstrap resamples (ARI **0.98**), fitting on 2017–18 and labelling 2019–20 (ARI **0.75**, because the load mix really did shift), and month-to-month churn in the state mix (0.18). Adding baseline-relative features (each tag minus its trailing 7-day running median) makes the monthly mix 16% smoother. But the states become slightly *less* reproducible (ARI 0.97 and 0.73), so the absolute features are kept. The full comparison is in section 3 of the notebook.

## Headline results

- **39 shutdowns, 6,286 h down (19.9%).** Durations form two groups: 26 stops under 2.5 days and 13 outages of 4–42 days. Each of 2017–2019 has a 3–6 week stop starting in late September, and 23 Mar – 1 May 2020 matches the COVID-19 lockdown.
- **Every stop starts with the fan stopping.** In all 39, the fan stops while the gas is still at ~890 °C, and the gas then takes a median ~7 h to cool past the threshold.
- **Four running states:** High load (44% of fan-on running time), Normal load (50%), Degraded: outlet thermocouple (4.1%), Unstable draft (2.4%).
- **ID-fan trips are the dominant incident.** There were 91 transient trips (fan back within an hour, about one every 11 running days, ~100 °C dip) and 66 extended trips (an hour or more, ~200 °C dip).
- **The outlet gas thermocouple drifts for weeks and reads hotter than the inlet for most of mid-2019.** The material temperature transmitter is the least reliable tag.
- **Forecast, RMSE / MAE in °C** (averaged over all 12 horizons, test Jan–Aug 2020):

  | regime | share of origins | persistence | ARIMA | GBM |
  |---|---|---|---|---|
  | steady running | 77% | 17.3 / 12.7 | 13.8 / 10.2 | **13.6 / 10.0** |
  | disturbed | 4.5% | 55.4 / 38.7 | 52.2 / 35.3 | **41.7 / 27.9** |
  | shut down | 18% | **5.0 / 0.9** | 6.2 / 1.3 | 5.6 / 2.5 |
  | whole test set | 100% | 19.4 / 11.8 | 16.8 / 9.8 | **15.1 / 9.5** |

  The 4.5% of disturbed windows carry about 35% of all forecast error. When the cyclone is shut down, persistence wins because the temperature is flat; in production the forecast would be chosen by regime.

## How to validate

Every root cause here is a hypothesis from 5-minute averages. These are the checks that would confirm or refute each one with plant records:

| check against | events | confirms | refutes |
|---|---|---|---|
| **stop / maintenance log** | the 39 rows of `shutdown_periods.csv`, especially the 3–6 week stops each late September | the September stops are logged as planned maintenance, and 23 Mar – 1 May 2020 as the lockdown | logged stops missing from the table, or detected stops the log doesn't know (that would be a threshold problem) |
| **stop log: planned vs tripped** | the 39 `coast_down` rows in `fan_off_periods.csv` (the fan stops a median ~7 h before each stop) | the log shows these as planned fan stops (shutdown procedure), or as trips that ended the run | — (either answer is informative: it says whether "every stop starts with the fan" means trips or procedure) |
| **ID-fan motor current, interlock / trip log** | the 157 `*_fan_trip*` rows in `anomalous_periods.csv` (cases A, C) | fan current drops to zero, or an interlock trip is logged, in the same 5 minutes as the draft collapse | fan current stays normal while draft collapses: then it is a damper or duct event, not the fan |
| **damper position / control-loop trends** | the Unstable-draft episodes in `outputs/supporting/state_episodes.csv` (case B, 27 Dec 2017) | damper or fan-speed output oscillating during the episodes | steady actuator outputs: the oscillation comes from the process side (feed, blockage clearing) |
| **instrument maintenance records** | outlet thermocouple: Mar–May 2018, Feb–Apr 2019, and the step changes around the Apr and Oct 2019 stops (cases D, E) | thermocouple replaced, re-seated or recalibrated at those stops | no intervention logged: then the outlet ΔT shift is a real process change and needs a process explanation |
| **a portable reference measurement** | any current "Degraded" period | a calibrated probe at the outlet reads close to the inlet temperature | the probe agrees with the plant thermocouple |

## Limitations

- Root causes are hypotheses (see "How to validate"). The data has no fan current, damper positions or operator logs.
- State names come from cluster medians and physics, not from labelled ground truth.
- The data can't tell a planned fan stop from a trip; only a stop log can (see above).
- The test period holds only 12 hourly origins that span a stop or restart, too few to rank models on transitions.
- The anomaly threshold (99.5th percentile) sets how many events are reported. It should be tuned with operators against the alarm load they can handle.
