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

No `python3.12` on the machine? Run `uv venv --python 3.12 --seed .venv` instead (uv downloads Python 3.12), then the same `pip install` line. The pins need Python 3.12.

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
│   ├── fan_stop_precursors.csv  every fan stop: which warning signals fired in the 15-120 min before
│   ├── precursor_lift.csv       how much more often each signal fires before a fan stop than at random
│   ├── alert_rules.csv          the alert rules scored: firings, per running week, precision, stops caught
│   ├── fan_off_escalation.csv   once the fan is off, how often it becomes a shutdown (by minutes off)
│   ├── shutdown_sensitivity.csv events and downtime for 33 settings of the shutdown rules
│   ├── data_quality.csv
│   ├── state_episodes.csv
│   ├── forecast_metrics.csv     hourly origins, all three models
│   └── forecast_metrics_dense.csv  every 5-min origin, two test years, with bootstrap intervals
├── plots/                       21 figures, one subfolder per section of the brief (index below)
├── cyclone/                     the code the notebook calls, one module per step
└── data/data.xlsx               the raw export
```

| module | what it does |
|---|---|
| `cyclone/data.py` | loading, status-code handling, physical validity rules, strict 5-min grid, short-gap interpolation |
| `cyclone/shutdowns.py` | Otsu threshold + hysteresis + merge/min-length rules, sensitivity of all four settings, and the fan-off layer (fan stopped while hot) |
| `cyclone/states.py` | features, RobustScaler → PCA → KMeans, rule-based state naming, minimum-dwell smoothing, episode stats, stability checks |
| `cyclone/anomalies.py` | per-state Isolation Forest + robust-z attribution, fan-trip events, event building |
| `cyclone/precursors.py` | what happens before each fan stop, lift against random times, alert rules with measured precision |
| `cyclone/forecasting.py` | persistence / ARIMA / gradient boosting, rolling-origin evaluation by regime (hourly, and every 5 min on two test years) |
| `cyclone/plots.py` | all figures |

### Output CSVs

| file | one row per | key columns |
|---|---|---|
| **`shutdown_periods.csv`** | shutdown (39) | start, end, duration_h, missing_pct |
| **`anomalous_periods.csv`** | anomalous event (411) | start, end, duration_min, detector (`isolation_forest` / `transient_fan_trip` / `fan_trip_extended`), state, top_variables (tag ± robust σ), peak_time, minutes_since_restart, minutes_to_next_shutdown |
| **`clusters_summary.csv`** | operating state (4) | description, mean / std / p05 / p50 / p95 of every tag and derived feature, share of running time, episode count and duration stats |
| **`forecasts.csv`** | (origin, horizon): 5,262 hourly origins × 12 steps | target_time, regime, y_true, pred_persistence, pred_arima, pred_gbm |
| `outputs/supporting/fan_off_periods.csv` | fan-off period while hot (229) | start, end, duration_min, kind, inlet temperature before / minimum / dip |
| `outputs/supporting/fan_stop_precursors.csv` | fan stop (39 shutdowns + 157 trips) | onset, target, clean_lookback, minutes before the stop that each signal last fired |
| `outputs/supporting/precursor_lift.csv` | (window, scope, target, signal) | events, events_with_precursor, % of events, % of random times, lift |
| `outputs/supporting/alert_rules.csv` | alert rule | trigger, lead, firings, per_running_week, precision_pct, stops caught |
| `outputs/supporting/fan_off_escalation.csv` | minutes into a fan-off | still off, become shutdown, precision with and without the < 700 °C condition |
| `outputs/supporting/shutdown_sensitivity.csv` | setting (threshold, hysteresis, merge, minimum length) | events, downtime_h |
| `outputs/supporting/forecast_metrics.csv` | (regime, model) | n, RMSE, MAE |
| `outputs/supporting/forecast_metrics_dense.csv` | (test year, regime) | origins, separate episodes, persistence and GBM RMSE / MAE, GBM − persistence with 95% interval |
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
| `2_shutdowns/04_shutdown_sensitivity.png` | events and downtime for 27 combinations of hysteresis, merge window and minimum length | 2. Shutdowns |
| `3_states/05_states_scatter.png` | the four states in temperature-vs-draft space and in across-cyclone differentials (ΔT vs Δdraft) | 3. States |
| `3_states/05_state_mix_monthly.png` | share of running time in each state, month by month | 3. States |
| `3_states/06_temp_drop_drift.png` | inlet − outlet temperature over time: the thermocouple drift behind the "Degraded" state | 3. States |
| `4_anomalies/07_anomaly_timeline.png` | every anomalous event on the timeline, by detector | 4. Anomalies |
| `4_anomalies/08_rca_20170614_1130.png` | case A: transient fan trip with recovery | 4. Root cause |
| `4_anomalies/08_rca_20171227_1745.png` | case B: repeated draft disturbances and temperature overshoot | 4. Root cause |
| `4_anomalies/08_rca_20170909_0450.png` | case C: extended fan trip, 8 h with the fan off | 4. Root cause |
| `4_anomalies/08_rca_20180621_1530.png` | case D: start-up idle (a former false positive) and an intermittent outlet thermocouple | 4. Root cause |
| `4_anomalies/08_rca_drift_2019.png` | case E: slow drift of the outlet gas thermocouple, Feb–Apr 2019 | 4. Root cause |
| `4_anomalies/10_precursors.png` | share of fan stops with each warning signal 15–120 min before, against random running times | 4. Precursors |
| `5_forecast/09_fc_horizon_steady.png`, `5_forecast/09_fc_horizon_disturbed.png` | forecast error vs horizon for each model, steady and disturbed running | 5. Forecast |
| `5_forecast/09_fc_example.png` | 30-minute-ahead forecasts from all three models against the actual signal through a disturbance (19 Jul 2020) | 5. Forecast |

## Approach in one table

| step | choice | why |
|---|---|---|
| cleaning | remove only values that can't be real (status codes, range caps, dropouts, stale values); interpolate gaps ≤ 30 min only | statistical outliers are what step 4 needs to find; long gaps would be invented data |
| shutdowns | Otsu threshold on inlet gas temp (477 °C), hysteresis, 1 h merge, 30 min minimum | the signal is cleanly bimodal, so a data-derived threshold beats a guessed one. ±50 °C changes downtime by < 2%; all 27 combinations of hysteresis (+50/100/150 °C), merge (30 min/1 h/2 h) and minimum length (15/30/60 min) give 36–43 events and downtime within 1% |
| idle (fan off) | median draft > −60 Pa while hot, ≥ 10 min: classified as coast-down into a stop, start-up idle, or fan trip | no gas is pulled through the cyclone, so it's neither running nor a cold stop; the brief asks for idle periods to be excluded before clustering |
| states | 4 KMeans clusters on fan-on running data, scaled + PCA features including across-cyclone ΔT and Δdraft | running data is a continuum (HDBSCAN finds 100% noise), so KMeans cuts it into bands with distinct physical signatures. Silhouette's k = 3 is one 93% blob |
| anomalies | per-state Isolation Forest (pooled 99.5th percentile) + every fan trip as an incident | contextual by design; trips are found by the fan-off layer, so they can't hide as a "normal" cluster |
| precursors and alerts | signals in the 15–120 min before each fan stop vs random running times; candidate alerts scored per firing | lift alone flatters rare signals; precision per firing and alerts per week are what an operator lives with |
| forecast | persistence vs ARIMA(2,1,2) vs direct gradient boosting, hourly rolling origins on Jan–Aug 2020; persistence and GBM again at every 5-min origin on 2020 and on a 2019 fold | errors are reported per regime, because the overall average mostly reflects how much of the test set is shut down; the 2019 fold adds the stops 2020 lacks |

**Are the states stable?** They're checked three ways: refitting on bootstrap resamples (ARI **0.98**), fitting on 2017–18 and labelling 2019–20 (ARI **0.75**, because the load mix really did shift), and month-to-month churn in the state mix (0.18). Adding baseline-relative features (each tag minus its trailing 7-day running median) makes the monthly mix 16% smoother. But the states become slightly *less* reproducible (ARI 0.97 and 0.73), so the absolute features are kept. The full comparison is in section 3 of the notebook.

## Headline results

- **39 shutdowns, 6,286 h down (19.9%).** Durations form two groups: 26 stops under 2.5 days and 13 outages of 4–42 days. Each of 2017–2019 has a 3–6 week stop starting in late September, and 23 Mar – 1 May 2020 matches the COVID-19 lockdown.
- **Four running states:** High load (44% of fan-on running time), Normal load (50%), Degraded: outlet thermocouple (4.1%), Unstable draft (2.4%).
- **411 anomalous events:** 254 from the contextual Isolation Forest, 91 transient fan trips and 66 extended fan trips.
- **Forecast, RMSE / MAE in °C** (averaged over all 12 horizons, hourly origins, test Jan–Aug 2020):

  | regime | share of origins | persistence | ARIMA | GBM |
  |---|---|---|---|---|
  | steady running | 77% | 17.3 / 12.7 | 13.8 / 10.2 | **13.6 / 10.0** |
  | disturbed | 4.5% | 55.4 / 38.7 | 52.2 / 35.3 | **41.7 / 27.9** |
  | shut down | 18% | **5.0 / 0.9** | 6.2 / 1.3 | 5.6 / 2.5 |
  | whole test set | 100% | 19.4 / 11.8 | 16.8 / 9.8 | **15.1 / 9.5** |

  The 4.5% of disturbed windows carry about 35% of all forecast error. With an origin at every 5-min step, gradient boosting beats persistence in disturbed running on both test years (−12.5 °C RMSE over 101 episodes in 2020, −10.2 °C over 487 in 2019). It also wins at the 23 stops and restarts of 2019 (−11.7 °C; 95% day-block bootstrap interval −19.7 to −2.4). 2020 has only 6 such events, too few on their own.

## Insights and recommendations

Each insight is followed by the action it supports and the evidence behind that action (notebook section 6; numbers from `outputs/supporting/`).

**1. ID-fan trips are the main operational problem, and every shutdown starts with the fan stopping.** There were 157 fan trips: 91 transient (back within an hour, ~100 °C dip, about one every 11 running days) and 66 extended. In all 39 shutdowns the fan stops while the gas is still at ~890 °C, a median ~7 h before the line is cold.
→ **Rule 1, fan-off alarm:** median of the three drafts above −60 Pa for two samples while inlet gas is above 477 °C, suppressed for 1 h after a restart. *Evidence:* 196 firings in 147 running weeks (**1.3 per week**). In **93%** of them inlet gas then fell by more than 30 °C, so the gas flow really stopped (`alert_rules.csv`).
→ Pull ID-fan motor current and interlock logs for the 157 trip timestamps. One common root cause would be the highest-value fix in this dataset.

**2. Nothing in these six tags warns of a fan trip; shutdowns have a visible run-up.** Looking 15–120 min before each fan stop (look-back of uninterrupted running only), the most common signal before a trip, draft weakening, appears before 26 of 125 trips (2.4× the random rate). Before a shutdown, draft weakening shows in 15 of 31 and an IF flag in 11 of 31 (5.6× and 13.6×): a ramp-down. As alerts at a 15–120 min lead, though, no signal is right more than 7% of the time.
→ Don't build an early-warning trip alarm from these tags; the cause sits upstream of them (motor, interlocks).
→ **Rule 2, escalate:** if Rule 1 is still active after 60 min and inlet gas is below 700 °C, treat it as becoming a shutdown. *Evidence:* **59%** of firings end in one (20 of 34, **0.23 per running week**), with a median **4.7 h** of warning before the line is cold. Without the temperature condition, only 37% of fans still off at 60 min become a shutdown (`fan_off_escalation.csv`).

**3. Part of what looks like process behaviour is instrument behaviour.** The outlet gas thermocouple drifts for weeks (Mar–May 2018, Feb–Mar 2019) and reads hotter than the inlet for most of mid-2019. That is the whole "Degraded" state (4.1% of running time). The material temperature transmitter has nearly all the `Unit Down` codes and ~1,150 dropouts while hot.
→ Alert when the daily median of inlet − outlet gas temperature leaves roughly −10 … +60 °C, and check every tag for stale values (identical for 20 min while running). Calibrate both instruments at each planned stop. *Evidence:* case E and the monthly state mix; tag faults carry no warning of fan stops (1.2–1.3×), so this is a data-quality action, not a process alarm.

**4. Contextual anomalies cluster right after restarts.** The ~16 h of fan-on running within 2 h of a restart hold 6 of the 254 IF events: ~380 per 1,000 h, against ~10 elsewhere. That is a small sample, but a large gap.
→ Use wider limits, or a start-up–specific model, for the first ~2 h after a restart. Treat IF events as an engineering review list, not an operator alarm. *Evidence:* only 6% of IF flags are followed by a fan stop within 2 h.

**5. Load has shifted upward and stops are seasonal, so models go stale.** 2017 ran 88% in the Normal-load band; Jan–Jul 2020 ran 76–95% High load each month. The longest stop of each year starts in late September.
→ Retrain the state, anomaly and forecast models monthly on a rolling window, and run the forecast by regime (persistence or a cooling curve when off, gradient boosting while running). *Evidence:* a model fitted on 2017–18 labels 2019–20 with ARI 0.75 (0.98 on resamples of the same period). Gradient boosting's gain is only clear when the process is disturbed (forecast table above).

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
- The 2020 test period has only 6 stops and restarts. The transition comparison rests on the 2019 fold (23 events), with the model trained on 2017–2018 only.
- The precursor analysis has small samples: 31 shutdowns and 125 trips have an uninterrupted look-back. The alert precisions are measured on the same 3.6 years the rules were described on, so they should be re-measured on new data before use.
- The anomaly threshold (99.5th percentile) sets how many events are reported. It should be tuned with operators against the alarm load they can handle.
