# Task 1 — Cyclone sensor analysis

Analysis of 3.6 years of 5-minute cyclone data (2017-01-01 → 2020-08-07, six tags). The work covers cleaning, shutdown detection, operating states, contextual anomalies with root-cause hypotheses, and a 1-hour forecast of inlet gas temperature.

## How to run

From the repository root:

```bash
uv sync                                   # Python 3.12 env from pyproject.toml / uv.lock
cp /path/to/data.xlsx given/data.xlsx     # the raw export is not committed
cd Task1
uv run jupyter nbconvert --to notebook --execute --inplace task1_analysis.ipynb
# or open it: uv run jupyter lab task1_analysis.ipynb, then Restart & Run All
```

A full run takes about 2–3 minutes. The first run parses the xlsx (~15 s) and caches it as `cache/raw.parquet`. Every CSV in this folder and every PNG in `plots/` is regenerated on each run, and all randomness is seeded.

## What's here

| path | contents |
|---|---|
| `task1_analysis.ipynb` | the narrative: every decision, with the evidence for it, executed with outputs |
| `cyclone/data.py` | loading, status-code handling, physical validity rules, strict 5-min grid, short-gap interpolation |
| `cyclone/shutdowns.py` | Otsu threshold + hysteresis + merge/min-length rules, yearly summary, threshold sensitivity |
| `cyclone/states.py` | features, RobustScaler → PCA → KMeans, rule-based state naming, minimum-dwell smoothing, episode stats |
| `cyclone/anomalies.py` | per-state Isolation Forest + robust-z attribution, fan-trip rule, event building |
| `cyclone/forecasting.py` | persistence / ARIMA / gradient boosting, rolling-origin evaluation by regime |
| `cyclone/plots.py` | all figures |

### Output CSVs (in this folder)

The first four are the required deliverables; the last three are supporting detail.

| file | one row per | key columns |
|---|---|---|
| `shutdown_periods.csv` | shutdown | start, end, duration_h, missing_pct |
| `anomalous_periods.csv` | anomalous event | start, end, duration_min, detector, state, top_variables (tag ± robust σ), minutes_since_restart, minutes_to_next_shutdown |
| `clusters_summary.csv` | operating state | description, mean / std / p05 / p50 / p95 of every tag and derived feature, share of running time, episode count and duration stats |
| `forecasts.csv` | (origin, horizon) | target_time, regime, y_true, pred_persistence, pred_arima, pred_gbm |
| `forecast_metrics.csv` | (subset, model) | RMSE, MAE |
| `data_quality.csv` | sensor | counts of each status code, invalid, dropout, stale, interpolated and still-missing samples |
| `state_episodes.csv` | state episode | state, start, end, duration_h |

### Plots (`plots/`)

They are numbered in notebook order: `01`–`03` EDA, `04` shutdowns, `05`–`06` states and the thermocouple drift, `07`–`08` anomalies and the five root-cause cases, `09` forecasting.

## Approach in one table

| step | choice | why |
|---|---|---|
| cleaning | remove only values that can't be real (status codes, range caps, dropouts, stale values); interpolate gaps ≤ 30 min only | statistical outliers are what step 4 needs to find; long gaps would be invented data |
| shutdowns | Otsu threshold on inlet gas temp (477 °C), hysteresis, 1 h merge, 30 min minimum | the signal is cleanly bimodal; a data-derived threshold beats a guessed one, and ±50 °C changes downtime by < 2% |
| states | 6 KMeans clusters on scaled + PCA features, including across-cyclone ΔT and Δdraft | running data is a continuum (HDBSCAN finds 100% noise), so KMeans cuts it into bands with distinct physical signatures. Silhouette's k = 3 is one 96% blob |
| anomalies | per-state Isolation Forest (pooled 99.5th percentile) + fan-trip rule | contextual by design. The rule covers the blind spot where a frequent upset becomes its own "normal" cluster |
| forecast | persistence vs ARIMA(2,1,2) vs direct gradient boosting, hourly rolling origins on Jan–Aug 2020 | errors are reported per regime, because the overall average mostly reflects how much of the test set is shut down |

## Headline results

- **39 shutdowns, 6,286 h down (19.9%).** Durations form two groups: 26 stops under 2.5 days and 13 outages of 4–42 days. Each of 2017–2019 has a 3–6 week stop starting in late September, and 23 Mar – 1 May 2020 matches the COVID-19 lockdown.
- **Six operating states:** High load, Normal load, Unstable draft, Degraded (outlet thermocouple), Trip cool-down (fan off), Heat-up / warm idle (fan off).
- **ID-fan trips are the dominant incident.** 123 trips recovered (about one every 8.4 running days, ~110 °C dip each) and 62 more where the fan stayed off. 23 of the 39 shutdowns follow an extended fan trip within 12 h.
- **The outlet gas thermocouple drifts for weeks and reads hotter than the inlet for most of mid-2019.** The material temperature transmitter is the least reliable tag.
- **Forecast, RMSE / MAE in °C** (averaged over all 12 horizons, test Jan–Aug 2020):

  | regime | share of origins | persistence | ARIMA | GBM |
  |---|---|---|---|---|
  | steady running | 77% | 17.3 / 12.7 | 13.8 / 10.2 | **13.6 / 10.0** |
  | disturbed | 4.5% | 55.4 / 38.7 | 52.2 / 35.3 | **41.7 / 27.9** |
  | shut down | 18% | **5.0 / 0.9** | 6.2 / 1.3 | 5.6 / 2.5 |
  | whole test set | 100% | 19.4 / 11.8 | 16.8 / 9.8 | **15.1 / 9.5** |

  When the cyclone is shut down, persistence wins because the temperature is flat. In production the forecast would be gated by regime.

  The 4.5% of disturbed windows carry about 35% of all forecast error.

## Limitations

- Root causes are hypotheses from 5-minute averages. Confirming them needs fan current, damper positions and operator or maintenance logs, none of which are in the data.
- State names come from cluster medians and physics, not from labelled ground truth.
- The test period holds only 12 hourly origins that span a stop or restart, too few to rank models on transitions.
- The anomaly threshold (99.5th percentile) sets how many events are reported. It should be tuned with operators against the alarm load they can handle.
