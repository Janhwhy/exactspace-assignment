"""1-hour-ahead (12-step) forecasting of Cyclone_Inlet_Gas_Temp.

Setup
- Chronological split: train on 2017-01 .. 2019-12, test on 2020-01 .. 2020-08.
  The test period contains normal running, short trips and the long
  Mar-May 2020 stop, so it exercises every regime.
- Rolling origins every hour in the test period; each origin forecasts t+1..t+12
  using only data up to t.
- Errors are reported overall *and per regime* (steady running / transition /
  shut down), because an average over everything mostly measures how much of the
  test set happens to be shut down.

Models
- Persistence: y(t+h) = y(t). The baseline every model has to beat.
- ARIMA(2,1,2): one parameter set fitted on the last weeks of training, then
  re-filtered on the trailing day at each origin (parameters fixed, state
  updated). That is how ARIMA would actually run online, and it keeps the
  comparison honest without refitting 5,000 models.
- Gradient boosting (sklearn HistGradientBoosting), direct strategy: one model
  per horizon on lag / rolling / cross-sensor features. Direct rather than
  recursive so errors don't compound over the 12 steps.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

TARGET = "Cyclone_Inlet_Gas_Temp"
H = 12
TRAIN_END = "2019-12-31 23:55"
TEST_START = "2020-01-01 00:00"
ORIGIN_EVERY = 12  # one origin per hour
SEED = 7


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features at each timestamp t (only uses data <= t)."""
    y = df[TARGET]
    f = pd.DataFrame(index=df.index)
    for k in range(12):
        f[f"lag_{k}"] = y.shift(k)
    f["lag_24"] = y.shift(24)
    f["lag_288"] = y.shift(288)   # same time yesterday
    for w in (12, 72):
        f[f"mean_{w}"] = y.rolling(w, min_periods=w // 2).mean()
        f[f"std_{w}"] = y.rolling(w, min_periods=w // 2).std()
    f["slope_6"] = y.diff(6)
    f["min_72"] = y.rolling(72, min_periods=36).min()
    for c in ["Cyclone_Gas_Outlet_Temp", "Cyclone_Material_Temp", "Cyclone_Inlet_Draft",
              "Cyclone_cone_draft", "Cyclone_Outlet_Gas_draft"]:
        f[c] = df[c]
        f[f"{c}_d6"] = df[c].diff(6)
    hour = df.index.hour + df.index.minute / 60
    f["hod_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hour / 24)
    return f


def targets(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    return pd.DataFrame({h: y.shift(-h) for h in range(1, H + 1)}, index=df.index)


DISTURBED_RANGE_C = 100.0


def regime(df: pd.DataFrame, origins: pd.DatetimeIndex, threshold: float) -> pd.Series:
    """Regime of the window [t-1h, t+1h] around each origin.

    shut_down       - below the off-threshold the whole window
    transition      - the window crosses the off-threshold (stop or restart)
    disturbed       - running throughout, but inlet temp swings > 100 C (draft upsets etc.)
    steady_running  - everything else
    """
    y = df[TARGET]
    w = 2 * H + 1
    running = (y > threshold).astype(float).where(y.notna())
    frac = running.rolling(w, center=True, min_periods=H).mean().reindex(origins)
    swing = (y.rolling(w, center=True, min_periods=H).max()
             - y.rolling(w, center=True, min_periods=H).min()).reindex(origins)
    return pd.Series(np.select(
        [frac <= 0, frac < 1, swing > DISTURBED_RANGE_C],
        ["shut_down", "transition", "disturbed"], "steady_running"), index=origins)


def fit_gbm(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    models = {}
    for h in Y.columns:
        ok = Y[h].notna()
        models[h] = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=60,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=SEED,
        ).fit(X[ok], Y.loc[ok, h])
    return models


def predict_arima(y: pd.Series, train_end: str, origins: pd.DatetimeIndex,
                  fit_days: int = 28, window: int = 288) -> pd.DataFrame:
    hist = y.loc[:train_end].iloc[-fit_days * 288:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(hist.to_numpy(), order=(2, 1, 2)).fit()
        vals = y.to_numpy()
        pos = y.index.get_indexer(origins)
        preds = np.full((len(origins), H), np.nan)
        for i, p in enumerate(pos):
            w = vals[p - window + 1 : p + 1]
            if np.isnan(w[-1]):
                continue
            preds[i] = res.apply(w).forecast(H)
    return pd.DataFrame(preds, index=origins, columns=range(1, H + 1))


def _train_rows(X: pd.DataFrame, train_end: str) -> pd.DatetimeIndex:
    """Training timestamps up to train_end, minus rows whose target window crosses it (no leakage)
    and rows without a current value."""
    tr = X.loc[:train_end].index
    return tr[(tr <= pd.Timestamp(train_end) - pd.Timedelta(minutes=5 * H)) & X.loc[tr, "lag_0"].notna()]


def run(df: pd.DataFrame, threshold: float):
    X = make_features(df)
    Y = targets(df)
    tr = _train_rows(X, TRAIN_END)
    origins = X.loc[TEST_START:].index[::ORIGIN_EVERY]
    origins = origins[X.loc[origins, "lag_0"].notna() & Y.loc[origins].notna().all(axis=1)]
    assert origins.min() > tr.max()

    models = fit_gbm(X.loc[tr], Y.loc[tr])
    preds = {
        "persistence": pd.DataFrame({h: X.loc[origins, "lag_0"] for h in range(1, H + 1)}),
        "arima": predict_arima(df[TARGET], TRAIN_END, origins),
        "gbm": pd.DataFrame({h: m.predict(X.loc[origins]) for h, m in models.items()}, index=origins),
    }
    reg = regime(df, origins, threshold)

    long = []
    for h in range(1, H + 1):
        long.append(pd.DataFrame({
            "origin": origins, "horizon": h,
            "target_time": origins + pd.Timedelta(minutes=5 * h),
            "regime": reg.to_numpy(),
            "y_true": Y.loc[origins, h].to_numpy(),
            **{f"pred_{m}": p[h].to_numpy() for m, p in preds.items()},
        }))
    fc = pd.concat(long).sort_values(["origin", "horizon"]).reset_index(drop=True)
    return fc, models


def metrics(fc: pd.DataFrame, by=("regime",)) -> pd.DataFrame:
    rows = []
    models = [c[5:] for c in fc.columns if c.startswith("pred_")]
    groups = [("all", fc)] + [(k, g) for col in by for k, g in fc.groupby(col)]
    for name, g in groups:
        for m in models:
            ok = g[f"pred_{m}"].notna()
            err = g.loc[ok, f"pred_{m}"] - g.loc[ok, "y_true"]
            rows.append({"subset": name, "model": m, "n": int(ok.sum()),
                         "rmse": np.sqrt((err ** 2).mean()), "mae": err.abs().mean()})
    return pd.DataFrame(rows).round(2)


def metrics_by_horizon(fc: pd.DataFrame, subset: str | None = None) -> pd.DataFrame:
    g = fc if subset is None else fc[fc["regime"] == subset]
    models = [c[5:] for c in g.columns if c.startswith("pred_")]
    out = {}
    for m in models:
        e = g[f"pred_{m}"] - g["y_true"]
        out[f"{m}_rmse"] = np.sqrt((e ** 2).groupby(g["horizon"]).mean())
        out[f"{m}_mae"] = e.abs().groupby(g["horizon"]).mean()
    return pd.DataFrame(out).round(2)



# Second fold for the dense evaluation: the 2020 test period has only 6 stops/restarts,
# so the transition comparison also runs on 2019 (11 stops) with a model trained on 2017-2018.
FOLDS = {"test 2020 (train 2017-2019)": (TRAIN_END, TEST_START, None),
         "test 2019 (train 2017-2018)": ("2018-12-31 23:55", "2019-01-01 00:00", "2019-12-31 23:55")}


def run_dense(df: pd.DataFrame, threshold: float, models_2020: dict, n_boot: int = 2000) -> pd.DataFrame:
    """Origin at every 5-min step instead of every hour (persistence and GBM only), on two folds.

    The hourly evaluation has 12 transition origins, too few to rank models. Every-step
    origins give 12x the samples, but they are strongly autocorrelated, so the table
    reports the number of separate regime episodes next to the origin count, and a
    95% interval for the RMSE difference from a block bootstrap over calendar days.
    ARIMA is left out: re-filtering it at ~100k origins costs over half an hour, and it
    was not competitive outside steady running in the hourly evaluation.
    """
    X = make_features(df)
    Y = targets(df)
    rng = np.random.default_rng(SEED)
    rows = []
    for fold, (train_end, test_start, test_end) in FOLDS.items():
        if fold.startswith("test 2020"):
            models = models_2020
        else:
            tr = _train_rows(X, train_end)
            models = fit_gbm(X.loc[tr], Y.loc[tr])
        origins = X.loc[test_start:test_end].index
        origins = origins[X.loc[origins, "lag_0"].notna() & Y.loc[origins].notna().all(axis=1)]
        y = Y.loc[origins].to_numpy()
        preds = {"persistence": np.repeat(X.loc[origins, "lag_0"].to_numpy()[:, None], H, axis=1),
                 "gbm": np.column_stack([models[h].predict(X.loc[origins]) for h in range(1, H + 1)])}
        sq = {m: ((p - y) ** 2).sum(axis=1) for m, p in preds.items()}  # per origin, summed over horizons
        ab = {m: np.abs(p - y).sum(axis=1) for m, p in preds.items()}
        reg = regime(df, origins, threshold).to_numpy()
        contiguous = np.r_[False, np.diff(origins.asi8) == pd.Timedelta("5min").value]
        for name in ["all", "steady_running", "disturbed", "transition", "shut_down"]:
            sel = np.ones(len(origins), bool) if name == "all" else reg == name
            if not sel.any():
                continue
            d = pd.DataFrame({"day": origins.normalize()[sel], **{m: v[sel] for m, v in sq.items()}}).groupby("day")
            sums, cnt = d.sum(), d.size().to_numpy() * H
            idx = rng.integers(0, len(cnt), size=(n_boot, len(cnt)))
            boot = {m: np.sqrt(sums[m].to_numpy()[idx].sum(1) / cnt[idx].sum(1)) for m in preds}
            prev = np.r_[False, sel[:-1]] & contiguous
            row = {"fold": fold, "subset": name, "origins": int(sel.sum()),
                   "separate_episodes": int((sel & ~prev).sum()), "days": len(cnt)}
            for m in preds:
                row[f"{m}_rmse"] = round(float(np.sqrt(sq[m][sel].sum() / (sel.sum() * H))), 2)
                row[f"{m}_mae"] = round(float(ab[m][sel].sum() / (sel.sum() * H)), 2)
            row["gbm_minus_persistence_rmse"] = round(row["gbm_rmse"] - row["persistence_rmse"], 2)
            lo, hi = np.percentile(boot["gbm"] - boot["persistence"], [2.5, 97.5])
            row["ci95_low"], row["ci95_high"] = round(float(lo), 2), round(float(hi), 2)
            rows.append(row)
    return pd.DataFrame(rows)
