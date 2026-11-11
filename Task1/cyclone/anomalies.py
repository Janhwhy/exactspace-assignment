"""State-conditional anomaly detection and event building.

Two detectors feed one event table:

1. Contextual Isolation Forest (detector = "isolation_forest").
   One IF per operating state, fitted on that state's own feature
   distribution, so a reading is judged against what is normal *for that
   state*. The flag threshold is one percentile of the pooled scores rather
   than a fixed contamination per state: a per-state contamination would force
   every state to have the same anomaly rate and throw away the comparison
   "which state is anomaly-prone?".

2. Fan-trip rule (detector = "transient_fan_trip" / "fan_trip_extended").
   The ID fan stopping while the line is hot (draft collapses to ~0 Pa) is
   detected by the fan-off layer in shutdowns.py and excluded from the
   running data the states and the Isolation Forests are fitted on. Every
   fan-off period that is not part of a stop or a start-up is reported here
   as an incident: "transient" when the fan was back within 60 min,
   "extended" when it stayed off longer. Without this rule the detector would
   have a blind spot: an upset that happens often enough becomes its own
   cluster and is then "normal for its state".

Attribution for both: robust z-scores (median / MAD) per tag, per state. The
detector says "this is unusual"; the z-scores say which tags made it unusual
and in which direction, in units an operator can read.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .data import SHORT

ATTRIB_COLS = list(SHORT) + ["temp_drop", "draft_dp"]
FLAG_PCT = 99.5          # pooled percentile of IF score
GAP_STEPS = 3            # flagged points within 15 min belong to the same event
MIN_POINTS = 2           # single-sample blips are dropped unless extreme
EXTREME_PCT = 99.95
SEED = 7


def score(feats: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per-sample IF anomaly score and per-tag robust z, conditional on state."""
    out, stats = [], {}
    for s, g in feats.groupby(labels):
        X = RobustScaler().fit_transform(g)
        iso = IsolationForest(n_estimators=200, max_samples=4096, random_state=SEED, n_jobs=-1).fit(X)
        sc = pd.DataFrame(index=g.index)
        sc["state"] = s
        sc["score"] = -iso.score_samples(X)  # higher = more anomalous
        a = g[ATTRIB_COLS]
        med = a.median()
        mad = ((a - med).abs().median() * 1.4826).replace(0, np.nan)
        stats[s] = (med, mad)
        sc[[f"z_{c}" for c in ATTRIB_COLS]] = ((a - med) / mad).to_numpy()
        out.append(sc)
    sc = pd.concat(out).sort_index()
    sc.attrs["state_stats"] = stats
    return sc


def _event_row(g: pd.DataFrame, detector: str, state: int, names: dict,
               sd_starts: np.ndarray, sd_ends: np.ndarray, zref: pd.DataFrame | None = None) -> dict:
    """One output row from the samples of an event. `zref` = samples used for attribution."""
    zcols = [c for c in g.columns if c.startswith("z_")]
    zmean = (g if zref is None else zref)[zcols].mean()
    top = zmean.abs().sort_values(ascending=False).head(3).index
    tags = [f"{SHORT.get(c[2:], c[2:])} {'+' if zmean[c] > 0 else '-'}{abs(zmean[c]):.1f}σ" for c in top]
    start, end = g.index[0], g.index[-1] + pd.Timedelta("5min")
    nxt = sd_starts[sd_starts >= end.to_datetime64()]
    prv = sd_ends[sd_ends <= start.to_datetime64()]
    return {
        "start": start, "end": end,
        "duration_min": int((end - start).total_seconds() / 60),
        "detector": detector,
        "state_id": int(state), "state": names[state],
        "peak_time": g["score"].idxmax(), "peak_score": round(g["score"].max(), 4),
        "top_variables": "; ".join(tags),
        "primary_variable": SHORT.get(top[0][2:], top[0][2:]),
        "minutes_since_restart": round((start - pd.Timestamp(prv[-1])).total_seconds() / 60) if len(prv) else np.nan,
        "minutes_to_next_shutdown": round((pd.Timestamp(nxt[0]) - end).total_seconds() / 60) if len(nxt) else np.nan,
    }


def isolation_forest_events(sc, shutdowns, names) -> tuple[pd.DataFrame, float]:
    thr = np.percentile(sc["score"], FLAG_PCT)
    extreme = np.percentile(sc["score"], EXTREME_PCT)
    flagged = sc[sc["score"] >= thr]
    t = flagged.index.asi8
    eid = np.cumsum(np.r_[True, np.diff(t) > GAP_STEPS * pd.Timedelta("5min").value])
    s0, e0 = shutdowns["start"].to_numpy(), shutdowns["end"].to_numpy()
    rows = [
        _event_row(g, "isolation_forest", g["state"].mode().iloc[0], names, s0, e0)
        for _, g in flagged.groupby(eid)
        if len(g) >= MIN_POINTS or g["score"].max() >= extreme
    ]
    return pd.DataFrame(rows), thr


def fan_trip_events(feats_all: pd.DataFrame, labels: pd.Series, fan: pd.DataFrame,
                    sc: pd.DataFrame, shutdowns: pd.DataFrame, names: dict) -> pd.DataFrame:
    """One row per fan trip (from shutdowns.fan_off_periods).

    Attribution measures the trip samples against the distribution of the state
    the line was in just before the fan stopped, so the implicated tags describe
    what changed relative to normal running.
    """
    stats = sc.attrs["state_stats"]
    s0, e0 = shutdowns["start"].to_numpy(), shutdowns["end"].to_numpy()
    temp = feats_all["Cyclone_Inlet_Gas_Temp"]
    rows = []
    for r in fan[fan["kind"].str.startswith("fan_trip")].itertuples():
        before = labels.loc[:r.start - pd.Timedelta("5min")]
        ref = int(before.iloc[-1]) if len(before) else int(labels.iloc[0])
        med, mad = stats[ref]
        window = feats_all.loc[r.start: r.end - pd.Timedelta("5min")]
        g = pd.DataFrame(((window[ATTRIB_COLS] - med) / mad).to_numpy(),
                         index=window.index, columns=[f"z_{c}" for c in ATTRIB_COLS])
        # no IF score exists for fan-off samples; "peak" is the lowest inlet temperature
        g["score"] = -temp.loc[window.index].fillna(temp.loc[window.index].max())
        row = _event_row(g, "transient_fan_trip" if r.kind == "fan_trip_transient" else "fan_trip_extended",
                         ref, names, s0, e0)
        row["state"] = f"fan off (was {names[ref]})"
        row["peak_score"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_events(sc, feats_all, labels, fan, shutdowns, names) -> pd.DataFrame:
    iso, thr = isolation_forest_events(sc, shutdowns, names)
    trips = fan_trip_events(feats_all, labels, fan, sc, shutdowns, names)
    ev = pd.concat([iso, trips]).sort_values("start").reset_index(drop=True)
    # an IF event that overlaps a trip describes the same incident: keep the trip row, note it
    if len(trips):
        overl = np.zeros(len(ev), dtype=bool)
        for r in trips.itertuples():
            overl |= ((ev["detector"] == "isolation_forest") & (ev["start"] < r.end) & (ev["end"] > r.start)).to_numpy()
        ev = ev[~overl].reset_index(drop=True)
    ev.insert(0, "event_id", range(1, len(ev) + 1))
    ev.attrs["threshold"] = thr
    return ev


def rate_by_state(sc: pd.DataFrame, ev: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Contextual-IF anomaly density per running state (fan trips are counted separately)."""
    thr = ev.attrs["threshold"]
    iso = ev[ev["detector"] == "isolation_forest"]
    g = sc.groupby("state")
    running_h = g.size() * 5 / 60
    out = pd.DataFrame({
        "state": [names[s] for s in running_h.index],
        "running_h": running_h.round(0).to_numpy(),
        "flagged_pct": (100 * g["score"].apply(lambda s: (s >= thr).mean())).round(2).to_numpy(),
        "if_events": [int((iso["state_id"] == s).sum()) for s in running_h.index],
    }, index=running_h.index)
    out["if_events_per_1000h"] = (1000 * out["if_events"] / out["running_h"]).round(1)
    return out
