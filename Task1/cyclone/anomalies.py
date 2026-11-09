"""State-conditional anomaly detection and event building.

Two detectors feed one event table:

1. Contextual Isolation Forest (detector = "isolation_forest").
   One IF per operating state, fitted on that state's own feature
   distribution, so a reading is judged against what is normal *for that
   state*. The flag threshold is one percentile of the pooled scores rather
   than a fixed contamination per state: a per-state contamination would force
   every state to have the same anomaly rate and throw away the comparison
   "which state is anomaly-prone?".

2. Upset-state rule (detector = "transient_fan_trip" / "fan_trip_extended").
   Purely contextual detection has a blind spot: an upset that happens often
   enough becomes its own cluster, and is then "normal for its state". Here
   that is the fan-off cool-down state -- the ID fan trips, draft collapses,
   temperatures fall. Every such episode that is not simply the first step of
   a detected shutdown is reported: "transient" when the next episode has the
   fan running again, "extended" when the fan stayed off (the line coasted
   down slowly, so the shutdown threshold is crossed only hours later or not
   at all before a restart).

Attribution for both: robust z-scores (median / MAD) per tag, per state. The
detector says "this is unusual"; the z-scores say which tags made it unusual
and in which direction, in units an operator can read.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .data import SHORT
from .states import episodes

ATTRIB_COLS = list(SHORT) + ["temp_drop", "draft_dp"]
FLAG_PCT = 99.5          # pooled percentile of IF score
GAP_STEPS = 3            # flagged points within 15 min belong to the same event
MIN_POINTS = 2           # single-sample blips are dropped unless extreme
EXTREME_PCT = 99.95
TRIP_STATE = "Trip cool-down (fan off)"
LEADS_TO_SHUTDOWN = pd.Timedelta("3h")
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


def transient_trip_events(sc, feats, labels, shutdowns, names) -> pd.DataFrame:
    """Fan-off episodes during running that did not lead into a shutdown.

    Attribution measures the trip samples against the distribution of the state
    the line was in just before the trip, so the implicated tags describe what
    changed relative to normal running.
    """
    trip_id = {v: k for k, v in names.items()}[TRIP_STATE]
    fan_off = {k for k, v in names.items() if "fan off" in v}
    stats = sc.attrs["state_stats"]
    ep = episodes(labels).reset_index(drop=True)
    s0, e0 = shutdowns["start"].to_numpy(), shutdowns["end"].to_numpy()
    zcols = [f"z_{c}" for c in ATTRIB_COLS]
    rows = []
    for j, r in ep.iterrows():
        if r.state != trip_id:
            continue
        nxt = s0[s0 >= r.end.to_datetime64()]
        if len(nxt) and pd.Timestamp(nxt[0]) - r.end <= LEADS_TO_SHUTDOWN:
            continue  # this cool-down is the start of a real stop, already in shutdown_periods
        # recovered = the very next episode (no gap in between) has the fan running again
        after = ep.iloc[j + 1] if j + 1 < len(ep) else None
        recovered = after is not None and not np.isnan(after.prev_state) and after.state not in fan_off
        g = sc.iloc[r.i0 : r.i1]
        ref_state = trip_id if np.isnan(r.prev_state) else int(r.prev_state)
        med, mad = stats[ref_state]
        zref = g.copy()
        zref[zcols] = ((feats.iloc[r.i0 : r.i1][ATTRIB_COLS] - med) / mad).to_numpy()
        detector = "transient_fan_trip" if recovered else "fan_trip_extended"
        rows.append(_event_row(g, detector, trip_id, names, s0, e0, zref=zref))
    return pd.DataFrame(rows)


def build_events(sc, feats, labels, shutdowns, names) -> pd.DataFrame:
    iso, thr = isolation_forest_events(sc, shutdowns, names)
    trips = transient_trip_events(sc, feats, labels, shutdowns, names)
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
    """Contextual-IF anomaly density per state (trips excluded: they *are* a state)."""
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
