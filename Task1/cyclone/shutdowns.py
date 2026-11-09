"""Shutdown / idle detection on the inlet gas temperature.

Inlet temperature is strongly bimodal (~0-100 C when the kiln/preheater is off,
830-990 C when running), so a threshold is the honest model here. The threshold
comes from the data (Otsu split of the histogram) rather than being hand-picked,
and hysteresis + duration rules stop the slow ramps from being chopped into many
short events.
"""

import numpy as np
import pandas as pd

TEMP = "Cyclone_Inlet_Gas_Temp"
HYSTERESIS_C = 100.0       # must climb this far above the off-threshold to count as running again
MIN_EVENT = pd.Timedelta("30min")
MERGE_GAP = pd.Timedelta("1h")   # a restart shorter than this is treated as part of the same stop


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Value that best separates the two modes of x (maximises between-class variance)."""
    x = x[~np.isnan(x)]
    hist, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _hysteresis(temp: np.ndarray, off_below: float, on_above: float) -> np.ndarray:
    """Boolean 'off' mask. Missing samples inherit the previous state."""
    off = np.zeros(len(temp), dtype=bool)
    state = bool(temp[~np.isnan(temp)][0] < off_below)
    for i, v in enumerate(temp):
        if not np.isnan(v):
            if state and v > on_above:
                state = False
            elif not state and v < off_below:
                state = True
        off[i] = state
    return off


def _runs(mask: pd.Series) -> pd.DataFrame:
    """Contiguous True runs as (start, end) where end is the last True timestamp + 1 step."""
    step = mask.index[1] - mask.index[0]
    edges = mask.astype(int).diff().fillna(int(mask.iloc[0]))
    starts = mask.index[edges == 1]
    ends = mask.index[edges == -1]
    if mask.iloc[-1]:
        ends = ends.append(pd.DatetimeIndex([mask.index[-1] + step]))
    return pd.DataFrame({"start": starts, "end": ends})


def detect(df: pd.DataFrame, threshold: float | None = None) -> tuple[pd.DataFrame, pd.Series, float]:
    """Return (shutdown events, per-timestamp off mask, threshold used)."""
    temp = df[TEMP]
    if threshold is None:
        threshold = otsu_threshold(temp.to_numpy())
    off = pd.Series(_hysteresis(temp.to_numpy(), threshold, threshold + HYSTERESIS_C), index=df.index)

    ev = _runs(off)
    # merge stops separated by a very short restart
    gap = ev["start"] - ev["end"].shift()
    group = (gap.isna() | (gap >= MERGE_GAP)).cumsum()
    ev = ev.groupby(group).agg(start=("start", "first"), end=("end", "last")).reset_index(drop=True)
    ev = ev[ev["end"] - ev["start"] >= MIN_EVENT].reset_index(drop=True)

    # rebuild the mask from the final events so both views agree
    off = pd.Series(False, index=df.index)
    for s, e in zip(ev["start"], ev["end"]):
        off.loc[s : e - pd.Timedelta("5min")] = True

    ev["duration_h"] = (ev["end"] - ev["start"]).dt.total_seconds() / 3600
    # data outages inside a stop are reported, not silently counted as running/off evidence
    ev["missing_pct"] = [
        round(100 * df.loc[s : e - pd.Timedelta("5min"), TEMP].isna().mean(), 1)
        for s, e in zip(ev["start"], ev["end"])
    ]
    ev.insert(0, "event_id", range(1, len(ev) + 1))
    assert (ev["start"].iloc[1:].to_numpy() >= ev["end"].iloc[:-1].to_numpy()).all(), "overlapping events"
    return ev, off, threshold


def summary(ev: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Downtime per calendar year: events, hours, % of the year's observed span."""
    rows = []
    for year, g in ev.groupby(ev["start"].dt.year):
        span_h = (index[index.year == year].size * 5) / 60
        rows.append({
            "year": year,
            "events": len(g),
            "downtime_h": round(g["duration_h"].sum(), 1),
            "downtime_pct": round(100 * g["duration_h"].sum() / span_h, 1),
            "median_event_h": round(g["duration_h"].median(), 1),
            "longest_event_h": round(g["duration_h"].max(), 1),
        })
    total = {
        "year": "all", "events": len(ev),
        "downtime_h": round(ev["duration_h"].sum(), 1),
        "downtime_pct": round(100 * ev["duration_h"].sum() / (index.size * 5 / 60), 1),
        "median_event_h": round(ev["duration_h"].median(), 1),
        "longest_event_h": round(ev["duration_h"].max(), 1),
    }
    return pd.DataFrame(rows + [total])


def sensitivity(df: pd.DataFrame, base: float, deltas=(-150, -100, -50, 0, 50, 100)) -> pd.DataFrame:
    """How much do the headline numbers move if the threshold is shifted?"""
    out = []
    for d in deltas:
        ev, _, _ = detect(df, base + d)
        out.append({"threshold_c": round(base + d), "events": len(ev),
                    "downtime_h": round(ev["duration_h"].sum(), 1)})
    return pd.DataFrame(out)
