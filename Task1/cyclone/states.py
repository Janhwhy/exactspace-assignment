"""Operating-state segmentation of the *running* data.

Pipeline: engineered features -> RobustScaler -> PCA -> KMeans, then a
minimum-dwell filter so a state is an episode rather than per-sample flicker.

Why these choices
- The six tags are highly collinear (|r| ~0.9), so raw tags alone mostly measure
  one thing (how hot the line is). The derived features add the dimensions an
  operator actually cares about: temperature drop across the cyclone, pressure
  drop across it, and how stable / ramping the process is.
- RobustScaler because heavy tails (the anomalies) should not set the scale.
- PCA removes the redundancy so KMeans' Euclidean distance isn't dominated by
  three near-identical temperature tags.
- KMeans over HDBSCAN: ~300k points, a need for every running sample to get a
  state, and centroids that can be read off and named. HDBSCAN is run on a
  sample as a cross-check (see notebook) and labels a large share as noise.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .data import SENSORS

MIN_DWELL_STEPS = 6  # 30 min: shorter episodes are absorbed into the surrounding state
SEED = 7


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SENSORS].copy()
    x["temp_drop"] = df["Cyclone_Inlet_Gas_Temp"] - df["Cyclone_Gas_Outlet_Temp"]
    x["draft_dp"] = df["Cyclone_Inlet_Draft"] - df["Cyclone_Outlet_Gas_draft"]
    x["inlet_temp_std_1h"] = df["Cyclone_Inlet_Gas_Temp"].rolling(12, min_periods=6).std()
    x["inlet_draft_std_1h"] = df["Cyclone_Inlet_Draft"].rolling(12, min_periods=6).std()
    x["inlet_temp_delta_30m"] = df["Cyclone_Inlet_Gas_Temp"].diff(6)
    x["inlet_draft_delta_30m"] = df["Cyclone_Inlet_Draft"].diff(6)
    return x


def active_mask(df: pd.DataFrame, off: pd.Series, feats: pd.DataFrame) -> pd.Series:
    return ~off & feats.notna().all(axis=1)


def choose_k(X: np.ndarray, ks=range(3, 8), n_eval: int = 15_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(n_eval, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X[idx], km.labels_[idx]),
            "davies_bouldin": davies_bouldin_score(X[idx], km.labels_[idx]),
        })
    return pd.DataFrame(rows)


def fit(feats: pd.DataFrame, k: int):
    """Fit scaler+PCA+KMeans. Returns (pipeline, raw labels as Series)."""
    pre = make_pipeline(RobustScaler(), PCA(n_components=0.95, random_state=SEED))
    X = pre.fit_transform(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    return pre, km, pd.Series(km.labels_, index=feats.index)


def name_states(feats: pd.DataFrame, labels: pd.Series) -> dict:
    """Map KMeans ids to names from each cluster's medians, so names don't depend on id order.

    Rules follow the physics: near-zero draft means the ID fan is off (so the
    cyclone isn't really processing gas), a large inlet-outlet temperature drop
    is the instrument-drift signature, high short-term draft variability is an
    unstable process, and the two remaining clusters differ mainly in load.
    """
    med = feats.groupby(labels).median()
    names, rest = {}, []
    for s, r in med.iterrows():
        if r["Cyclone_Inlet_Draft"] > -60:
            names[s] = "Trip cool-down (fan off)" if r["inlet_temp_delta_30m"] < -30 else "Heat-up / warm idle (fan off)"
        elif r["temp_drop"] > 80:
            names[s] = "Degraded: high outlet ΔT"
        elif r["inlet_draft_std_1h"] > 15:
            names[s] = "Unstable draft"
        else:
            rest.append(s)
    rest = sorted(rest, key=lambda s: med.loc[s, "Cyclone_Inlet_Gas_Temp"])
    labels_for_rest = ["Normal load", "High load"] if len(rest) == 2 else [f"Running {i}" for i in range(len(rest))]
    names.update(dict(zip(rest, labels_for_rest)))
    return names


def smooth(labels: pd.Series, min_steps: int = MIN_DWELL_STEPS) -> pd.Series:
    """Turn per-sample labels into episodes.

    1. Centered rolling mode over `min_steps + 1` samples removes flicker
       between neighbouring states (mostly Normal <-> High load).
    2. One sequential pass absorbs any run still shorter than `min_steps` into
       the run before it (using the already-smoothed label, so A-B-A-B
       patterns collapse instead of ping-ponging).
    A break in the 5-min index (shutdown / outage) always ends an episode, and
    the first run after a break is never absorbed into the run before the break.
    """
    k = int(labels.max()) + 1
    onehot = pd.get_dummies(labels).reindex(columns=range(k), fill_value=0).astype(float)
    block = (labels.index.to_series().diff() != pd.Timedelta("5min")).cumsum()
    counts = onehot.groupby(block.to_numpy()).transform(
        lambda c: c.rolling(min_steps + 1, center=True, min_periods=1).sum())
    lab = pd.Series(counts.to_numpy().argmax(axis=1), index=labels.index)

    v = lab.to_numpy().copy()
    runs = episodes(lab)
    for r in runs.itertuples():
        if r.steps < min_steps and not np.isnan(r.prev_state):
            v[r.i0 : r.i1] = v[r.i0 - 1]
    return pd.Series(v, index=labels.index)


def episodes(labels: pd.Series) -> pd.DataFrame:
    """Run-length encode a state series (on a possibly gappy 5-min index)."""
    t = labels.index
    v = labels.to_numpy()
    new = np.ones(len(v), dtype=bool)
    new[1:] = (v[1:] != v[:-1]) | (np.diff(t.asi8) != pd.Timedelta("5min").value)
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], len(v))
    block_start = np.ones(len(starts), dtype=bool)  # does this run begin after a gap?
    block_start[1:] = np.diff(t.asi8)[starts[1:] - 1] != pd.Timedelta("5min").value
    prev = np.where(block_start, np.nan, np.r_[np.nan, v[starts[:-1]]])
    return pd.DataFrame({
        "state": v[starts], "start": t[starts], "end": t[ends - 1] + pd.Timedelta("5min"),
        "steps": ends - starts, "i0": starts, "i1": ends, "prev_state": prev,
    })


def profile(df: pd.DataFrame, feats: pd.DataFrame, labels: pd.Series, names: dict) -> pd.DataFrame:
    """clusters_summary: distribution of every tag + frequency/duration per state."""
    ep = episodes(labels)
    ep["duration_h"] = ep["steps"] * 5 / 60
    cols = SENSORS + ["temp_drop", "draft_dp", "inlet_temp_std_1h"]
    rows = []
    for s, g in feats.groupby(labels):
        row = {"state_id": s, "state": names[s], "share_of_running_pct": round(100 * len(g) / len(labels), 1)}
        for c in cols:
            q = g[c].quantile([0.05, 0.5, 0.95])
            row |= {f"{c}_mean": g[c].mean(), f"{c}_std": g[c].std(),
                    f"{c}_p05": q[0.05], f"{c}_p50": q[0.5], f"{c}_p95": q[0.95]}
        e = ep[ep["state"] == s]["duration_h"]
        # per 730 h (one month) of *running* time, since states only exist while running
        row |= {"episodes": len(e), "episodes_per_running_month": len(e) / (labels.index.size * 5 / 60 / 730),
                "median_episode_h": e.median(), "p90_episode_h": e.quantile(0.9), "max_episode_h": e.max()}
        rows.append(row)
    return pd.DataFrame(rows).round(2)
