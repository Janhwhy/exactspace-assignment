"""What happens before the ID fan stops, and which alert rules that supports.

Every shutdown in this record starts with the fan stopping while the gas is hot
(the coast-down), and every fan trip is the fan stopping mid-run. So "can we see
a stop or a trip coming?" is one question: is there a signal in the 15-120 min
before the fan stops?

Method
- Onsets: the start of each fan-off period that is a coast-down (39, the fan stop
  that becomes a shutdown) or a fan trip (157). Start-up idle is not an onset.
- Precursors, evaluated only while the line is hot with the fan pulling:
  a contextual-IF flag, the Unstable-draft state, and the 1 % tails of four
  movement features (draft weakening, draft noise, temperature falling or rising
  over 30 min) plus tag faults. Tails are taken as percentiles of fan-on running
  so every signal is equally rare and the hit rates are comparable.
- Lift: share of onsets with the precursor anywhere in [onset - hi, onset - lo]
  divided by the same share for random fan-on times that are not themselves in
  the 2 h before a fan stop. The "clean" scope keeps only onsets (and random
  times) whose look-back plus 1 h of feature memory is uninterrupted fan-on
  running; without it, the recovery from an earlier trip (fan trips cluster)
  shows up as a "precursor" of the next one.
- Alert rules are scored per firing (2 h refractory): how many firings were
  followed by a fan stop inside the lead window, alerts per running week, and
  how many fan stops were caught.
"""

import numpy as np
import pandas as pd

from .data import SENSORS

WINDOWS_MIN = [(15, 60), (15, 120)]
TAIL = 0.01
FEATURE_MEMORY = pd.Timedelta("1h")   # longest rolling window in states.features
REFRACTORY = pd.Timedelta("2h")
ESCALATE_TEMP_C = 700.0


def fan_on(off: pd.Series, fan_off: pd.Series) -> pd.Series:
    return ~off & ~fan_off


def signals(df, feats, scores, labels, names, if_threshold, on) -> tuple[pd.DataFrame, dict]:
    """Boolean precursor signals on the full grid (False wherever the fan is not pulling)."""
    unstable = next(k for k, v in names.items() if v.startswith("Unstable"))
    sig = pd.DataFrame(index=df.index)
    sig["IF flag"] = (scores["score"] >= if_threshold).reindex(df.index, fill_value=False)
    sig["Unstable-draft state"] = (labels == unstable).reindex(df.index, fill_value=False)
    run = feats[on]
    cuts = {}
    for name, col, q, side in [("draft weakening (30 min)", "inlet_draft_delta_30m", 1 - TAIL, ">"),
                               ("draft noisy (1 h std)", "inlet_draft_std_1h", 1 - TAIL, ">"),
                               ("inlet temp falling (30 min)", "inlet_temp_delta_30m", TAIL, "<"),
                               ("inlet temp rising (30 min)", "inlet_temp_delta_30m", 1 - TAIL, ">")]:
        cuts[name] = round(float(run[col].quantile(q)), 1)
        sig[name] = feats[col] > cuts[name] if side == ">" else feats[col] < cuts[name]
    sig["tag fault (any tag missing)"] = df[SENSORS].isna().any(axis=1)
    return sig & on.to_numpy()[:, None], cuts


def onsets(fan: pd.DataFrame) -> dict:
    return {"shutdown": pd.DatetimeIndex(fan.loc[fan["kind"] == "coast_down", "start"]),
            "fan trip": pd.DatetimeIndex(fan.loc[fan["kind"].str.startswith("fan_trip"), "start"])}


def _count_in(mask: pd.Series, times, lo, hi) -> np.ndarray:
    """Number of True samples of `mask` in [t - hi, t - lo] for each t."""
    c = np.r_[0, np.cumsum(mask.to_numpy())]
    a = mask.index.searchsorted(times - hi)
    b = mask.index.searchsorted(times - lo, side="right")
    return c[b] - c[a]


def _clean(on: pd.Series, times, back) -> np.ndarray:
    """Was the fan pulling for the whole of [t - back, t)?"""
    c = np.r_[0, np.cumsum((~on).to_numpy())]
    a = on.index.searchsorted(times - back)
    b = on.index.searchsorted(times, side="left")
    return (c[b] - c[a]) == 0


def random_times(on: pd.Series, fan: pd.DataFrame) -> pd.DatetimeIndex:
    """Fan-on samples not within 2 h before a fan stop (the reference population)."""
    t = on.index[on.to_numpy()]
    stops = pd.DatetimeIndex(fan.loc[fan["kind"] != "startup_idle", "start"]).sort_values()
    i = stops.searchsorted(t)
    nxt = stops[np.minimum(i, len(stops) - 1)]
    return t[(i >= len(stops)) | (nxt - t > pd.Timedelta("2h"))]


def lift(sig: pd.DataFrame, fan: pd.DataFrame, on: pd.Series) -> pd.DataFrame:
    ref_all = random_times(on, fan)
    rows = []
    for lo_m, hi_m in WINDOWS_MIN:
        lo, hi = pd.Timedelta(minutes=lo_m), pd.Timedelta(minutes=hi_m)
        for scope in ["all", "clean look-back"]:
            ref = ref_all if scope == "all" else ref_all[_clean(on, ref_all, hi + FEATURE_MEMORY)]
            for target, t in onsets(fan).items():
                if scope != "all":
                    t = t[_clean(on, t, hi + FEATURE_MEMORY)]
                for c in sig.columns:
                    hit = _count_in(sig[c], t, lo, hi) > 0
                    base = (_count_in(sig[c], ref, lo, hi) > 0).mean()
                    rows.append({"window": f"{lo_m}-{hi_m} min before", "scope": scope, "target": target,
                                 "precursor": c, "events": len(t), "events_with_precursor": int(hit.sum()),
                                 "pct_of_events": round(100 * hit.mean(), 1),
                                 "pct_of_random_times": round(100 * base, 1),
                                 "lift": round(hit.mean() / base, 1) if base else np.nan})
    return pd.DataFrame(rows)


def per_onset(sig: pd.DataFrame, fan: pd.DataFrame, on: pd.Series, lo_m=15, hi_m=120) -> pd.DataFrame:
    """One row per fan stop: which precursors fired in the look-back, and how long before."""
    lo, hi = pd.Timedelta(minutes=lo_m), pd.Timedelta(minutes=hi_m)
    rows = []
    for target, times in onsets(fan).items():
        for t in times:
            row = {"onset": t, "target": target,
                   "clean_lookback": bool(_clean(on, pd.DatetimeIndex([t]), hi + FEATURE_MEMORY)[0])}
            for c in sig.columns:
                w = sig[c].loc[t - hi: t - lo]
                row[f"{c} [min before]"] = int((t - w.index[w.to_numpy()][-1]).total_seconds() / 60) if w.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def firings(mask: pd.Series, refractory: pd.Timedelta = REFRACTORY) -> pd.DatetimeIndex:
    """Alert times: the first True sample, then again only after `refractory`."""
    out, last = [], None
    for x in mask.index[mask.to_numpy()]:
        if last is None or x - last >= refractory:
            out.append(x)
            last = x
    return pd.DatetimeIndex(out)


def score_rule(name, trigger, alerts, targets, running_weeks, lead_min) -> dict:
    """Precision = firings followed by a fan stop in (t + lead_lo, t + lead_hi]; caught = stops with a firing before them."""
    lo, hi = (pd.Timedelta(minutes=m) for m in lead_min)
    targets = targets.sort_values()
    a = targets.searchsorted(alerts + lo, side="right")
    b = targets.searchsorted(alerts + hi, side="right")
    hit = b > a
    caught = np.array([((alerts >= t - hi) & (alerts < t - lo)).any() for t in targets])
    return {"rule": name, "trigger": trigger, "lead": f"{lead_min[0]}-{lead_min[1]} min",
            "success_means": "fan stops inside the lead window",
            "firings": len(alerts), "per_running_week": round(len(alerts) / running_weeks, 2),
            "successes": int(hit.sum()), "precision_pct": round(100 * hit.mean(), 1) if len(alerts) else np.nan,
            "targets": len(targets), "targets_caught": int(caught.sum())}


def precursor_rules(sig, fan, on) -> pd.DataFrame:
    """The strongest precursors run as alerts, at a useful lead (15-120 min) and at a short one (0-15 min)."""
    weeks = on.sum() * 5 / 60 / 168
    stops = pd.DatetimeIndex(fan.loc[fan["kind"] != "startup_idle", "start"])
    combos = {"IF flag": sig["IF flag"],
              "draft weakening": sig["draft weakening (30 min)"],
              "IF flag AND draft weakening": sig["IF flag"] & sig["draft weakening (30 min)"]}
    rows = []
    for name, m in combos.items():
        al = firings(m)
        for lead in [(15, 120), (0, 15)]:
            rows.append(score_rule(f"precursor: {name}", name, al, stops, weeks, lead))
    return pd.DataFrame(rows)


def fan_off_alarm(fan: pd.DataFrame, on: pd.Series) -> dict:
    """Rule 1: the fan-off detector run as a live alarm (it fires 10 min into a fan-off period).

    No log says whether the fan really stopped, so the measurable check is that the
    gas responded: inlet temperature fell by more than 30 C during the period.
    """
    f = fan[fan["kind"] != "startup_idle"]
    weeks = on.sum() * 5 / 60 / 168
    ok = f["dip_c"] < -30
    return {"rule": "1. fan-off alarm",
            "trigger": "median of the 3 drafts > -60 Pa for 2 samples while inlet gas > threshold; not within 1 h of a restart",
            "lead": "0: detects the stop 10 min in", "success_means": "inlet gas then falls > 30 C (the gas flow really stopped)",
            "firings": len(f), "per_running_week": round(len(f) / weeks, 2),
            "successes": int(ok.sum()), "precision_pct": round(100 * ok.mean(), 1),
            "targets": len(f), "targets_caught": len(f)}


def escalation(df: pd.DataFrame, fan: pd.DataFrame, minutes=(10, 30, 60, 90, 120), cut=ESCALATE_TEMP_C) -> pd.DataFrame:
    """Once the fan is off, how often does it become a shutdown?

    For each time-into-the-fan-off m: periods still off at m, the share that end as a
    shutdown, and the same with the extra condition inlet gas < `cut` at m.
    """
    f = fan[fan["kind"] != "startup_idle"]
    temp = df["Cyclone_Inlet_Gas_Temp"]
    n_sd = int((f["kind"] == "coast_down").sum())
    rows = []
    for m in minutes:
        g = f[f["duration_min"] >= m]
        sd = (g["kind"] == "coast_down").to_numpy()
        cold = np.array([temp.asof(s + pd.Timedelta(minutes=m)) for s in g["start"]]) < cut
        rows.append({"minutes_into_fan_off": m, "still_off": len(g), "become_shutdown": int(sd.sum()),
                     "precision_pct": round(100 * sd.mean(), 1),
                     f"still_off_and_inlet_below_{int(cut)}C": int(cold.sum()),
                     "become_shutdown_with_temp_rule": int((sd & cold).sum()),
                     "precision_with_temp_rule_pct": round(100 * sd[cold].mean(), 1) if cold.any() else np.nan,
                     "shutdowns_caught_with_temp_rule": f"{int((sd & cold).sum())} of {n_sd}"})
    return pd.DataFrame(rows)


def escalation_rule(df: pd.DataFrame, fan: pd.DataFrame, on: pd.Series, minutes: int = 60,
                    cut: float = ESCALATE_TEMP_C) -> dict:
    """Rule 2: escalate a fan-off alarm to "this is becoming a shutdown"."""
    f = fan[fan["kind"] != "startup_idle"]
    temp = df["Cyclone_Inlet_Gas_Temp"]
    at = f["start"] + pd.Timedelta(minutes=minutes)
    fire = (f["duration_min"] >= minutes).to_numpy() & np.array([temp.asof(t) < cut for t in at])
    sd = (f["kind"] == "coast_down").to_numpy()
    lead_h = ((f["end"] - at)[fire & sd].dt.total_seconds() / 3600).median()  # coast-down ends when the line is cold
    weeks = on.sum() * 5 / 60 / 168
    return {"rule": "2. escalate: becoming a shutdown",
            "trigger": f"fan-off alarm still active after {minutes} min and inlet gas < {cut:.0f} C",
            "lead": f"median {lead_h:.1f} h before the line is cold", "success_means": "the fan stop ends in a shutdown",
            "firings": int(fire.sum()), "per_running_week": round(fire.sum() / weeks, 2),
            "successes": int((fire & sd).sum()), "precision_pct": round(100 * sd[fire].mean(), 1),
            "targets": int(sd.sum()), "targets_caught": int((fire & sd).sum())}
