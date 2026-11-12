"""Plot helpers. Every function returns the figure and optionally saves a PNG."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import SENSORS, SHORT

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
})

TEMP_COLORS = {"Cyclone_Inlet_Gas_Temp": "#c0392b", "Cyclone_Gas_Outlet_Temp": "#e67e22",
               "Cyclone_Material_Temp": "#8e44ad"}
DRAFT_COLORS = {"Cyclone_Inlet_Draft": "#1f6fb2", "Cyclone_cone_draft": "#16a085",
                "Cyclone_Outlet_Gas_draft": "#5d6d7e"}
COLORS = {**TEMP_COLORS, **DRAFT_COLORS}
SHUTDOWN_COLOR = "#7f8c8d"


def save(fig, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def shade(ax, spans, color=SHUTDOWN_COLOR, alpha=0.25, label=None):
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0, label=label if i == 0 else None)


def temps_and_drafts(df, start, end, title, shutdowns=None, path=None, resample=None):
    """Two stacked panels (temperatures, drafts) for a time slice."""
    sl = df.loc[start:end, SENSORS]
    if resample:
        sl = sl.resample(resample).mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for c, col in TEMP_COLORS.items():
        a1.plot(sl.index, sl[c], color=col, lw=0.8, label=SHORT[c])
    for c, col in DRAFT_COLORS.items():
        a2.plot(sl.index, sl[c], color=col, lw=0.8, label=SHORT[c])
    if shutdowns is not None:
        spans = [(s, e) for s, e in zip(shutdowns["start"], shutdowns["end"])
                 if e >= pd.Timestamp(start) and s <= pd.Timestamp(end)]
        for ax in (a1, a2):
            shade(ax, spans, label="shutdown")
    a1.set_ylabel("temperature (°C)")
    a2.set_ylabel("draft (Pa)")
    a1.set_title(title)
    a1.legend(ncol=4, loc="lower left")
    a2.legend(ncol=4, loc="lower left")
    return save(fig, path)


def distributions(df, path=None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    for ax, c in zip(axes.flat, SENSORS):
        ax.hist(df[c].dropna(), bins=120, color=COLORS[c], alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(SHORT[c])
    fig.suptitle("Sensor distributions (log count) — every tag is bimodal: off vs running",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def correlation(corr: pd.DataFrame, title, path=None):
    labels = [SHORT.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, path)


def shutdown_year(df, events, year, threshold, path=None):
    """One calendar year of inlet temperature with every shutdown shaded."""
    s, e = f"{year}-01-01", f"{year}-12-31 23:55"
    temp = df.loc[s:e, "Cyclone_Inlet_Gas_Temp"].resample("1h").mean()
    draft = df.loc[s:e, "Cyclone_Inlet_Draft"].resample("1h").mean()
    ev = events[(events["end"] >= s) & (events["start"] <= e)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(temp.index, temp, color=TEMP_COLORS["Cyclone_Inlet_Gas_Temp"], lw=0.7)
    a1.axhline(threshold, color="black", ls="--", lw=0.8, label=f"off threshold {threshold:.0f} °C")
    a2.plot(draft.index, draft, color=DRAFT_COLORS["Cyclone_Inlet_Draft"], lw=0.7)
    for ax in (a1, a2):
        shade(ax, zip(ev["start"], ev["end"]), color="#e74c3c", alpha=0.18, label="shutdown")
    a1.set_ylabel("inlet gas temp (°C)")
    a2.set_ylabel("inlet draft (Pa)")
    a1.set_title(f"{year}: {len(ev)} shutdowns, {ev['duration_h'].sum():.0f} h down (hourly means)")
    a1.legend(loc="lower left")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return save(fig, path)


STATE_COLORS = ["#2e86c1", "#f5b041", "#8e44ad", "#17a589", "#c0392b", "#34495e"]
FAN_OFF_COLOR = "#2c3e50"


def anomaly_context(df, labels, names, event, title, pad="6h", path=None, shutdowns=None, fan_off=None,
                    span_label="anomaly"):
    """Temps, drafts and the state ribbon (running states + fan-off time) around one anomaly event."""
    s = event["start"] - pd.Timedelta(pad)
    e = event["end"] + pd.Timedelta(pad)
    sl = df.loc[s:e]
    lab = labels.loc[s:e]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 6.8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 3, 0.6]})
    for c, col in TEMP_COLORS.items():
        a1.plot(sl.index, sl[c], color=col, lw=1, label=SHORT[c])
    for c, col in DRAFT_COLORS.items():
        a2.plot(sl.index, sl[c], color=col, lw=1, label=SHORT[c])
    for ax in (a1, a2):
        ax.axvspan(event["start"], event["end"], color="#f1c40f", alpha=0.35, lw=0, label=span_label)
        if shutdowns is not None:
            sp = [(a, b) for a, b in zip(shutdowns["start"], shutdowns["end"]) if b >= s and a <= e]
            shade(ax, sp, label="shutdown")
    a1.set_ylabel("°C")
    a2.set_ylabel("Pa")
    a1.legend(ncol=5, loc="lower left")
    a2.legend(ncol=5, loc="lower left")
    a1.set_title(title, loc="left")
    # state ribbon
    for st, col in zip(sorted(names), STATE_COLORS):
        m = lab == st
        a3.fill_between(lab.index, 0, 1, where=m.to_numpy(), color=col, step="post", lw=0, label=names[st])
    if fan_off is not None:
        fo = fan_off.loc[s:e]
        a3.fill_between(fo.index, 0, 1, where=fo.to_numpy(), color=FAN_OFF_COLOR, step="post", lw=0, label="fan off")
    a3.set_yticks([])
    a3.grid(False)
    a3.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.5), fontsize=7)
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    return save(fig, path)


def state_scatter(feats, labels, names, path=None, n=40_000, seed=7):
    """Two physically meaningful projections of the running data, coloured by state."""
    idx = np.random.default_rng(seed).choice(len(feats), size=min(n, len(feats)), replace=False)
    f, lab = feats.iloc[idx], labels.iloc[idx]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for st, col in zip(sorted(names), STATE_COLORS):
        m = (lab == st).to_numpy()
        a1.scatter(f["Cyclone_Inlet_Draft"][m], f["Cyclone_Inlet_Gas_Temp"][m], s=2, alpha=0.4, color=col, label=names[st])
        a2.scatter(f["draft_dp"][m], f["temp_drop"][m], s=2, alpha=0.4, color=col)
    a1.set_xlabel("inlet draft (Pa)")
    a1.set_ylabel("inlet gas temp (°C)")
    a1.set_title("Load axis: temperature vs draft")
    a2.set_xlabel("draft drop across cyclone, inlet − outlet (Pa)")
    a2.set_ylabel("gas temp drop across cyclone (°C)")
    a2.set_title("Across-cyclone differentials")
    a2.set_ylim(-150, 300)
    a1.legend(markerscale=6, loc="lower left")
    return save(fig, path)


def state_share_monthly(labels, names, path=None):
    share = labels.groupby(labels.index.to_period("M")).value_counts(normalize=True).unstack().fillna(0)
    share = share.reindex(columns=sorted(names))
    fig, ax = plt.subplots(figsize=(14, 4))
    x = share.index.to_timestamp()
    bottom = np.zeros(len(share))
    for st, col in zip(share.columns, STATE_COLORS):
        ax.bar(x, share[st], width=25, bottom=bottom, color=col, label=names[st])
        bottom += share[st].to_numpy()
    ax.set_ylabel("share of running time")
    ax.set_title("Operating-state mix by month (running time only)")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(0, 1)
    return save(fig, path)


def temp_drop_drift(df, running, shutdowns, path=None):
    """Daily median of inlet − outlet gas temperature while running."""
    d = (df["Cyclone_Inlet_Gas_Temp"] - df["Cyclone_Gas_Outlet_Temp"]).where(running)
    daily = d.resample("1D").median()
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(daily.index, daily, color="#17a589", lw=0.9)
    ax.axhline(0, color="black", lw=0.8)
    shade(ax, zip(shutdowns["start"], shutdowns["end"]), alpha=0.2, label="shutdown")
    ax.set_ylabel("inlet − outlet gas temp (°C)")
    ax.set_title("Temperature drop across the cyclone drifts for months and goes negative — "
                 "an instrument signature, not a process one")
    ax.legend(loc="upper left")
    return save(fig, path)


def anomaly_timeline(df, anomalies, shutdowns, path=None):
    temp = df["Cyclone_Inlet_Gas_Temp"].resample("6h").mean()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(temp.index, temp, color="#bfc9ca", lw=0.7)
    shade(ax, zip(shutdowns["start"], shutdowns["end"]), alpha=0.15, label="shutdown")
    styles = {"isolation_forest": ("#c0392b", "o", "contextual IF"),
              "transient_fan_trip": ("#1f6fb2", "v", "transient fan trip"),
              "fan_trip_extended": ("#8e44ad", "s", "fan trip, fan stayed off")}
    for det, (col, mk, lbl) in styles.items():
        a = anomalies[anomalies["detector"] == det]
        y = df["Cyclone_Inlet_Gas_Temp"].reindex(a["peak_time"]).to_numpy()
        ax.scatter(a["peak_time"], y, color=col, marker=mk, s=16, label=f"{lbl} ({len(a)})", zorder=3)
    ax.set_ylabel("inlet gas temp (°C, 6-h mean)")
    ax.set_title("Anomalous events over the full record")
    ax.legend(ncol=4, loc="lower left")
    return save(fig, path)


def forecast_horizon(by_h: pd.DataFrame, title, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = {"persistence": "#7f8c8d", "arima": "#e67e22", "gbm": "#1f6fb2"}
    for m, col in cols.items():
        ax.plot(by_h.index * 5, by_h[f"{m}_rmse"], marker="o", ms=3, color=col, label=f"{m} RMSE")
        ax.plot(by_h.index * 5, by_h[f"{m}_mae"], ls="--", color=col, alpha=0.7, label=f"{m} MAE")
    ax.set_xlabel("forecast horizon (minutes ahead)")
    ax.set_ylabel("error (°C)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=7)
    return save(fig, path)


def forecast_example(df, fc, start, end, horizon, path=None):
    """Actual vs h-step-ahead predictions (plotted at their target time) over a window."""
    g = fc[(fc["horizon"] == horizon) & (fc["target_time"] >= start) & (fc["target_time"] <= end)]
    y = df.loc[start:end, "Cyclone_Inlet_Gas_Temp"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(y.index, y, color="black", lw=0.9, label="actual (5-min)")
    for m, col in {"persistence": "#7f8c8d", "arima": "#e67e22", "gbm": "#1f6fb2"}.items():
        ax.plot(g["target_time"], g[f"pred_{m}"], marker=".", lw=1, color=col, label=f"{m}, {horizon * 5} min ahead")
    ax.set_ylabel("inlet gas temp (°C)")
    ax.set_title(f"{horizon * 5}-minute-ahead forecasts around a disturbance")
    ax.legend(ncol=4, loc="lower left")
    return save(fig, path)


def duration_hist(events, path=None):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.logspace(np.log10(0.5), np.log10(events["duration_h"].max() * 1.1), 25)
    ax.hist(events["duration_h"], bins=bins, color=SHUTDOWN_COLOR)
    ax.set_xscale("log")
    ax.set_xlabel("shutdown duration (h, log scale)")
    ax.set_ylabel("events")
    ax.set_title("Two populations: stops under 2.5 days, and outages of 4-42 days")
    return save(fig, path)


def sensitivity_grid(grid: pd.DataFrame, path=None):
    """One heatmap per hysteresis value: events (colour + number) and downtime per cell."""
    hys = sorted(grid["hysteresis_c"].unique())
    fig, axes = plt.subplots(1, len(hys), figsize=(3.6 * len(hys), 3.4), sharey=True)
    vmin, vmax = grid["events"].min(), grid["events"].max()
    for ax, hy in zip(np.atleast_1d(axes), hys):
        g = grid[grid["hysteresis_c"] == hy]
        rows, cols = list(dict.fromkeys(g["merge_gap"])), list(dict.fromkeys(g["min_length"]))
        ev = g.pivot(index="merge_gap", columns="min_length", values="events").loc[rows, cols]
        dt = g.pivot(index="merge_gap", columns="min_length", values="downtime_h").loc[rows, cols]
        ax.imshow(ev.to_numpy(), cmap="Greys", vmin=vmin - 3, vmax=vmax + 30)
        for i in range(len(rows)):
            for j in range(len(cols)):
                default = bool(g[(g["merge_gap"] == rows[i]) & (g["min_length"] == cols[j])]["is_default"].iloc[0])
                ax.text(j, i, f"{ev.iat[i, j]}\n{dt.iat[i, j]:,.0f} h", ha="center", va="center", fontsize=8,
                        fontweight="bold" if default else "normal", color="#c0392b" if default else "black")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("minimum length")
        ax.grid(False)
        ax.set_title(f"hysteresis +{hy:.0f} °C", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("merge stops closer than")
    fig.suptitle("Shutdown rules: events and downtime per setting (red = the setting used)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def precursor_lift(lift: pd.DataFrame, window: str, path=None):
    """Share of fan stops with each precursor in the look-back vs share of random running times."""
    d = lift[(lift["window"] == window) & (lift["scope"] == "clean look-back")]
    pre = list(dict.fromkeys(d["precursor"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(pre))
    for ax, (target, g) in zip(axes, d.groupby("target", sort=False)):
        g = g.set_index("precursor").loc[pre]
        ax.barh(y - 0.2, g["pct_of_events"], height=0.38, color="#c0392b", label=f"before a {target}")
        ax.barh(y + 0.2, g["pct_of_random_times"], height=0.38, color=SHUTDOWN_COLOR, label="at random running times")
        for yi, (p, l, n) in enumerate(zip(g["pct_of_events"], g["lift"], g["events_with_precursor"])):
            ax.text(p + 1, yi - 0.2, f"{n} of {g['events'].iloc[0]}  ({l:.1f}x)", va="center", fontsize=7.5)
        ax.set_yticks(y, pre)
        ax.set_xlim(0, max(60, d["pct_of_events"].max() + 20))
        ax.set_xlabel(f"% with the signal {window}")
        ax.set_title(f"{target}s ({g['events'].iloc[0]} with a clean look-back)", fontsize=10)
        ax.legend(loc="lower right")
    axes[0].invert_yaxis()  # shared y: once only
    fig.suptitle("Signals before the fan stops: more frequent than at random, but rare in absolute terms",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return save(fig, path)
