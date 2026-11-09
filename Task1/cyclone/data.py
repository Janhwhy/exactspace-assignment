"""Loading and cleaning of the cyclone sensor export.

The raw file is a PI-historian style export: numeric readings mixed with
status strings ("I/O Timeout", "Not Connect", ...) whenever a tag could not be
read. Cleaning here is limited to things that are *not physically possible*.
Statistical outliers are kept on purpose -- they are what the anomaly
detection step is supposed to find.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Material_Temp",
    "Cyclone_Inlet_Draft",
    "Cyclone_cone_draft",
    "Cyclone_Outlet_Gas_draft",
]
TEMPS = [c for c in SENSORS if c.endswith("Temp")]
DRAFTS = [c for c in SENSORS if c not in TEMPS]

SHORT = {
    "Cyclone_Inlet_Gas_Temp": "inlet_temp",
    "Cyclone_Gas_Outlet_Temp": "outlet_temp",
    "Cyclone_Material_Temp": "material_temp",
    "Cyclone_Inlet_Draft": "inlet_draft",
    "Cyclone_cone_draft": "cone_draft",
    "Cyclone_Outlet_Gas_draft": "outlet_draft",
}

FREQ = "5min"
MAX_FILL_STEPS = 6  # interpolate gaps up to 30 min; longer gaps stay missing

# Physical validity limits, chosen from the observed extremes:
#  - 1375.0 appears as an exact repeated maximum on two temperature tags -> transmitter range cap
#  - material temperature reaches -185 C, which cannot happen in a hot process
#  - cone draft has 4 isolated positive spikes (66-489 Pa) while all drafts otherwise stay below ~+42 Pa
TEMP_RANGE = (-10.0, 1370.0)
DRAFT_RANGE = (-600.0, 50.0)

# Signal dropouts that are inside the valid range, only detectable from context:
#  - a temperature tag reading near ambient (<100 C) while the other two temperature
#    tags are both >500 C. Heat can't vanish at one point of the gas path; this catches
#    material temp sitting at exactly 0.0 and inlet temp blipping to 0 / 30 C.
#  - any tag repeating the identical value for >=20 min while the line is hot. Readings
#    have 0.01 resolution and never sit perfectly still while running, so this is the
#    historian holding (or someone substituting) a value -- e.g. inlet temp at exactly
#    800.0 for 25 min.
HOT_C = 500.0
AMBIENT_C = 100.0
STALE_STEPS = 4


def load_raw(xlsx_path: Path, cache_dir: Path) -> pd.DataFrame:
    """Read the export as strings (so status codes survive), cached as parquet."""
    cache = cache_dir / "raw.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = pd.read_excel(xlsx_path, sheet_name="internship-data-1", dtype=str)
        raw["time"] = pd.to_datetime(raw["time"])
        raw = raw.set_index("time")[SENSORS]
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache)
    # One time unit everywhere: pandas 3 parses to microseconds, pandas 2 to nanoseconds.
    # Downstream code compares integer index steps against Timedelta values (ns).
    raw.index = raw.index.as_unit("ns")
    return raw


def _stale(s: pd.Series, min_steps: int) -> pd.Series:
    """True on every sample of a run where the value is unchanged for >= min_steps samples."""
    change = s.ne(s.shift())
    run_id = change.cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (run_len >= min_steps) & s.notna()


def _fill_short_gaps(s: pd.Series, max_steps: int) -> pd.Series:
    """Time-interpolate only NaN runs of length <= max_steps.

    pandas' `interpolate(limit=...)` would partially fill long gaps too, which
    invents data inside multi-hour outages.
    """
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    fillable = isna & (run_len <= max_steps)
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~fillable, filled)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean frame on a strict 5-min grid, data-quality table).

    The clean frame keeps the sensor columns plus boolean flags:
      was_missing  - timestamp absent from the export
      any_fault    - at least one tag was a status code or physically invalid
      any_filled   - at least one tag was interpolated
    """
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    status = raw.where(numeric.isna() & raw.notna())

    limits = {c: TEMP_RANGE if c in TEMPS else DRAFT_RANGE for c in SENSORS}
    invalid = pd.DataFrame(
        {c: ~numeric[c].between(*limits[c]) & numeric[c].notna() for c in SENSORS}
    )
    t = numeric[TEMPS].mask(invalid[TEMPS])
    dropout = pd.DataFrame(False, index=numeric.index, columns=SENSORS)
    for c in TEMPS:
        others = [o for o in TEMPS if o != c]
        dropout[c] = (t[c] < AMBIENT_C) & (t[others] > HOT_C).all(axis=1)
    # "hot" = at least two of the three temperatures say the line is running,
    # so one faulty tag can't decide it on its own
    hot = (t > HOT_C).sum(axis=1) >= 2
    stale = pd.DataFrame({c: _stale(numeric[c], STALE_STEPS) & hot for c in SENSORS})
    numeric = numeric.mask(invalid | dropout | stale)

    # Data-quality table: one row per sensor
    quality = status.apply(lambda c: c.value_counts()).fillna(0).astype(int).T
    quality["physically_invalid"] = invalid.sum()
    quality["dropout_ambient"] = dropout.sum()
    quality["stale_value"] = stale.sum()
    invalid = invalid | dropout | stale

    grid = pd.date_range(raw.index.min(), raw.index.max(), freq=FREQ)
    df = numeric.reindex(grid)
    df.index.name = "time"
    was_missing = ~grid.isin(raw.index)
    quality["missing_timestamps"] = int(was_missing.sum())

    nan_before = df.isna()
    df = df.apply(_fill_short_gaps, max_steps=MAX_FILL_STEPS)
    filled = nan_before & df.notna()
    quality["interpolated"] = filled.sum()
    quality["still_missing"] = df.isna().sum()

    fault = status.notna().reindex(grid, fill_value=False) | invalid.reindex(grid, fill_value=False)
    df["was_missing"] = was_missing
    df["any_fault"] = fault.any(axis=1).to_numpy()
    df["any_filled"] = filled.any(axis=1).to_numpy()

    assert df.index.is_unique and (np.diff(df.index.asi8) == pd.Timedelta(FREQ).value).all()
    return df, quality.rename_axis("sensor").reset_index()
