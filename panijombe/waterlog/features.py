"""Feature engineering shared by training and the live app.

Using one module for both guarantees the model sees the same features
in production that it was trained on (no train/serve skew).
"""
from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIN_FEATURES = ["r24", "r1max", "r3max", "r6max", "rprev1", "r3d", "r7d"]
SEASON_FEATURES = ["doy_sin", "doy_cos"]
AREA_FEATURES = ["elev", "rel_elev", "dscc", "lat", "lon", "prior"]
FEATURES = RAIN_FEATURES + SEASON_FEATURES + AREA_FEATURES

# Physical knowledge built into the model: more rain can never lower the
# risk, higher / relatively higher ground can never raise it, and an area
# with a worse waterlogging history can never be safer.
MONOTONIC = {
    "r24": 1, "r1max": 1, "r3max": 1, "r6max": 1, "rprev1": 1, "r3d": 1,
    "r7d": 0, "doy_sin": 0, "doy_cos": 0,
    "elev": -1, "rel_elev": -1, "dscc": 0, "lat": 0, "lon": 0, "prior": 1,
}


def daily_rain_features(hourly: pd.Series) -> pd.DataFrame:
    """Turn an hourly rainfall series (mm, local-time index) into one row
    of rainfall features per calendar day."""
    s = hourly.sort_index().asfreq("h").fillna(0.0).clip(lower=0)
    roll3 = s.rolling(3, min_periods=1).sum()
    roll6 = s.rolling(6, min_periods=1).sum()
    day = s.index.floor("D")
    df = pd.DataFrame({
        "r24": s.groupby(day).sum(),
        "r1max": s.groupby(day).max(),
        "r3max": roll3.groupby(day).max(),
        "r6max": roll6.groupby(day).max(),
    })
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df.asfreq("D").fillna(0.0)
    df["rprev1"] = df["r24"].shift(1)
    df["r3d"] = df["r24"].shift(1).rolling(3, min_periods=1).sum()
    df["r7d"] = df["r24"].shift(1).rolling(7, min_periods=1).sum()
    doy = df.index.dayofyear.values
    df["doy_sin"] = np.sin(2 * math.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * math.pi * doy / 365.25)
    return df.fillna(0.0)


def area_static_features(areas: pd.DataFrame, elev_points: list[float] | None) -> pd.DataFrame:
    """Elevation of each area's centre and its height relative to a ring of
    4 points ~700 m away (negative = sits in a local hollow)."""
    out = areas[["id", "lat", "lon"]].copy()
    out["dscc"] = (areas["corp"] == "DSCC").astype(int)
    if elev_points is not None and len(elev_points) == 5 * len(areas):
        e = np.asarray(elev_points, dtype=float).reshape(len(areas), 5)
        out["elev"] = e[:, 0]
        out["rel_elev"] = e[:, 0] - e[:, 1:].mean(axis=1)
    elif elev_points is not None and len(elev_points) == len(areas):
        # centres only: compare each area with the other areas within 2.5 km
        e = np.asarray(elev_points, dtype=float)
        lat, lon = areas["lat"].to_numpy(), areas["lon"].to_numpy()
        dist = np.hypot((lat[:, None] - lat[None, :]) * 111.0, (lon[:, None] - lon[None, :]) * 101.8)
        near = (dist > 0) & (dist < 2.5)
        neigh = np.array([e[row].mean() if row.any() else e[i] for i, row in enumerate(near)])
        out["elev"] = e
        out["rel_elev"] = e - neigh
    else:
        out["elev"] = np.nan
        out["rel_elev"] = np.nan
    return out


def susceptibility_prior(events: pd.DataFrame, area_ids, strength: float = 5.0) -> pd.Series:
    """Smoothed share of documented events in which each area was named.
    Shrinks towards the city average so rarely-reported areas are not
    pushed to exactly zero."""
    n = len(events)
    counts = pd.Series(0.0, index=list(area_ids))
    for a in events["areas"]:
        for x in str(a).split(";"):
            if x in counts.index:
                counts[x] += 1
    if n == 0:
        return counts * 0 + 0.1
    mean_rate = counts.mean() / n
    return (counts + strength * mean_rate) / (n + strength)


# ---------------------------------------------------------------- parsers
def read_open_meteo_csv(path_or_text) -> pd.Series:
    """Parse an Open-Meteo CSV export (one or several locations) into a
    single hourly series: the mean across locations."""
    text = Path(path_or_text).read_text() if Path(str(path_or_text)).exists() else str(path_or_text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    data_block = next(b for b in blocks if re.search(r"(^|,)time(,|$)", b.splitlines()[0]))
    df = pd.read_csv(io.StringIO(data_block))
    pcol = next(c for c in df.columns if c.startswith("precipitation") or c.startswith("rain"))
    df["time"] = pd.to_datetime(df["time"])
    if "location_id" in df.columns:
        wide = df.pivot_table(index="time", columns="location_id", values=pcol, aggfunc="mean")
        series = wide.mean(axis=1)
    else:
        series = df.set_index("time")[pcol]
    return series.astype(float).rename("precip")


def read_elevation_json(texts: list[str]) -> list[float]:
    """Accepts the raw responses of the Open-Meteo elevation links, in order."""
    out: list[float] = []
    for t in texts:
        m = re.search(r"\{.*\}", t, re.S)
        out.extend(json.loads(m.group(0))["elevation"])
    return out
