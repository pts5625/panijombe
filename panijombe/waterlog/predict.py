"""Live prediction: Open-Meteo multi-model forecast -> features -> model -> decision."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from .features import daily_rain_features

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Dhaka")
# Same two points the model was trained on (north and south Dhaka).
POINTS = {"lat": "23.80,23.74", "lon": "90.40,90.41"}
ENSEMBLE = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "jma_seamless"]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_SECONDS = 600
HEAVY_MM_PER_HOUR = 2.0     # hourly rate that starts to fill drains
DRAIN_HOURS = 3             # typical hours water stays on Dhaka roads after a burst

MODEL = joblib.load(ROOT / "model" / "model.joblib")
CARD = json.loads((ROOT / "model" / "model_card.json").read_text())
FEATURES = CARD["features"]
DEC = CARD["decision"]
H_MOD, H_HIGH = DEC["hazard_moderate"], DEC["hazard_high"]
MATRIX = np.array(DEC["matrix"])
LEVELS = ["low", "moderate", "high"]
AREAS = pd.read_csv(ROOT / "data" / "areas.csv").merge(
    pd.read_csv(ROOT / "data" / "area_features.csv")[["id", "elev", "rel_elev", "dscc", "prior"]], on="id")
NEUTRAL = {  # city-typical values, used to explain what drives each prediction
    "elev": float(AREAS["elev"].median()) if AREAS["elev"].notna().any() else 0.0,
    "rel_elev": 0.0,
    "prior": float(AREAS["prior"].median()),
}

_cache: dict = {"t": 0.0, "data": None}
_lock = threading.Lock()


# ------------------------------------------------------------ live data
def _request(params):
    r = requests.get(FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json(), r.url


def fetch_forecast() -> dict:
    """Hourly rain for the past 7 days + next 3 days from several weather
    models. Returns {member_name: hourly Series} plus metadata."""
    sample = os.getenv("PJ_SAMPLE_FORECAST")
    with _lock:
        if _cache["data"] and time.time() - _cache["t"] < CACHE_SECONDS:
            return _cache["data"]
        base = {"latitude": POINTS["lat"], "longitude": POINTS["lon"], "hourly": "precipitation",
                "past_days": 7, "forecast_days": 3, "timezone": "Asia/Dhaka"}
        if sample:
            raw, url = json.loads(Path(sample).read_text()), "sample-file (offline test)"
        else:
            try:
                raw, url = _request({**base, "models": ",".join(ENSEMBLE)})
            except Exception:
                raw, url = _request(base)  # single best-match model
        locs = raw if isinstance(raw, list) else [raw]
        members: dict[str, list[pd.Series]] = {}
        for loc in locs:
            h = loc["hourly"]
            idx = pd.to_datetime(h["time"])
            for k, v in h.items():
                if not k.startswith("precipitation"):
                    continue
                name = k.replace("precipitation", "").strip("_") or "best_match"
                s = pd.Series(pd.to_numeric(pd.Series(v), errors="coerce").values, index=idx)
                if s.notna().mean() > 0.8:
                    members.setdefault(name, []).append(s)
        series = {m: pd.concat(v, axis=1).mean(axis=1).interpolate(limit=3).fillna(0) for m, v in members.items()}
        if not series:
            raise RuntimeError("Open-Meteo returned no precipitation data")
        data = {"members": series, "fetched_at": datetime.now(TZ).isoformat(timespec="minutes"),
                "url": url, "models": list(series)}
        _cache.update(t=time.time(), data=data)
        return data


# ------------------------------------------------------------ model
def _rows(daily_row: pd.Series, areas: pd.DataFrame) -> pd.DataFrame:
    X = areas[["elev", "rel_elev", "dscc", "lat", "lon", "prior"]].copy()
    for c in daily_row.index:
        X[c] = daily_row[c]
    return X[FEATURES]


def decide(p: np.ndarray) -> dict:
    """Hazard (city) x susceptibility (area's rank today) -> level per area,
    plus a 0-100 gauge score that sits inside the level's band."""
    city = float(np.sort(p)[-5:].mean())
    hz = 2 if city >= H_HIGH else 1 if city >= H_MOD else 0
    rk = (pd.Series(p).rank(pct=True)).to_numpy()
    su = np.where(rk > 2 / 3, 2, np.where(rk > 1 / 3, 1, 0))
    lvl = MATRIX[hz, su]
    base = np.array([0, 40, 70])[lvl]
    width = np.array([39 * min(1.0, city / H_MOD), 29, 30])[lvl]
    score = np.round(base + width * (0.6 + 0.4 * rk) if hz == 0 else base + width * rk).astype(int)
    return {"city": city, "hazard": LEVELS[hz], "level": [LEVELS[i] for i in lvl], "score": score.tolist()}


def run_city() -> dict:
    """Risk for every area, today and tomorrow, using every weather model."""
    fc = fetch_forecast()
    today = pd.Timestamp(datetime.now(TZ).date())
    days = [today, today + pd.Timedelta(days=1)]
    per_member = {}
    for m, s in fc["members"].items():
        d = daily_rain_features(s)
        per_member[m] = {day: (MODEL.predict_proba(_rows(d.loc[day], AREAS))[:, 1], d.loc[day]) for day in days if day in d.index}
    out = {"fetched_at": fc["fetched_at"], "models": fc["models"], "url": fc["url"], "days": {}}
    for day in days:
        mem = [m for m in per_member if day in per_member[m]]
        ps = np.array([per_member[m][day][0] for m in mem])
        rain = pd.DataFrame([per_member[m][day][1] for m in mem])
        med = np.median(ps, axis=0)
        final = decide(med)
        member_lv = np.array([decide(p)["level"] for p in ps])
        agree = (member_lv == np.array(final["level"])[None, :]).mean(axis=0)
        out["days"][str(day.date())] = {
            "hazard": final["hazard"], "city_score": round(final["city"], 4),
            "areas": [{"id": a, "p": float(p), "level": lv, "score": sc, "agreement": float(g)}
                      for a, p, lv, sc, g in zip(AREAS["id"], med, final["level"], final["score"], agree)],
            "rain_mm": round(float(rain["r24"].median()), 1),
            "rain_mm_range": [round(float(rain["r24"].min()), 1), round(float(rain["r24"].max()), 1)],
            "peak_3h_mm": round(float(rain["r3max"].median()), 1),
        }
    return out


def _window(hourly: pd.Series, day: pd.Timestamp, now: datetime):
    """Hours in which waterlogging is likely: heavy-rain hours plus drain-down time."""
    h = hourly[(hourly.index >= day) & (hourly.index < day + pd.Timedelta(days=1))]
    h = h[h.index >= pd.Timestamp(now.replace(tzinfo=None)).floor("h")] if day.date() == now.date() else h
    wet = h[h >= HEAVY_MM_PER_HOUR]
    if wet.empty:
        if h.empty or h.max() < 0.5:
            return None
        peak = h.idxmax()
        return {"start": peak.strftime("%H:00"), "end": (peak + pd.Timedelta(hours=DRAIN_HOURS)).strftime("%H:00"),
                "peak": peak.strftime("%H:00"), "heavy": False}
    start, end = wet.index.min(), wet.index.max() + pd.Timedelta(hours=1 + DRAIN_HOURS)
    return {"start": start.strftime("%H:00"), "end": min(end, day + pd.Timedelta(hours=23, minutes=59)).strftime("%H:00"),
            "peak": wet.idxmax().strftime("%H:00"), "heavy": True}


def explain(area_row: pd.Series, daily_row: pd.Series) -> list[dict]:
    """Share of the push behind this prediction from each group of factors.
    Uses the logistic half of the model, whose log-odds split exactly into a
    sum of parts; each part is measured against a neutral reference
    (no rain / a city-typical area)."""
    lr, scaler = MODEL.lr[-1], MODEL.lr[0]
    w = dict(zip(MODEL.features, lr.coef_[0] / scaler.scale_))
    x = _rows(daily_row, AREAS[AREAS["id"] == area_row["id"]]).iloc[0]
    ref = {**{f: 0.0 for f in ("r24", "r1max", "r3max", "r6max")}, **NEUTRAL}
    groups = {"rain": ["r24", "r1max", "r3max", "r6max"], "terrain": ["elev", "rel_elev"], "history": ["prior"]}
    c = {g: sum(w[f] * (x[f] - ref[f]) for f in fs if f in w) for g, fs in groups.items()}
    total = sum(abs(v) for v in c.values()) or 1.0
    return [{"factor": g, "effect": int(round(100 * v / total))} for g, v in c.items()]


def run_area(area_id: str) -> dict:
    if area_id not in set(AREAS["id"]):
        raise KeyError(area_id)
    city = run_city()
    fc = fetch_forecast()
    now = datetime.now(TZ)
    hourly = pd.concat(fc["members"].values(), axis=1).median(axis=1)
    area = AREAS[AREAS["id"] == area_id].iloc[0]
    days = []
    for dstr, info in city["days"].items():
        a = next(x for x in info["areas"] if x["id"] == area_id)
        day = pd.Timestamp(dstr)
        d = daily_rain_features(hourly)
        days.append({**a, "hazard": info["hazard"], "date": dstr, "rel": "today" if day.date() == now.date() else "tomorrow", "rain_mm": info["rain_mm"], "rain_mm_range": info["rain_mm_range"],
                     "peak_3h_mm": info["peak_3h_mm"], "window": _window(hourly, day, now),
                     "why": explain(area, d.loc[day]) if day in d.index else []})
    nxt = hourly[(hourly.index >= pd.Timestamp(now.replace(tzinfo=None)).floor("h"))].head(36)
    return {
        "area": {"id": area_id, "name_en": area["name_en"], "name_bn": area["name_bn"],
                 "lat": float(area["lat"]), "lon": float(area["lon"]),
                 "elev": None if pd.isna(area["elev"]) else round(float(area["elev"]), 1),
                 "rel_elev": None if pd.isna(area["rel_elev"]) else round(float(area["rel_elev"]), 1),
                 "history": round(float(area["prior"]), 3)},
        "days": days,
        "hourly": [{"t": t.strftime("%Y-%m-%dT%H:00"), "mm": round(float(v), 1)} for t, v in nxt.items()],
        "live": {"source": "Open-Meteo forecast API", "fetched_at": city["fetched_at"],
                 "models": city["models"], "request": city["url"]},
    }


REPLAY = json.loads((ROOT / "static" / "replay.json").read_text()) if (ROOT / "static" / "replay.json").exists() else []


def replay(date: str) -> dict:
    ev = next((e for e in REPLAY if e["date"] == date), None)
    if ev is None:
        raise KeyError(date)
    ids = list(AREAS["id"])
    p = np.array([ev["scores"][a] for a in ids])
    d = decide(p)
    rep = set(ev["reported"])
    lv = dict(zip(ids, d["level"]))
    hit_high = sum(lv[a] == "high" for a in rep)
    hit_any = sum(lv[a] != "low" for a in rep)
    return {"date": date, "hazard": d["hazard"], "reported_mm": ev["reported_mm"], "reanalysis_mm": ev["reanalysis_mm"],
            "areas": [{"id": a, "level": l, "score": s, "reported": a in rep} for a, l, s in zip(ids, d["level"], d["score"])],
            "n_reported": len(rep), "reported_high": hit_high, "reported_flagged": hit_any,
            "others_high": sum(lv[a] == "high" for a in ids if a not in rep), "n_others": len(ids) - len(rep)}
