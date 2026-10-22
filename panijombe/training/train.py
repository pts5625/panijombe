"""Train and validate the Pani Jombe? waterlogging model.

Inputs (in ../data):
  areas.csv            40 Dhaka areas with coordinates
  events.csv           31 documented waterlogging events (news reports, with sources)
  rain_hourly.csv      Open-Meteo historical hourly rainfall export (ERA5 reanalysis)
  elevation_*.json     Open-Meteo elevation API responses (optional but recommended)

Outputs:
  ../model/model.joblib       trained gradient-boosting model
  ../model/model_card.json    features, thresholds, validation metrics
  ../data/area_features.csv   static features per area used by the app
  report.md                   validation report

Run:  python training/train.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waterlog.features import (AREA_FEATURES, FEATURES, MONOTONIC, RAIN_FEATURES,  # noqa: E402
                               SEASON_FEATURES, area_static_features, daily_rain_features,
                               read_elevation_json, read_open_meteo_csv)
from waterlog.model import HGB_PARAMS, BlendModel  # noqa: E402

DATA, MODEL_DIR = ROOT / "data", ROOT / "model"
W_UNNAMED = 0.3        # trust in "not named in the news report" as a negative
DRY_SAMPLE = 0.05      # share of completely dry days kept
PRIOR_STRENGTH = 5.0
TEST_FROM_YEAR = 2023  # temporal hold-out: train <= 2022, test >= 2023
SEED = 7


# ------------------------------------------------------------------ inputs
def load_inputs():
    areas = pd.read_csv(DATA / "areas.csv")
    events = pd.read_csv(DATA / "events.csv", parse_dates=["date"])
    rain_path = next((p for p in [DATA / "rain_hourly.csv", *sorted(DATA.glob("open-meteo*.csv"))] if p.exists()), None)
    if rain_path is None:
        sys.exit("Missing rainfall file: put the Open-Meteo CSV export in data/rain_hourly.csv")
    rain = read_open_meteo_csv(rain_path)
    elev_files = sorted(DATA.glob("elevation_*.json"))
    elev = read_elevation_json([p.read_text() for p in elev_files]) if elev_files else None
    static = area_static_features(areas, elev)
    return areas, events, rain, static


def align_events(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """News reports describe rain that often falls overnight. Each event is
    pinned to whichever of (reported day - 1, reported day) had more rain
    in the reanalysis. Consecutive reports of one storm are merged."""
    rows = []
    for e in events.itertuples():
        cands = [d for d in (e.date - pd.Timedelta(days=1), e.date) if d in daily.index]
        if not cands:
            continue
        best = max(cands, key=lambda d: daily.at[d, "r24"])
        rows.append({"event_id": e.event_id, "reported": e.date.date(), "day": best,
                     "areas": set(str(e.areas).split(";")), "reported_mm": e.reported_rain_mm,
                     "era5_r24": round(float(daily.at[best, "r24"]), 1)})
    al = pd.DataFrame(rows)
    merged = al.groupby("day").agg(event_id=("event_id", "+".join),
                                   areas=("areas", lambda s: set().union(*s)),
                                   era5_r24=("era5_r24", "max")).reset_index()
    return al, merged


# ----------------------------------------------------------------- dataset
def build_rows(daily, static, ev_days, prior_events, rng, rain_scale):
    """One row per (area, day). Returns features, label, weight, metadata."""
    area_ids = static["id"].tolist()
    n_ev = len(prior_events)
    counts = pd.Series(0.0, index=area_ids)
    for s in prior_events["areas"]:
        for a in s:
            if a in counts.index:
                counts[a] += 1
    m = counts.mean() / max(n_ev, 1)
    prior_all = (counts + PRIOR_STRENGTH * m) / (n_ev + PRIOR_STRENGTH)
    ev_map = {r.day: r.areas for r in ev_days.itertuples()}
    prior_ev_days = set(prior_events["day"]) if n_ev else set()

    keep = (daily["r24"] >= 0.5) | (rng.random(len(daily)) < DRY_SAMPLE) | daily.index.isin(list(ev_map))
    d = daily[keep]
    dry_boost = np.where(d["r24"] >= 0.5, 1.0, 1.0 / DRY_SAMPLE)

    frames = []
    st = static.set_index("id")
    for a in area_ids:
        f = d.copy()
        for c in ["elev", "rel_elev", "dscc", "lat", "lon"]:
            f[c] = st.at[a, c]
        named = np.array([a in ev_map.get(day, ()) for day in d.index])
        is_ev = d.index.isin(list(ev_map))
        # leave-one-event-out prior for days that are themselves training events
        own = np.array([(a in ev_map[day]) if day in prior_ev_days else False for day in d.index])
        in_prior = np.array([day in prior_ev_days for day in d.index])
        f["prior"] = np.where(in_prior,
                              (counts[a] - own + PRIOR_STRENGTH * m) / (n_ev - 1 + PRIOR_STRENGTH),
                              prior_all[a])
        f["y"] = named.astype(int)
        w_neg = np.exp(-d["r24"].values / rain_scale) * dry_boost
        f["w"] = np.where(named, 1.0, np.where(is_ev, W_UNNAMED, w_neg))
        f["area"] = a
        f["is_event_day"] = is_ev
        frames.append(f)
    out = pd.concat(frames)
    out.index.name = "day"
    return out.reset_index(), prior_all


def make_model(features):
    return HistGradientBoostingClassifier(**HGB_PARAMS, monotonic_cst=[MONOTONIC[f] for f in features])


def balanced_weights(df):
    w = df["w"].values.copy()
    pos = df["y"].values == 1
    w[pos] *= 0.5 * w[~pos].sum() / max(w[pos].sum(), 1e-9)
    return w


def fit_predict(train, test, features, kind="blend"):
    w = balanced_weights(train)
    if kind == "blend":
        m = BlendModel(features).fit(train, train["y"], w)
        return m, m.score(test)
    if kind == "hgb":
        m = make_model(features)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    if kind == "hgb":
        m.fit(train[features], train["y"], sample_weight=w)
    else:
        m.fit(train[features], train["y"], logisticregression__sample_weight=w)
    return m, m.predict_proba(test[features])[:, 1]


# --------------------------------------------------------------- metrics
def evaluate(df, score, rain_scale):
    """Three honest views of skill, all on data the model never saw."""
    df = df.assign(score=score)
    rainy = df[(df["r24"] >= 0.5)]
    conf = rainy[(rainy["y"] == 1) | (~rainy["is_event_day"])]
    res = {}
    if conf["y"].nunique() == 2:
        res["auc_area_day"] = roc_auc_score(conf["y"], conf["score"])
        res["ap_area_day"] = average_precision_score(conf["y"], conf["score"])
        res["base_rate"] = conf["y"].mean()
    # spatial skill: on event days, are the reported areas ranked above the others?
    aucs, wts = [], []
    for _, g in df[df["is_event_day"]].groupby("day"):
        if g["y"].nunique() == 2:
            aucs.append(roc_auc_score(g["y"], g["score"]))
            wts.append(g["y"].sum())
    res["auc_spatial_within_event"] = float(np.average(aucs, weights=wts)) if aucs else np.nan
    # event detection: does the city light up on event days and not on other rainy days?
    city = rainy.groupby("day").agg(y=("is_event_day", "max"),
                                    s=("score", lambda x: np.sort(x.values)[-5:].mean()))
    if city["y"].nunique() == 2:
        res["auc_event_day"] = roc_auc_score(city["y"], city["s"])
        res["n_event_days"] = int(city["y"].sum())
        res["n_rain_days"] = int(len(city))
    return res


def leave_one_year_out(daily, static, ev_days, features, rng, rain_scale):
    """Each year is predicted by a model that never saw that year, with the
    area-history feature also rebuilt from the other years only."""
    outs = []
    years = daily.index.year
    for yr in sorted(set(ev_days["day"].dt.year)):
        ev_tr = ev_days[ev_days["day"].dt.year != yr]
        ev_te = ev_days[ev_days["day"].dt.year == yr]
        tr, _ = build_rows(daily[years != yr], static, ev_tr, ev_tr, rng, rain_scale)
        te, _ = build_rows(daily[years == yr], static, ev_te, ev_tr, rng, rain_scale)
        _, p = fit_predict(tr, te, features)
        outs.append(te.assign(oof=p))
    return pd.concat(outs, ignore_index=True)


# rows: hazard low/moderate/high; columns: susceptibility third low/mid/top -> level 0/1/2
DECISION_MATRIX = np.array([[0, 0, 0], [0, 1, 1], [1, 2, 2]])


def city_score(x):
    return float(np.sort(np.asarray(x))[-5:].mean())


def decide(rows, city_s, h_mod, h_high):
    """rows: one per (area, day) with an 'oof' score. Returns level 0/1/2."""
    s = rows["day"].map(city_s).values
    hz = np.where(s >= h_high, 2, np.where(s >= h_mod, 1, 0))
    rk = rows.groupby("day")["oof"].rank(pct=True).values
    su = np.where(rk > 2 / 3, 2, np.where(rk > 1 / 3, 1, 0))
    return DECISION_MATRIX[hz, su]


def export_replay(lo, ev_days, events, al):
    """Out-of-fold scores on each documented storm day, for the app's
    'test it on a real storm' page (each year scored by a model that never saw it)."""
    out = []
    for r in ev_days.sort_values("day").itertuples():
        g = lo[lo["day"] == r.day]
        if g.empty:
            continue
        rep = al[al["day"] == r.day]
        out.append({"date": str(r.day.date()), "reported_mm": [x for x in rep["reported_mm"].tolist() if x == x],
                    "reanalysis_mm": round(float(g["r24"].iloc[0]), 1), "reported": sorted(r.areas),
                    "scores": {a: round(float(v), 4) for a, v in zip(g["area"], g["oof"])}})
    (ROOT / "static" / "replay.json").write_text(json.dumps(out))


def export_event_sources(events):
    """Source links for the app's How it works page."""
    out = [{"date": str(e.date.date()), "links": [u.strip() for u in str(e.source).split("|")]}
           for e in events.itertuples()]
    (ROOT / "static" / "events.json").write_text(json.dumps(out, indent=1))


# -------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    areas, events, rain, static = load_inputs()
    daily = daily_rain_features(rain)
    daily = daily[daily.index >= daily.index.min() + pd.Timedelta(days=7)]
    al, ev_days = align_events(events, daily)
    rain_scale = float(ev_days["era5_r24"].median())
    # lat/lon dropped after leave-one-year-out model selection (training/experiments.py)
    features = [f for f in FEATURES if f not in ("lat", "lon")
                and not (f in ("elev", "rel_elev") and static["elev"].isna().all())]
    print(f"rain: {rain.index.min()} -> {rain.index.max()}  ({len(rain):,} hours)")
    print(f"events: {len(events)} reports -> {len(ev_days)} event days; "
          f"median reanalysis rain on event days = {rain_scale:.1f} mm")
    print(f"features: {features}")

    # ---------- temporal hold-out
    tr_ev = ev_days[ev_days["day"].dt.year < TEST_FROM_YEAR]
    df_tr, _ = build_rows(daily[daily.index.year < TEST_FROM_YEAR], static, tr_ev, tr_ev, rng, rain_scale)
    df_te, _ = build_rows(daily[daily.index.year >= TEST_FROM_YEAR], static,
                          ev_days[ev_days["day"].dt.year >= TEST_FROM_YEAR], tr_ev, rng, rain_scale)
    feats_noprior = [f for f in features if f != "prior"]
    rain_only = [f for f in features if f in RAIN_FEATURES + SEASON_FEATURES]
    table = {}
    _, p = fit_predict(df_tr, df_te, features, "blend");        table["Pani Jombe model (boosting + logistic blend)"] = evaluate(df_te, p, rain_scale)
    _, p = fit_predict(df_tr, df_te, feats_noprior, "blend");   table["  same, without area history"] = evaluate(df_te, p, rain_scale)
    _, p = fit_predict(df_tr, df_te, features, "hgb");          table["Gradient boosting alone"] = evaluate(df_te, p, rain_scale)
    _, p = fit_predict(df_tr, df_te, features, "logreg");       table["Logistic regression alone"] = evaluate(df_te, p, rain_scale)
    _, p = fit_predict(df_tr, df_te, rain_only, "hgb");         table["Rain-only ML (no area features)"] = evaluate(df_te, p, rain_scale)
    table["Rule: 3-hour rain total"] = evaluate(df_te, df_te["r3max"].values, rain_scale)

    # ---------- leave-one-year-out, for thresholds and robustness
    df_all, prior_all = build_rows(daily, static, ev_days, ev_days, rng, rain_scale)
    lo = leave_one_year_out(daily, static, ev_days, features, rng, rain_scale)
    oof = lo["oof"]
    lo[["day", "area", "y", "is_event_day", "r24", "r3max", "oof"]].to_pickle(ROOT / "training" / "oof.pkl")
    loyo = evaluate(lo, oof.values, rain_scale)
    # ---------- decision layer: hazard (city) x susceptibility (area rank that day)
    rainy = lo[lo["r24"] >= 0.5].copy()
    city = rainy.groupby("day").agg(s=("oof", city_score), ev=("is_event_day", "max"), r=("r24", "first"))
    light_days = city[(~city["ev"]) & (city["r"] < rain_scale / 2)]
    # false-alarm budget on light-rain days: hazard "moderate" <= 8%, "high" <= 2%
    h_mod, h_high = float(light_days["s"].quantile(0.92)), float(light_days["s"].quantile(0.98))
    rainy["lvl"] = decide(rainy, city["s"], h_mod, h_high)
    pos = rainy[rainy["y"] == 1]
    unrep = rainy[rainy["is_event_day"] & (rainy["y"] == 0)]
    light = rainy[(~rainy["is_event_day"]) & (rainy["r24"] < rain_scale / 2)]
    heavy = rainy[(~rainy["is_event_day"]) & (rainy["r24"] >= rain_scale)]
    ev_city = city[city["ev"]]
    rates = {
        "event_days_hazard_moderate_or_high": float((ev_city["s"] >= h_mod).mean()),
        "event_days_hazard_high": float((ev_city["s"] >= h_high).mean()),
        "reported_areas_flagged_moderate_or_high": float((pos["lvl"] >= 1).mean()),
        "reported_areas_flagged_high": float((pos["lvl"] >= 2).mean()),
        "unreported_areas_on_event_days_flagged_high": float((unrep["lvl"] >= 2).mean()),
        "light_rain_days_flagged_moderate_or_high": float((light["lvl"] >= 1).mean()),
        "light_rain_days_flagged_high": float((light["lvl"] >= 2).mean()),
        "heavy_rain_no_report_days_flagged_high": float((heavy["lvl"] >= 2).mean()),
    }

    # ---------- final model on everything
    final = BlendModel(features).fit(df_all, df_all["y"], balanced_weights(df_all))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "model.joblib")
    st = static.copy()
    st["prior"] = st["id"].map(prior_all)
    st.to_csv(DATA / "area_features.csv", index=False)

    def clean(d):
        return {k: (round(v, 3) if isinstance(v, float) and not np.isnan(v) else v) for k, v in d.items()}
    card = {
        "trained_on": str(date.today()),
        "algorithm": "Rank blend of histogram gradient boosting (monotonic physical constraints) and logistic regression",
        "model_selection": "Chosen by leave-one-year-out cross-validation among 5 variants; 2023+ hold-out used only for the final report",
        "features": features,
        "monotonic": {f: MONOTONIC[f] for f in features},
        "decision": {"hazard_moderate": round(h_mod, 4), "hazard_high": round(h_high, 4),
                     "city_score": "mean of the 5 highest area scores that day",
                     "susceptibility": "area's rank among the 40 areas that day, in thirds",
                     "matrix": DECISION_MATRIX.tolist()},
        "rain_source": "Open-Meteo historical weather API (ERA5 reanalysis), hourly",
        "rain_period": [str(rain.index.min().date()), str(rain.index.max().date())],
        "n_reports": int(len(events)), "n_event_days": int(len(ev_days)),
        "n_training_rows": int(len(df_all)), "n_positive_area_days": int(df_all["y"].sum()),
        "median_reanalysis_rain_on_event_days_mm": round(rain_scale, 1),
        "temporal_holdout": {"train": f"<= {TEST_FROM_YEAR - 1}", "test": f">= {TEST_FROM_YEAR}",
                             "results": {k: clean(v) for k, v in table.items()}},
        "leave_one_year_out": clean(loyo),
        "alert_rates_leave_one_year_out": clean(rates),
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    export_event_sources(events)
    export_replay(lo, ev_days, events, al)

    # ---------- human-readable report
    lines = ["# Validation report\n",
             f"Rain data {card['rain_period'][0]} to {card['rain_period'][1]}; "
             f"{len(events)} news reports -> {len(ev_days)} event days; "
             f"{card['n_positive_area_days']} reported area-days.\n",
             f"## Temporal hold-out (train <= {TEST_FROM_YEAR-1}, test >= {TEST_FROM_YEAR})\n",
             "| Model | Area-day ROC-AUC | Area-day PR-AUC | Spatial AUC (within event days) | Event-day AUC |",
             "|---|---|---|---|---|"]
    for k, v in table.items():
        lines.append(f"| {k.strip()} | {v.get('auc_area_day', float('nan')):.3f} | {v.get('ap_area_day', float('nan')):.3f} "
                     f"| {v.get('auc_spatial_within_event', float('nan')):.3f} | {v.get('auc_event_day', float('nan')):.3f} |")
    lines += ["", f"Base rate of positives in the area-day test set: {table['Rule: 3-hour rain total'].get('base_rate', float('nan')):.4f}",
              "", "## Leave-one-year-out", json.dumps(clean(loyo), indent=1), "",
              "## Alert behaviour (leave-one-year-out)", json.dumps(clean(rates), indent=1), "",
              "## Event alignment (reported date vs reanalysis day used)",
              "| Event | Reported | Day used | Reported mm | Reanalysis mm |", "|---|---|---|---|---|"]
    lines += [f"| {r.event_id} | {r.reported} | {r.day.date()} | {r.reported_mm} | {r.era5_r24} |" for r in al.itertuples()]
    (ROOT / "training" / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:len(table) + 6]))
    print("\nLOYO:", clean(loyo))
    print("Alert rates:", clean(rates))
    print("Decision:", card["decision"])


if __name__ == "__main__":
    main()
