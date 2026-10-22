"""Model selection by leave-one-year-out only (the 2023+ hold-out is not used here).

Compares a few principled variants and prints the LOYO metrics for each.
Run:  python training/experiments.py
"""
import numpy as np
import pandas as pd

import train as T
from waterlog.features import FEATURES


def loyo(daily, static, ev, feats, rng, scale, model_kw=None, blend=False):
    outs = []
    years = daily.index.year
    for yr in sorted(set(ev["day"].dt.year)):
        ev_tr, ev_te = ev[ev["day"].dt.year != yr], ev[ev["day"].dt.year == yr]
        tr, _ = T.build_rows(daily[years != yr], static, ev_tr, ev_tr, rng, scale)
        te, _ = T.build_rows(daily[years == yr], static, ev_te, ev_tr, rng, scale)
        m = T.make_model(feats)
        if model_kw:
            m.set_params(**model_kw)
        m.fit(tr[feats], tr["y"], sample_weight=T.balanced_weights(tr))
        p = m.predict_proba(te[feats])[:, 1]
        if blend:
            _, q = T.fit_predict(tr, te, feats, "logreg")
            p = 0.5 * (pd.Series(p).rank(pct=True).values + pd.Series(q).rank(pct=True).values)
        outs.append(te.assign(oof=p))
    lo = pd.concat(outs, ignore_index=True)
    return T.evaluate(lo, lo["oof"].values, scale)


def main():
    rng = np.random.default_rng(T.SEED)
    _, events, rain, static = T.load_inputs()
    daily = T.daily_rain_features(rain)
    daily = daily[daily.index >= daily.index.min() + pd.Timedelta(days=7)]
    _, ev = T.align_events(events, daily)
    scale = float(ev["era5_r24"].median())
    no_ll = [f for f in FEATURES if f not in ("lat", "lon")]
    configs = {
        "A current": (FEATURES, None, False),
        "B no lat/lon": (no_ll, None, False),
        "C no lat/lon, stronger regularisation": (no_ll, {"min_samples_leaf": 150, "max_iter": 200, "l2_regularization": 5.0}, False),
        "D B + logistic blend": (no_ll, None, True),
    }
    for name, (f, kw, bl) in configs.items():
        r = loyo(daily, static, ev, f, np.random.default_rng(T.SEED), scale, kw, bl)
        print(f"{name:42s} area-day {r['auc_area_day']:.3f}  PR {r['ap_area_day']:.3f}  "
              f"spatial {r['auc_spatial_within_event']:.3f}  event-day {r['auc_event_day']:.3f}")


if __name__ == "__main__":
    main()
