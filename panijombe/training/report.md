# Validation report

Rain data 2015-01-01 to 2026-10-15; 31 news reports -> 30 event days; 331 reported area-days.

## Temporal hold-out (train <= 2022, test >= 2023)

| Model | Area-day ROC-AUC | Area-day PR-AUC | Spatial AUC (within event days) | Event-day AUC |
|---|---|---|---|---|
| Pani Jombe model (boosting + logistic blend) | 0.874 | 0.046 | 0.709 | 0.847 |
| same, without area history | 0.860 | 0.043 | 0.541 | 0.849 |
| Gradient boosting alone | 0.874 | 0.030 | 0.707 | 0.837 |
| Logistic regression alone | 0.832 | 0.105 | 0.712 | 0.805 |
| Rain-only ML (no area features) | 0.857 | 0.031 | 0.500 | 0.839 |
| Rule: 3-hour rain total | 0.851 | 0.056 | 0.500 | 0.825 |

Base rate of positives in the area-day test set: 0.0056

## Leave-one-year-out
{
 "auc_area_day": 0.83,
 "ap_area_day": 0.02,
 "base_rate": 0.004,
 "auc_spatial_within_event": 0.729,
 "auc_event_day": 0.799,
 "n_event_days": 30,
 "n_rain_days": 2276
}

## Alert behaviour (leave-one-year-out)
{
 "event_days_hazard_moderate_or_high": 0.833,
 "event_days_hazard_high": 0.733,
 "reported_areas_flagged_moderate_or_high": 0.852,
 "reported_areas_flagged_high": 0.65,
 "unreported_areas_on_event_days_flagged_high": 0.436,
 "light_rain_days_flagged_moderate_or_high": 0.061,
 "light_rain_days_flagged_high": 0.014,
 "heavy_rain_no_report_days_flagged_high": 0.675
}

## Event alignment (reported date vs reanalysis day used)
| Event | Reported | Day used | Reported mm | Reanalysis mm |
|---|---|---|---|---|
| E01 | 2015-07-15 | 2015-07-15 | 62.0 | 21.8 |
| E02 | 2016-05-21 | 2016-05-21 | 43.0 | 93.0 |
| E03 | 2017-04-24 | 2017-04-24 | 22.0 | 43.1 |
| E04 | 2017-07-12 | 2017-07-11 | 103.0 | 30.3 |
| E05 | 2017-08-03 | 2017-08-03 | 121.0 | 23.6 |
| E06 | 2017-09-11 | 2017-09-10 | 80.0 | 25.9 |
| E07 | 2018-04-29 | 2018-04-29 | 50.0 | 15.4 |
| E08 | 2019-07-12 | 2019-07-11 | 114.0 | 21.3 |
| E09 | 2020-07-20 | 2020-07-20 | 63.0 | 13.7 |
| E10 | 2020-10-11 | 2020-10-11 | 54.0 | 1.1 |
| E11 | 2021-06-01 | 2021-06-01 | 85.0 | 40.5 |
| E12 | 2021-06-21 | 2021-06-20 | 15.0 | 17.4 |
| E13 | 2022-05-25 | 2022-05-25 | 24.0 | 3.2 |
| E14 | 2022-10-24 | 2022-10-24 | 125.0 | 128.1 |
| E15 | 2023-07-01 | 2023-07-01 | 84.0 | 26.4 |
| E16 | 2023-08-09 | 2023-08-08 | nan | 18.1 |
| E17 | 2023-09-21 | 2023-09-20 | 113.0 | 18.5 |
| E18 | 2024-05-11 | 2024-05-10 | 87.0 | 0.9 |
| E19 | 2024-05-27 | 2024-05-27 | 116.0 | 179.3 |
| E20 | 2024-06-26 | 2024-06-26 | nan | 20.0 |
| E21 | 2024-07-12 | 2024-07-11 | 130.0 | 18.4 |
| E22 | 2024-09-26 | 2024-09-25 | 79.0 | 32.3 |
| E23 | 2025-07-08 | 2025-07-08 | 77.0 | 32.9 |
| E24 | 2025-08-17 | 2025-08-17 | 50.0 | 9.2 |
| E25 | 2025-09-22 | 2025-09-22 | 105.0 | 19.6 |
| E26 | 2025-10-01 | 2025-10-01 | 203.0 | 20.2 |
| E27 | 2026-05-02 | 2026-05-01 | 28.0 | 23.4 |
| E28 | 2026-07-10 | 2026-07-10 | 15.0 | 38.9 |
| E29 | 2026-07-12 | 2026-07-12 | 175.0 | 47.2 |
| E30 | 2026-07-13 | 2026-07-12 | 110.0 | 47.2 |
| E31 | 2026-09-20 | 2026-09-20 | nan | 14.2 |