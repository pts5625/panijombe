---
title: Pani Jombe - Dhaka Waterlogging Alert
emoji: 🌧️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Will water collect on your Dhaka road today? ML early warning.
---

# পানি জমবে? (Pani Jombe?)

Dhaka waterlogging early warning. Pick your area and the app answers one question: will water collect on the roads today, and when.

Created by **Pritom and Anika**, Department of Urban and Regional Planning (URP), Khulna University of Engineering & Technology (KUET), for the BIP Apps4Solutions competition, World Town Planning Day 2026.

## How it works

1. **Live API:** the server pulls the hourly rain forecast (past 7 days plus next 3 days) for Dhaka from the Open-Meteo API, from five weather models (ECMWF, GFS, ICON, GEM, JMA).
2. **Trained ML model:** a rank blend of gradient boosting (with monotonic physical constraints: more rain never lowers risk, higher ground never raises it) and logistic regression. It turns the rain signals (heaviest 1/3/6 hours, daily total, antecedent rain, season) plus fixed area traits (ground height, height relative to neighbours, flooding history) into a score for each of 40 areas. It runs once per weather model; the app uses the median and reports how many models agree.
3. **Decision (hazard × susceptibility):** hazard is citywide (is storm-type rain coming?); susceptibility is the area's rank among the 40 that day. A risk matrix gives low, moderate or high, plus the hours to avoid and an explanation of what drove the decision.
4. **Live AI API:** Google Gemini rewrites the decision as plain Bangla or English advice for the user's situation (commuter, shopkeeper, driver, at home). Without a key, built-in advice is used.

Training data: 31 waterlogging reports (2015 to 2026, 30 storm days, 331 reported area-days) from national newspapers that name the flooded places (sources in `data/events.csv`), matched with hourly ERA5 reanalysis rain from Open-Meteo and Copernicus elevation.

## Results

Trained up to 2022, tested on 2023 to 2026 (never seen in training or model selection). ROC-AUC, 1.0 = perfect, 0.5 = guessing:

| Approach | Finds the right areas on a storm day | Tells storm days from other rainy days | Overall (area-day) |
|---|---|---|---|
| **Pani Jombe model** | **0.709** | **0.847** | **0.874** |
| Same, without area history | 0.541 | 0.849 | 0.860 |
| Gradient boosting alone | 0.707 | 0.837 | 0.874 |
| Logistic regression alone | 0.712 | 0.805 | 0.832 |
| Rain-only ML | 0.500 | 0.839 | 0.857 |
| Rule: 3-hour rain total | 0.500 | 0.825 | 0.851 |

As a warning system (leave-one-year-out): alerts on 83% of documented storm days; 85% of reported areas flagged moderate or high (65% high) against 44% high for unreported areas; only 6% of light-rain area-days get any alert (1.4% high).

The model variant was chosen by leave-one-year-out cross-validation (`training/experiments.py`); full output is in `training/report.md`. The app's How it works page replays every documented storm using out-of-fold predictions.

Known limits: newspaper reports favour busy central areas and miss some storms; ERA5 rain reads about 4 times lower than BMD gauges on storm days (median 21.6 mm against about 80 mm), which smooths short downpours; the 90 m elevation model partly includes buildings.

## Deploy free on Hugging Face Spaces

1. Create an account at huggingface.co, then **New Space**. Choose **Docker**, then the **Blank** template, then **Public**.
2. Upload every file and folder in this project (drag and drop works in **Files**, then **Add file**, then **Upload files**).
3. Get a free Gemini key at https://aistudio.google.com/apikey.
4. In the Space, open **Settings**, then **Variables and secrets**, then **New secret**. Name it `GEMINI_API_KEY` and paste the key.
5. The Space builds in 2 to 4 minutes. Your link is `https://<username>-<space-name>.hf.space`.

Optional: set a variable `GEMINI_MODEL` to pin a specific Gemini model. Otherwise the app picks the newest Flash model your key can use.

## Run locally

```
pip install -r requirements.txt
uvicorn app:app --port 7860
```

## Retrain

Put the Open-Meteo historical export in `data/rain_hourly.csv` and the elevation responses in `data/elevation_1.json` and `data/elevation_2.json`, then run `python training/train.py`.

## Credits

Weather data: Open-Meteo (CC BY 4.0). Map: © OpenStreetMap contributors. Advice wording: Google Gemini.
