"""Pani Jombe? — Dhaka waterlogging early warning.
Created by Pritom and Anika, Department of Urban and Regional Planning, KUET.

Run locally:  uvicorn app:app --port 7860
"""
from __future__ import annotations

import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from waterlog import advice, predict

ROOT = Path(__file__).resolve().parent


def _err(msg, status=502):
    return JSONResponse({"error": msg}, status_code=status)


def index(request):
    return FileResponse(ROOT / "static" / "index.html")


def areas(request):
    a = predict.AREAS
    return JSONResponse([{"id": r.id, "name_en": r.name_en, "name_bn": r.name_bn, "lat": r.lat, "lon": r.lon}
                         for r in a.itertuples()])


def risk(request):
    area = request.query_params.get("area", "")
    lang = request.query_params.get("lang", "bn")
    persona = request.query_params.get("persona", "commuter")
    if lang not in ("bn", "en"):
        lang = "bn"
    try:
        res = predict.run_area(area)
    except KeyError:
        return _err("Unknown area", 404)
    except Exception as e:  # live data down: say so, never invent a forecast
        return _err(f"Live weather data is unavailable right now ({type(e).__name__}). Please try again in a few minutes.")
    res["advice"] = advice.advise(res, lang, persona)
    return JSONResponse(res)


def city(request):
    try:
        return JSONResponse(predict.run_city())
    except Exception as e:
        return _err(f"Live weather data is unavailable right now ({type(e).__name__}).")


def replay(request):
    try:
        return JSONResponse(predict.replay(request.query_params.get("date", "")))
    except KeyError:
        return _err("No documented storm on that date", 404)


def replay_list(request):
    return JSONResponse([{"date": e["date"], "n_reported": len(e["reported"]), "reported_mm": e["reported_mm"]}
                         for e in predict.REPLAY])


def model_card(request):
    return JSONResponse(predict.CARD)


def health(request):
    return JSONResponse({"ok": True, "gemini_key": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))})


app = Starlette(routes=[
    Route("/", index),
    Route("/api/areas", areas),
    Route("/api/risk", risk),
    Route("/api/city", city),
    Route("/api/model", model_card),
    Route("/api/replay", replay),
    Route("/api/replays", replay_list),
    Route("/api/health", health),
    Mount("/static", StaticFiles(directory=ROOT / "static"), name="static"),
])
