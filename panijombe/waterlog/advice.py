"""Turns the model's decision into plain advice (Gemini, with a built-in fallback)."""
from __future__ import annotations

import os
import re
import threading
import time

import requests

API = "https://generativelanguage.googleapis.com/v1beta"
_model_choice: dict = {"name": None, "t": 0.0}
_cache: dict = {}
_lock = threading.Lock()

PERSONAS = {
    "commuter": ("office or school commuter", "অফিস/স্কুলগামী যাত্রী"),
    "shop": ("shopkeeper with ground-floor goods", "নিচতলার দোকানদার"),
    "driver": ("rickshaw, CNG or ride-share driver", "রিকশা/সিএনজি/রাইড চালক"),
    "home": ("family at home (ground floor)", "নিচতলায় বসবাসকারী পরিবার"),
}

FALLBACK = {
    "en": {
        "high": "Waterlogging is very likely in {area}{when}. Avoid travelling through it if you can, keep goods off the floor, and keep your phone charged.",
        "moderate": "Water may collect on low roads in {area}{when}. Leave earlier or later, avoid underpasses, and wear sandals rather than shoes.",
        "low": "No waterlogging expected in {area}. Normal travel is fine; check again if heavy rain starts.",
    },
    "bn": {
        "high": "{area}-এ{when} জলাবদ্ধতার সম্ভাবনা খুব বেশি। সম্ভব হলে এই এলাকা দিয়ে যাতায়াত এড়িয়ে চলুন, মালপত্র মেঝে থেকে উঁচুতে রাখুন এবং ফোনে চার্জ রাখুন।",
        "moderate": "{area}-এ{when} নিচু রাস্তায় পানি জমতে পারে। একটু আগে বা পরে বের হোন, আন্ডারপাস এড়িয়ে চলুন, জুতার বদলে স্যান্ডেল পরুন।",
        "low": "{area}-এ জলাবদ্ধতার সম্ভাবনা নেই। স্বাভাবিকভাবে চলাচল করুন; ভারী বৃষ্টি শুরু হলে আবার দেখে নিন।",
    },
}


def _key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _pick_model(key: str) -> str | None:
    """Use GEMINI_MODEL if set; otherwise ask the API which Flash models this
    key can use and take the newest one (names change every few months)."""
    if os.getenv("GEMINI_MODEL"):
        return os.getenv("GEMINI_MODEL")
    if _model_choice["name"] and time.time() - _model_choice["t"] < 86400:
        return _model_choice["name"]
    try:
        r = requests.get(f"{API}/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=10)
        r.raise_for_status()
        names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        flash = [n for n in names if "flash" in n and not any(x in n for x in ("image", "tts", "audio", "live", "exp", "preview", "thinking"))]
        pool = flash or [n for n in names if "flash" in n] or names

        def ver(n):
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return (float(m.group(1)) if m else 0.0, "lite" not in n, -len(n))
        best = max(pool, key=ver) if pool else None
    except Exception:
        best = "gemini-2.5-flash"
    _model_choice.update(name=best, t=time.time())
    return best


def _fmt_when(day: dict, lang: str) -> str:
    w = day.get("window")
    rel = day.get("rel", "today")
    label = {"en": {"today": " today", "tomorrow": " tomorrow"}, "bn": {"today": " আজ", "tomorrow": " আগামীকাল"}}[lang][rel]
    # the time window is already shown as its own line in the app, so not repeated here
    return "" if day["level"] == "low" else label


def fallback(day: dict, area_name: str, lang: str) -> str:
    return FALLBACK[lang][day["level"]].format(area=area_name, when=_fmt_when(day, lang))


def advise(result: dict, lang: str = "bn", persona: str = "commuter") -> dict:
    """Returns {'text', 'source'}; never raises."""
    day = result["days"][0] if result["days"][0]["level"] != "low" or len(result["days"]) == 1 else max(
        result["days"], key=lambda d: d["p"])
    area_name = result["area"]["name_bn" if lang == "bn" else "name_en"]
    base = {"text": fallback(day, area_name, lang), "source": "built-in"}
    key = _key()
    if not key:
        return base
    ck = (result["area"]["id"], day["date"], day["level"], str(day.get("window")), lang, persona)
    with _lock:
        hit = _cache.get(ck)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
    model = _pick_model(key)
    who = PERSONAS.get(persona, PERSONAS["commuter"])[0]
    w = day.get("window") or {}
    facts = (
        f"Area: {result['area']['name_en']}, Dhaka. Day: {day.get('rel', 'today')} ({day['date']}). "
        f"Model decision: {day['level'].upper()} waterlogging risk (score {day['score']}/100). "
        f"Forecast rain that day: about {day['rain_mm']} mm (weather models range {day['rain_mm_range'][0]}–{day['rain_mm_range'][1]} mm); "
        f"heaviest 3 hours about {day['peak_3h_mm']} mm. "
        f"Likely waterlogging window: {w.get('start', 'none')} to {w.get('end', 'none')}. "
        f"Weather models agreeing with this level: {round(day['agreement'] * 100)}%."
    )
    lang_name = "Bangla (Bengali script, simple everyday words)" if lang == "bn" else "simple English"
    prompt = (
        "You write public-safety advice for ordinary people in Dhaka. "
        f"Reader: a {who}. Using ONLY the facts below, write 2 short sentences in {lang_name}: "
        "first the decision and the time window, then one or two concrete actions for this reader. "
        "Do not add numbers, places or facts that are not given. No greeting, no markdown.\n\n" + facts)
    try:
        r = requests.post(f"{API}/models/{model}:generateContent", headers={"x-goog-api-key": key},
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"maxOutputTokens": 400}}, timeout=20)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = " ".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not text:
            return base
        out = {"text": text, "source": f"Gemini ({model})"}
        with _lock:
            _cache[ck] = (time.time(), out)
        return out
    except Exception:
        return base
