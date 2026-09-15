"""Backend logic for the Diet and Workout Recommendation app.

Pure Python — no UI framework here. Used by app/streamlit_app.py.

NOTE: the repository name ("Gemini-pro") is legacy. The app is powered by
NVIDIA NIM (Nemotron-3-Super-120B), not Google Gemini.
"""

import hashlib
import html
import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# --------------------------------------------------------------------------
# Secrets & configuration (loaded from .env — never commit real keys)
# --------------------------------------------------------------------------
load_dotenv()

NVAPI_KEY = os.environ.get("NVAPI_KEY") or os.environ.get("NVIDIA_API_KEY", "")

if not NVAPI_KEY:
    print("\n" + "=" * 64)
    print("[STARTUP WARNING] NVAPI_KEY is not set.")
    print("  Copy .env.example to .env and add your NVIDIA NIM API key.")
    print("  Plan generation will fail until a key is provided.")
    print("=" * 64 + "\n")


def _get_model() -> ChatNVIDIA:
    """Lazily build the model so a missing key never crashes startup."""
    return ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        nvidia_api_key=NVAPI_KEY,
        temperature=0.6,
        top_p=1,
        max_completion_tokens=4096,
    )


# --------------------------------------------------------------------------
# Local images (downloaded by scripts/fetch_images.py into assets/images)
# --------------------------------------------------------------------------
ASSETS_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "images"
)


def _image_exists(category: str, n: int = 0) -> bool:
    return os.path.isfile(os.path.join(ASSETS_IMAGES_DIR, f"{category}_{n}.jpg"))


def image_path(category: str, n: int = 0) -> str | None:
    """Absolute path to a local image, or None so the UI can show a placeholder."""
    p = os.path.join(ASSETS_IMAGES_DIR, f"{category}_{n}.jpg")
    return p if os.path.isfile(p) else None
    


# --------------------------------------------------------------------------
# Dish & exercise photos: web image search + weighted relevance + local cache.
# No AI image generation. No paid API key. No generic breakfast/lunch/dinner
# fallback. No positional workout mapping.
#
# Every meal/workout is web-searched (DuckDuckGo -> Bing -> Wikimedia Commons),
# candidates are relevance-scored, contradictory titles are rejected, and the
# best match is downloaded, validated and cached under a normalized slug. If
# nothing trustworthy is found the resolvers return None and the UI renders the
# item without a photo.
#
# Priority: CORRECT IMAGE > NO IMAGE > WRONG IMAGE.
# --------------------------------------------------------------------------
MEALS_IMG_DIR = os.path.join(ASSETS_IMAGES_DIR, "meals")
WORKOUTS_IMG_DIR = os.path.join(ASSETS_IMAGES_DIR, "workouts")

_WEB_UA = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
}
_SEARCH_TIMEOUT = 12.0        # seconds per image-search HTTP request
_DOWNLOAD_TIMEOUT = 20.0      # seconds per image download
_MAX_CANDIDATES_PER_QUERY = 10
_MIN_IMAGE_BYTES = 1500
_MIN_IMAGE_DIM = 120

_GEN_LOCKS: dict = {}
_GEN_LOCKS_GUARD = threading.Lock()
_MANIFEST_LOCK = threading.Lock()

# Slugs whose web search already returned nothing trustworthy. Kept for the
# lifetime of the process so Streamlit reruns don't re-search them each time.
_NO_IMAGE_FOUND: set = set()
_NO_IMAGE_FOUND_GUARD = threading.Lock()

_PROTEIN_WORDS = ("chicken", "mutton", "fish", "egg", "paneer", "tofu")

_MEAL_STOPWORDS = {
    "with", "and", "or", "of", "in", "for", "from", "to", "a", "an", "the",
    "on", "at", "you",
}

_MEAL_DECORATIVE_WORDS = {
    "grilled", "roasted", "baked", "steamed", "stir", "fried", "cooked",
    "sautéed", "sauteed", "veggies", "vegetables", "veg", "salad", "healthy",
    "fresh", "bowl", "plate", "side", "mix", "home", "style",
}

_MEAL_CUISINE_WORDS = {
    "idli", "sambar", "dosa", "poha", "upma", "dal", "dhal", "roti", "naan",
    "paratha", "biryani", "paneer", "tikka", "masala", "rajma", "chole",
    "khichdi", "pulao", "pulav", "raita", "rasam", "tandoori", "curd",
    "chapati", "chana", "matar", "palak", "tadka", "vada", "kofta",
}


def _log_image(msg: str) -> None:
    """Internal [IMAGE] debugging to stderr (never exposed in the UI)."""
    print(f"[IMAGE] {msg}", flush=True)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:60]


def _search_query(name: str) -> str:
    """Clean a meal/workout name for cache keys & query phrases: drops
    '(2 pcs)', '2 pieces', '1 pc', quantities and '+'/'&'/'/' separators so
    'Idli Sambar (2 pcs)' and 'Idli Sambar - 2 pieces' share one visual slug.
    Meaningful food words (roti, rice, dal, ...) are never removed."""
    t = re.sub(r"\(.*?\)", " ", name)  # noqa: B034
    t = re.sub(
        r"\b\d+(?:\.\d+)?\s*(pc|pcs|piece|pieces|no|nos|g|gm|grams?|kg|ml|cm"
        r"|cups?|bowls?|plates?|servings?|slices?|tbsp|tsp|sets?|reps?|times?"
        r"|rounds?)\b",
        " ", t, flags=re.IGNORECASE,
    )
    t = re.sub(r"\b\d+\s*x\s*\d+\b", " ", t, flags=re.IGNORECASE)  # "3 x 15"
    t = re.sub(r"\b[x×]\s*\d+(?:\.\d+)?\b", " ", t)               # stray "x 15"
    t = re.sub(r"\b\d+(?:\.\d+)?\b", " ", t)                      # stray counts
    t = t.replace("+", " ").replace("&", " ").replace(":", " ").replace("/", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t or (name or "").strip()


def _meal_slug(name: str) -> str:
    """Cache slug for a meal (quantity-normalized: one visual per dish)."""
    return _slug(_search_query(name))


def _normalize_workout_name(name: str) -> str:
    """Canonical lowercase workout name: strip counts/parens, unify forms."""
    t = re.sub(r"\(.*?\)", " ", (name or "").lower())  # noqa: B034
    t = re.sub(r"\b\d+(?:\.\d+)?\s*(sets?|reps?|times?|rounds?|min(?:ute)?s?|sec(?:ond)?s?)\b", " ", t)
    t = re.sub(r"\b\d+\s*x\s*\d+\b", " ", t)
    t = re.sub(r"\b[x×]\s*\d+(?:\.\d+)?\b", " ", t)   # stray "x 15"
    t = re.sub(r"\b\d+(?:\.\d+)?\s*[x×]\b", " ", t)   # stray "3 x"
    t = re.sub(r"\s+[x×]\s+", " ", t)                 # stray " x "
    t = t.replace("-", " ").replace("_", " ").replace("×", "x")
    t = re.sub(r"\bpushups?\b", "push ups", t)
    t = re.sub(r"\bpullups?\b", "pull ups", t)
    t = re.sub(r"\bsitups?\b", "sit ups", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _WORKOUT_ALIASES.get(t, t)


_WORKOUT_ALIASES = {
    "pushup": "push ups", "push up": "push ups", "pushups": "push ups",
    "db rows": "dumbbell rows", "db row": "dumbbell rows",
    "dumbbell row": "dumbbell rows",
    "bent over row": "dumbbell rows", "bent over rows": "dumbbell rows",
    "jumping jack": "jumping jacks", "calf raise": "calf raises",
    "glute bridges": "glute bridge",
    "situps": "sit ups", "situp": "sit ups",
    "pullups": "pull ups", "pullup": "pull ups",
    "squat": "squats", "lunge": "lunges",
}


def _workout_slug(name: str) -> str:
    return _slug(_normalize_workout_name(name))


# --------------------------------------------------------------------------
# Cache bookkeeping (manifest.json per cache directory): records keep the real
# file extension; unverified files are wiped at startup (re-searched lazily).
# --------------------------------------------------------------------------
def _manifest_path(directory: str) -> str:
    return os.path.join(directory, "manifest.json")


def _read_manifest(directory: str) -> dict:
    p = _manifest_path(directory)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _write_manifest(directory: str, records: dict) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        with open(_manifest_path(directory), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass


def _record_generated(directory: str, slug: str, ext: str) -> None:
    """Append one cached-image record to the manifest (thread-safe)."""
    with _MANIFEST_LOCK:
        records = _read_manifest(directory)
        records[slug] = {"ext": ext, "generated": True}
        _write_manifest(directory, records)


def _list_cache_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return [
        fn for fn in os.listdir(directory)
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))
    ]


def _cleanup_unverified_images(directory: str) -> int:
    """Delete cached images the manifest does not vouch for.

    Anything recorded (or salvaged from an older cache) survives; strays are
    wiped at startup and simply re-searched on the next plan generation.
    Returns the number removed.
    """
    if not os.path.isdir(directory):
        return 0
    try:
        trusted = set(_read_manifest(directory).keys())
    except Exception:  # noqa: BLE001
        trusted = set()
    removed = 0
    for fn in _list_cache_files(directory):
        stem = fn[: fn.rfind(".")]
        if stem in trusted:
            continue
        try:
            os.remove(os.path.join(directory, fn))
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    return removed


def _image_ext(data: bytes) -> str | None:
    """Real image format signature -> file extension (None if not an image)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _cached_path(directory: str, slug: str) -> str | None:
    """Path of the cached image for a slug, or None."""
    rec = _read_manifest(directory).get(slug)
    if not rec:
        return None
    p = os.path.join(directory, f"{slug}.{rec.get('ext', 'jpg')}")
    return p if os.path.isfile(p) else None


def _lock_for(slug: str) -> threading.Lock:
    """Per-slug in-process lock: a slug is only searched/downloaded once even
    when parallel workers / Streamlit reruns race."""
    with _GEN_LOCKS_GUARD:
        lock = _GEN_LOCKS.get(slug)
        if lock is None:
            lock = threading.Lock()
            _GEN_LOCKS[slug] = lock
        return lock


# --------------------------------------------------------------------------
# Text helpers for scoring
# --------------------------------------------------------------------------
def _strip_html(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t or "")).strip()


def _text_has(text: str, tok: str) -> bool:
    """True if tok (optionally with s/es plural) appears in normalized text."""
    variants = {"sambar": ("sambar", "sambhar"),
                "dal": ("dal", "dhal", "daal"), "dhal": ("dal", "dhal", "daal"),
                "yogurt": ("yogurt", "yoghurt"), "yoghurt": ("yogurt", "yoghurt"),
                "biryani": ("biryani", "biriyani", "biriani"),
                "soup": ("soup", "soup"),
                "pancake": ("pancake", "pancakes"), "tofu": ("tofu",), }.get(tok, (tok,))
    for v in variants:
        if re.search(rf"(?<![a-z0-9]){re.escape(v)}(?:s|es)?(?![a-z0-9])", text):
            return True
    return False


def _phrase_in(text: str, phrase: str) -> bool:
    return bool(phrase and phrase.strip().lower() and re.search(re.escape(phrase.strip().lower()), text))


def _segments(name: str) -> list[str]:
    segs = re.split(r"\s*(?:\+| and )\s*", name or "", flags=re.IGNORECASE)
    return [c for c in (_search_query(s) for s in segs) if c]


def _meal_tokens(name: str) -> tuple[list[str], list[str]]:
    clean = _search_query(name)
    toks = [t for t in re.split(r"[^a-z0-9']+", clean.lower()) if t and t not in _MEAL_STOPWORDS]
    majors = [t for t in toks if t not in _MEAL_DECORATIVE_WORDS]
    return toks, majors


def _protein_of(name: str) -> str | None:
    _, majors = _meal_tokens(name)
    return next((w for w in _PROTEIN_WORDS if w in majors), None)


def _protein_conflict(prot: str | None, text: str) -> str | None:
    """If the dish names a specific protein, any OTHER protein present in a
    candidate title is a contradiction (Paneer Tikka must never be Chicken
    Tikka). Returns the offending protein word or None."""
    if not prot:
        return None
    for w in _PROTEIN_WORDS:
        if w != prot and _text_has(text, w):
            return w
    return None


# --------------------------------------------------------------------------
# Targeted query builders (never a bare generic term like 'breakfast')
# --------------------------------------------------------------------------
# Alias phrases used ONLY to widen search / match a prepared dish. The original
# item name is always the primary query.
_MEAL_PHRASE_ALIASES = {
    "poha": ("kanda poha", "flattened rice", "poha recipe"),
    "chilla": ("cheela", "moong dal pancake", "lentil pancake"),
    "cheela": ("chilla", "moong dal pancake", "lentil pancake"),
    "roti": ("chapati", "phulka", "indian roti"),
    "rice": ("chawal", "bhat"),
    "dal": ("daal", "dhal", "lentil curry", "lentil dal"),
    "biryani": ("biriyani", "biriani"),
    "soup": ("shorba", "broth"),
    "curd": ("yogurt", "yoghurt", "dahi"),
    "sambar": ("sambhar",),
}

# Dish-form words: if a candidate text contains one of these it is describing a
# prepared dish, not a raw ingredient.
_DISH_MARKERS = frozenset({
    "chilla", "cheela", "pancake", "soup", "curry", "tikka", "khichdi",
    "kitchari", "dosa", "idli", "vada", "paratha", "sandwich", "toast",
    "upma", "porridge", "biryani", "oats", "oatmeal", "salad", "thali",
    "sabzi", "cutlet", "chutney", "raita", "tadka", "bhaji", "bhurji",
    "stir", "fry", "roast", "tawa", "grill", "rice", "roti", "chawal",
    "bhat", "pulao", "pulav", "pilaf", "naan", "chapati", "bowl", "plate",
    "platter", "kebab", "kabab", "roll", "wrap", "gravy", "stew", "chili",
    "chowder", "ramen", "curries", "parathas", "rotis",
})

# Token-level synonyms: a candidate that uses a synonym still counts as a hit.
_TOKEN_SYNONYMS = {
    "chilla": ("cheela", "chila"),
    "cheela": ("chilla", "chila"),
    "roti": ("chapati", "chappati", "phulka", "flatbread"),
    "chapati": ("roti", "chappati", "phulka", "flatbread"),
    "rice": ("chawal", "bhat"),
    "dal": ("daal", "dhal", "lentil", "lentils"),
    "dhal": ("daal", "dal", "lentil", "lentils"),
    "biryani": ("biriyani", "biriani"),
    "sambar": ("sambhar",),
    "yogurt": ("yoghurt", "curd", "dahi"),
    "curd": ("yogurt", "yoghurt", "dahi"),
    "masala": ("spiced",),
    "tadka": ("tempered",),
    "oats": ("oatmeal", "rolled oats"),
    "egg": ("eggs", "omelette", "omelet"),
    "mushroom": ("mushrooms", "button mushrooms"),
    "peanuts": ("peanut", "groundnut"),
    "curry": ("curries", "salan",),
}

_WORKOUT_EXERCISE_ALIASES = {
    "push ups": ("push up", "pushup", "push ups", "pushups"),
    "squats": ("squat", "squats"),
    "plank": ("plank", "planks", "forearm plank"),
    "lunges": ("lunge", "lunges"),
    "dumbbell rows": ("dumbbell row", "dumbbell rows", "db row", "db rows",
                      "bent over dumbbell row", "one arm dumbbell row",
                      "one armed dumbbell row", "bent over row"),
    "glute bridge": ("glute bridge", "glute bridges", "glutes bridge",
                     "hip raise", "hip raises", "bridging"),
    "jumping jacks": ("jumping jack", "jumping jacks"),
    "calf raises": ("calf raise", "calf raises", "heel raise", "heel raises",
                    "standing calf raise"),
}

_WORKOUT_UNRELATED_TERMS = (
    "bench press", "chest press", "shoulder press", "deadlift", "squat press",
    "bicep curl", "tricep curl", "lat pulldown", "rowing machine", "treadmill",
    "sit up", "sit ups", "pull up", "pull ups", "kettlebell swing",
    "handstand", "muscle up", "snatch",
)


_GENERIC_FOOD_WORDS = re.compile(
    r"\b(recipe|recipes|food|dish|dishes|photo|photos|image|images|picture|"
    r"pictures|stock|plate|meal|homemade|indian|easy|healthy|cooking|kitchen"
    r"|prepared|delicious)\b"
)


def _tok_hit(text: str, tok: str) -> bool:
    """True if tok (or a synonym of it, with s/es plural) appears in text."""
    if _text_has(text, tok):
        return True
    for v in _TOKEN_SYNONYMS.get(tok, ()):
        if re.search(rf"(?<![a-z0-9]){re.escape(v)}(?:s|es)?(?![a-z0-9])", text):
            return True
    return False


def _alias_hit(name: str, text: str) -> bool:
    """True if a known alias/synonym phrase of this dish appears in text."""
    lc_name = " ".join(_segments(name)).lower()
    for key, alis in _MEAL_PHRASE_ALIASES.items():
        if key not in lc_name:
            continue
        for a in alis:
            if _phrase_in(text, a):
                return True
    return False


def _query_term_hit(name: str, text: str) -> bool:
    """True if any non-stopword search-query token also appears in text."""
    for q in build_meal_image_queries(name):
        for t in re.split(r"[^a-z0-9]+", q.lower()):
            if t and t not in _MEAL_STOPWORDS and _tok_hit(text, t):
                return True
    return False


def _generic_food_hit(text: str) -> bool:
    return bool(_GENERIC_FOOD_WORDS.search(text))


def _primary_dish_phrase(clean: str, segs: list[str]) -> str:
    """Pick the phrase that best represents what the dish looks like. For
    compound meals the main components matter more than every ingredient."""
    if not segs:
        return clean
    first = segs[0]
    if len(segs) >= 2 and any(k in first.lower() for k in ("paneer", "dal", "curry", "soup", "tofu", "mushroom")):
        return f"{first} {segs[1]}"
    return first


def build_meal_image_queries(name: str) -> list[str]:
    """Dedicated, targeted image queries for a meal. Serving quantities are
    dropped; aliases widen the search; the dish name is never replaced."""
    segs = _segments(name)
    clean = " ".join(segs) or _search_query(name) or str(name or "").strip()
    lc = re.sub(r"\s+", " ", (_search_query(name) or "").lower()).strip()
    primary = _primary_dish_phrase(clean, segs)
    tokens = set(re.split(r"[^a-z0-9]+", lc))

    qs: list[str] = []

    if "poha" in tokens and ("peanut" in " ".join(tokens) or "peanuts" in tokens):
        qs += ["Poha with peanuts food", "Kanda poha peanuts",
               "Poha peanuts Indian breakfast", "Poha recipe peanuts"]
    elif "chilla" in tokens or "cheela" in tokens:
        qs += ["Moong dal chilla food", "Moong dal cheela",
               "Moong dal chilla Indian breakfast", "Moong dal chilla recipe"]
    elif "dal" in tokens and "soup" in tokens:
        qs += ["Dal soup Indian food", "Indian lentil soup",
               "Dal soup with roti", "Dal roti Indian food"]
    elif "tofu" in tokens and "quinoa" in tokens:
        qs += ["Stir fried tofu quinoa", "Tofu quinoa bowl",
               "Tofu quinoa healthy meal"]
    elif "dal" in tokens and ("rice" in tokens or "chawal" in tokens):
        qs += ["Dal rice Indian food", "Dal chawal", "Dal rice plate"]
    elif "paneer" in tokens and "tikka" in tokens:
        qs += ["Paneer tikka Indian food", "Paneer tikka kabab",
               "Paneer tikka dish", "Paneer tikka recipe"]
    elif "paneer" in tokens and ("grill" in lc or "grilled" in lc or "grill" in tokens):
        qs += ["Grilled paneer vegetables", "Paneer vegetables plate",
               "Grilled paneer recipe", "Paneer sabzi"]
    elif "mushroom" in tokens and "curry" in tokens:
        qs += ["Mushroom curry rice", "Mushroom curry with rice",
               "Mushroom curry Indian food", "Mushroom curry recipe"]
    elif "idli" in tokens and "sambar" in tokens:
        qs += ["Idli sambar Indian food", "Idli with sambar",
               "South Indian idli sambar", "Idli sambar recipe"]
    elif any(w in tokens for w in _MEAL_CUISINE_WORDS):
        qs += [f"{clean} Indian food", f"{clean} recipe", f"{clean} dish"]
    else:
        qs += [f"{clean} food", f"{clean} recipe", f"{clean} dish"]

    # intersect with aliases so e.g. "poha" also searches "kanda poha"
    lc_name = " ".join(segs).lower()
    for key, alis in _MEAL_PHRASE_ALIASES.items():
        if key not in lc_name:
            continue
        for a in alis:
            qs.append(a)

    out: list[str] = []
    seen: set[str] = set()
    for q in qs:
        k = re.sub(r"\s+", " ", (q or "").strip()).lower()
        if k and k not in seen:
            seen.add(k)
            out.append(q.strip())
    return out[:4]


def build_workout_image_queries(name: str) -> list[str]:
    n = _normalize_workout_name(name)
    return {
        "push ups": ["push up exercise", "push ups proper form", "push up"],
        "squats": ["bodyweight squat exercise", "squat exercise proper form", "squats exercise"],
        "plank": ["plank exercise", "forearm plank exercise", "plank hold"],
        "lunges": ["lunge exercise", "forward lunge proper form", "lunges exercise"],
        "dumbbell rows": ["dumbbell row exercise", "one arm dumbbell row",
                          "bent over dumbbell row"],
        "glute bridge": ["glute bridge exercise", "hip bridge exercise",
                         "glute bridge proper form"],
        "jumping jacks": ["jumping jacks exercise", "jumping jack cardio"],
        "calf raises": ["standing calf raise exercise", "calf raise exercise",
                        "heel raise exercise"],
    }.get(n) or [f"{n} exercise", f"{n} proper form"]


# --------------------------------------------------------------------------
# Web image search (keyless): DuckDuckGo (ddgs package) primary, then Bing
# and Wikimedia Commons as fallbacks if DDG returns too few candidates.
# All results are candidates that still undergo scoring + validation.
# --------------------------------------------------------------------------
def _http_text(url: str, timeout: float = _SEARCH_TIMEOUT) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_WEB_UA)  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(4_000_000)
            return (raw.decode("utf-8", "replace") or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _candidate(url: str, title: str, desc: str, width: int, height: int,
               source: str, page: str) -> dict | None:
    url = (url or "").strip()
    if not url.startswith("http") or len(url) > 2048 or " " in url:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", " ".join(x for x in (title, desc, page) if x)).lower()
    return {
        "url": url,
        "title": (title or "").strip(),
        "desc": (desc or "").strip(),
        "text": text,
        "width": int(width or 0),
        "height": int(height or 0),
        "source": source,
        "page": (page or "").strip(),
    }


def _search_ddg_images(query: str, max_results: int = _MAX_CANDIDATES_PER_QUERY) -> list[dict]:
    """DuckDuckGo image search via the keyless 'ddgs' package (primary)."""
    try:
        from ddgs import DDGS
    except Exception:  # noqa: BLE001
        return _search_ddg_images_legacy(query, max_results)
    try:
        out = []
        with DDGS(timeout=_SEARCH_TIMEOUT) as dgs:
            items = list(dgs.images(query, max_results=max_results, safesearch="off"))
        for it in items or []:
            if not isinstance(it, dict):
                continue
            width = 0
            height = 0
            try:
                width = int(it.get("width") or 0)
                height = int(it.get("height") or 0)
            except Exception:  # noqa: BLE001
                pass
            cand = _candidate(
                it.get("image") or it.get("thumbnail") or "",
                it.get("title") or "", "",
                width, height, "duckduckgo",
                it.get("url") or it.get("source") or "",
            )
            if cand:
                out.append(cand)
        return out[:max_results]
    except Exception:  # noqa: BLE001
        return []


def _search_ddg_images_legacy(query: str, max_results: int = _MAX_CANDIDATES_PER_QUERY) -> list[dict]:
    """Bare DuckDuckGo i.js fallback when the ddgs package is unavailable."""
    try:
        html = _http_text(
            "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
            + "&t=h_&iax=images&ia=images"
        )
        if not html:
            return []
        m = re.search(r'vqd=["\']?([0-9-]+)["\']?', html) or re.search(r"vqd=?([0-9-]+)", html)
        if not m:
            return []
        txt = _http_text(
            "https://duckduckgo.com/i.js?l=us-en&o=json&p=1&vqd=" + m.group(1)
            + "&q=" + urllib.parse.quote(query)
        )
        if not txt:
            return []
        body = txt.strip().lstrip("(").rstrip(");\n").rstrip(")")
        data = json.loads(body)
        items = data if isinstance(data, list) else data.get("results", [])
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cand = _candidate(
                it.get("image") or it.get("tnail") or it.get("thumbnail") or "",
                it.get("title") or "", it.get("url") or "",
                (it.get("width") or 0), (it.get("height") or 0),
                "duckduckgo", it.get("url") or "",
            )
            if cand:
                out.append(cand)
        return out[:max_results]
    except Exception:  # noqa: BLE001
        return []


def _search_bing_images(query: str, max_results: int = _MAX_CANDIDATES_PER_QUERY) -> list[dict]:
    """Bing image search (secondary, keyless HTML scrape of image result JSON)."""
    try:
        page = _http_text(
            "https://www.bing.com/images/search?q=" + urllib.parse.quote(query)
            + "&form=HDRSC2"
        )
        if not page:
            return []
        metas = re.findall(r'\bm="([^"]+)"', page) or re.findall(r"\bm='([^']+)'", page)
        out = []
        for ms in metas:
            try:
                meta = json.loads(html.unescape(ms))
            except Exception:  # noqa: BLE001
                continue
            cand = _candidate(
                meta.get("murl") or meta.get("turl") or "",
                meta.get("md") or meta.get("t") or "",
                "", meta.get("mwidth") or 0, meta.get("mheight") or 0,
                "bing", meta.get("purl") or meta.get("c") or "",
            )
            if cand:
                out.append(cand)
        return out[:max_results]
    except Exception:  # noqa: BLE001
        return []


def _search_wikimedia_images(query: str, max_results: int = _MAX_CANDIDATES_PER_QUERY) -> list[dict]:
    """Wikimedia Commons file search (secondary, keyless MediaWiki API)."""
    try:
        params = urllib.parse.urlencode({
            "action": "query", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
            "gsrlimit": str(max_results),
            "prop": "imageinfo", "iiprop": "url|size|extmetadata",
            "iiurlwidth": "1000", "format": "json",
        })
        txt = _http_text("https://commons.wikimedia.org/w/api.php?" + params)
        if not txt:
            return []
        data = json.loads(txt)
        pages = (data.get("query") or {}).get("pages") or {}
        out = []
        for page in pages.values():
            ii = (page.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata") or {}
            desc = _strip_html((meta.get("ImageDescription") or {}).get("value", ""))
            obj = _strip_html((meta.get("ObjectName") or {}).get("value", ""))
            cats = _strip_html(" ".join((meta.get("Categories") or {}).get("value", "").split("|")))
            cand = _candidate(
                ii.get("url") or "", page.get("title") or "",
                " ".join(x for x in (desc, obj, cats) if x),
                ii.get("width") or 0, ii.get("height") or 0,
                "wikimedia", "https://commons.wikimedia.org/wiki/" + (page.get("title") or ""),
            )
            if cand:
                out.append(cand)
        return out[:max_results]
    except Exception:  # noqa: BLE001
        return []


def search_web_images(query: str, max_results: int = 10) -> list[dict]:
    """Collect actual image URLs for a query. DuckDuckGo primary; Bing and
    Wikimedia are consulted only if DDG falls short, so search stays fast."""
    out: list[dict] = []
    seen: set[str] = set()

    def _add(cands):
        for cand in cands:
            u = cand.get("url") or ""
            if u in seen:
                continue
            seen.add(u)
            out.append(cand)

    try:
        _add(_search_ddg_images(query, max_results))
    except Exception:  # noqa: BLE001
        pass
    if len(out) < 6:
        try:
            _add(_search_bing_images(query, max_results))
        except Exception:  # noqa: BLE001
            pass
    if len(out) < 6:
        try:
            _add(_search_wikimedia_images(query, max_results))
        except Exception:  # noqa: BLE001
            pass
    return out


# --------------------------------------------------------------------------
# Relevance scoring (weighted) + contradiction filter
#
#   exact dish phrase               +100
#   prepared-dish word in result     +40
#   other main ingredient           +30
#   alias / synonym                 +25
#   query term present              +15
#   generic food / photo            +5
#   ingredient-only result          -80
#   missing named protein           -60
#   contradictory protein / dish    -inf (hard reject: wrong > none)
# --------------------------------------------------------------------------
_ACCEPT_MIN_SCORE = 55.0
_CONTRADICT = float("-inf")


def _meal_analysis(name: str, cand: dict) -> dict:
    """Compute all signals used for both scoring and rejection reasons."""
    text = cand.get("text") or ""
    toks, majors = _meal_tokens(name)
    phrase = re.sub(r"\s+", " ", (_search_query(name) or "")).strip().lower()
    prot = _protein_of(name)

    found = [t for t in majors if _tok_hit(text, t)]
    distinct = len(set(found))
    conflict = _protein_conflict(prot, text)
    text_markers = [m for m in _DISH_MARKERS if _tok_hit(text, m)]
    marker_hits = [t for t in majors if t in _DISH_MARKERS and _tok_hit(text, t)]
    phrase_hit = bool(phrase) and _phrase_in(text, phrase)
    alias = _alias_hit(name, text)
    query = _query_term_hit(name, text)
    generic = _generic_food_hit(text)
    prepared = bool(text_markers) or phrase_hit or alias
    ingredient_only = bool(found) and not prepared
    unrelated = not found and not alias and not query
    missing_protein = prot is not None and not _tok_hit(text, prot)

    return {
        "found": list(set(found)),
        "distinct": distinct,
        "marker_hits": list(set(marker_hits)),
        "phrase_hit": phrase_hit,
        "alias": alias,
        "query": query,
        "generic": generic,
        "prepared": prepared,
        "ingredient_only": ingredient_only,
        "unrelated": unrelated,
        "conflict": conflict,
        "missing_protein": missing_protein,
        "majors": majors,
    }


def _score_meal_candidate(name: str, cand: dict) -> float:
    """Weighted relevance for a meal candidate. Returns raw weighted score;
    the resolver applies _ACCEPT_MIN_SCORE. Contradictions return -inf."""
    a = _meal_analysis(name, cand)
    if not a["majors"] or not (cand.get("text") or ""):
        return _CONTRADICT
    if a["conflict"]:
        return _CONTRADICT
    if a["unrelated"]:
        return _CONTRADICT

    score = 0.0
    if a["phrase_hit"]:
        score += 100.0
    score += 40.0 * len(a["marker_hits"])
    score += 30.0 * (a["distinct"] - len(a["marker_hits"]))
    if a["alias"]:
        score += 25.0
    if a["query"]:
        score += 15.0
    if a["generic"]:
        score += 5.0
    if a["ingredient_only"]:
        score -= 80.0
    if a["missing_protein"]:
        score -= 60.0
    if int(cand.get("width") or 0) >= 300 or int(cand.get("height") or 0) >= 300:
        score += 5.0
    return score


def _meal_reject_reason(name: str, cand: dict) -> str | None:
    """Human-readable rejection reason for logging, or None if acceptable."""
    a = _meal_analysis(name, cand)
    if a["conflict"]:
        return f"contradictory ingredient: {a['conflict']}"
    if a["unrelated"]:
        return "clearly unrelated"
    if a["ingredient_only"]:
        return "ingredient only"
    if a["missing_protein"] and not a["prepared"]:
        return "missing main protein"
    return None


def _workout_info(name: str) -> dict:
    n = _normalize_workout_name(name)
    return {
        "name": n,
        "aliases": _WORKOUT_EXERCISE_ALIASES.get(n) or (n,),
    }


def _score_workout_candidate(name: str, cand: dict) -> float:
    """Weighted relevance for a workout candidate. Must match the requested
    exercise; any different-exercise alias or unrelated lift is a hard reject."""
    info = _workout_info(name)
    n = info["name"]
    aliases = info["aliases"]
    text = cand.get("text") or ""
    if not aliases or not text:
        return _CONTRADICT

    # Contradiction filter: different exercise present => reject.
    for other, oa in _WORKOUT_EXERCISE_ALIASES.items():
        if other == n:
            continue
        if any(_text_has(text, a) for a in oa):
            return _CONTRADICT
    if (any(_text_has(text, u) for u in _WORKOUT_UNRELATED_TERMS)
            and "alternative" not in text and " vs " not in text):
        return _CONTRADICT

    # Must reasonably match the requested exercise.
    matched = [a for a in aliases if _text_has(text, a)]
    if not matched:
        return float("-inf")

    score = 40.0 * len(matched[:2])
    if _text_has(text, "proper form"):
        score += 25.0
    if _text_has(text, "exercise") or _text_has(text, "how to"):
        score += 15.0
    if any(_text_has(text, w) for w in ("illustration", "step", "guide", "diagram", "workout")):
        score += 5.0
    if int(cand.get("width") or 0) >= 300 or int(cand.get("height") or 0) >= 300:
        score += 5.0
    return score


def _workout_reject_reason(name: str, cand: dict) -> str | None:
    info = _workout_info(name)
    n = info["name"]
    aliases = info["aliases"]
    text = cand.get("text") or ""
    for other, oa in _WORKOUT_EXERCISE_ALIASES.items():
        if other == n:
            continue
        if any(_text_has(text, a) for a in oa):
            return f"different exercise: {other}"
    if any(_text_has(text, u) for u in _WORKOUT_UNRELATED_TERMS) and "alternative" not in text and " vs " not in text:
        return "unrelated exercise"
    if not any(_text_has(text, a) for a in aliases):
        return "exercise not recognized"
    return None


# --------------------------------------------------------------------------
# Download + image validation (kept independent of scoring)
# --------------------------------------------------------------------------
def _download_bytes(url: str) -> bytes | None:
    """HTTP GET an image URL with retries on transient errors (429/403/timeout).
    Rejects HTML responses and empty bodies. Returns raw bytes or None."""
    if not url or " " in url:
        return None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={**_WEB_UA, "Accept": "image/*"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if ctype and not ctype.startswith("image") and "octet-stream" not in ctype:
                    return None
                data = resp.read()
            if not data or len(data) < _MIN_IMAGE_BYTES:
                return None
            return data
        except urllib.error.HTTPError as e:  # noqa: BLE001
            if e.code == 429 and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            if attempt < 2:
                time.sleep(1.0)
                continue
            return None
    return None


def _validate_image(data: bytes) -> str | None:
    """Verify bytes are a real, decodable image. Returns file extension or None."""
    if not data or len(data) < _MIN_IMAGE_BYTES:
        return None
    ext = _image_ext(data)
    if not ext:
        return None
    try:
        from PIL import Image
        import io as _io
        im = Image.open(_io.BytesIO(data))
        w, h = im.size
        im.verify()
        if w < _MIN_IMAGE_DIM or h < _MIN_IMAGE_DIM:
            return None
    except ImportError:
        pass  # PIL absent => trust magic bytes + min size
    except Exception:  # noqa: BLE001
        return None
    return ext


def _download_validated(url: str) -> tuple[bytes, str] | None:
    """Download + validate in one call (used by the resolver's candidate loop)."""
    data = _download_bytes(url)
    if data is None:
        return None
    ext = _validate_image(data)
    if ext is None:
        return None
    return data, ext


# --------------------------------------------------------------------------
# Cache-first resolution: query -> candidates -> rank -> download -> validate
# -> cache, stopping as soon as a sufficiently relevant image is obtained.
# --------------------------------------------------------------------------
def _best_accepted(cands: list[dict], scorer) -> list[tuple[float, dict]]:
    scored = []
    for c in cands:
        try:
            s = scorer(c)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(s, (int, float)) and float(s) >= _ACCEPT_MIN_SCORE:
            scored.append((float(s), c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def _resolve_photo(directory: str, slug: str, name: str, queries: list[str],
                   scorer, reasoner) -> str | None:
    """Search -> rank -> download/validate -> local cache. Tries queries in
    order and stops at the first sufficiently relevant, downloadable image."""
    cached = _cached_path(directory, slug)
    if cached:
        _log_image(f"Cache hit: {slug}")
        return cached
    with _NO_IMAGE_FOUND_GUARD:
        if slug in _NO_IMAGE_FOUND:
            return None
    with _lock_for(slug):
        cached = _cached_path(directory, slug)  # re-check inside the lock
        if cached:
            return cached
        for q in queries:
            _log_image(f"Searching: {q}")
            cands = search_web_images(q, _MAX_CANDIDATES_PER_QUERY)
            _log_image(f"Found {len(cands)} image candidates")
            if not cands:
                continue
            ranked = _best_accepted(cands, scorer)
            if not ranked:
                for c in cands[:8]:
                    try:
                        reason = reasoner(name, c) if reasoner else None
                    except Exception:  # noqa: BLE001
                        reason = None
                    if reason:
                        _log_image(f"Candidate rejected - {reason}: {c.get('title') or '<untitled>'}")
                continue
            for score, cand in ranked[:8]:
                _log_image(f"Candidate score {int(score)}: {cand.get('title') or '<untitled>'}")
                res = _download_validated(cand.get("url") or "")
                if not res:
                    _log_image(f"Download invalid: {cand.get('url', '')[:80]}")
                    continue
                data, ext = res
                dest = os.path.join(directory, f"{slug}.{ext}")
                try:
                    os.makedirs(directory, exist_ok=True)
                    Path(dest).write_bytes(data)
                except Exception:  # noqa: BLE001
                    _log_image(f"Could not write: {dest}")
                    return None
                _record_generated(directory, slug, ext)
                _log_image(f"Downloaded valid image")
                _log_image(f"Cached: {os.path.basename(dest)}")
                return dest
        _log_image(f"No suitable image found: {slug}")
        with _NO_IMAGE_FOUND_GUARD:
            _NO_IMAGE_FOUND.add(slug)
        return None


# --------------------------------------------------------------------------
# Batch generation + public resolvers (identical signatures to before)
# --------------------------------------------------------------------------
def _fetch_batch_with_deadline(fn, names: list[str], max_workers: int, deadline_s: float) -> list:
    """Run fn(name) in parallel but stop collecting after deadline_s. Photos are
    a best-effort nicety — they must never block plan generation."""
    if not names:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, nm): nm for nm in names}
        deadline = time.time() + deadline_s
        done, _ = wait(futs, timeout=max(0.0, deadline - time.time()))
        for fut in done:
            try:
                results.append(fut.result())
            except Exception:  # noqa: BLE001
                pass
        for fut in set(futs) - set(done):
            fut.cancel()
    return results


def cache_meal_images(names: list[str], max_workers: int = 3, deadline_s: float = 240.0) -> None:
    """Search + download (once) the missing meal images for the given names."""
    todo = [nm for nm in names if _meal_slug(nm) and not _cached_path(MEALS_IMG_DIR, _meal_slug(nm))]
    if not todo:
        return
    try:
        _fetch_batch_with_deadline(
            lambda nm: _resolve_photo(
                MEALS_IMG_DIR, _meal_slug(nm), nm,
                build_meal_image_queries(nm),
                lambda c: _score_meal_candidate(nm, c),
                lambda n, c: _meal_reject_reason(n, c),
            ),
            todo, max_workers, deadline_s,
        )
    except Exception:  # noqa: BLE001
        pass


def cache_workout_images(names: list[str], max_workers: int = 3, deadline_s: float = 120.0) -> None:
    """Search + download (once) the missing workout images for the given names."""
    todo = [nm for nm in names if _workout_slug(nm) and not _cached_path(WORKOUTS_IMG_DIR, _workout_slug(nm))]
    if not todo:
        return
    try:
        _fetch_batch_with_deadline(
            lambda nm: _resolve_photo(
                WORKOUTS_IMG_DIR, _workout_slug(nm), nm,
                build_workout_image_queries(nm),
                lambda c: _score_workout_candidate(nm, c),
                lambda n, c: _workout_reject_reason(n, c),
            ),
            todo, max_workers, deadline_s,
        )
    except Exception:  # noqa: BLE001
        pass


def meal_image_path(name: str) -> str | None:
    """Local path to a relevant, cached image of this meal, or None (no image;
    a generic fallback is NEVER shown for a specific dish)."""
    slug = _meal_slug(name)
    if not slug:
        return None
    cached = _cached_path(MEALS_IMG_DIR, slug)
    if cached:
        return cached
    return _resolve_photo(
        MEALS_IMG_DIR, slug, name,
        build_meal_image_queries(name),
        lambda c: _score_meal_candidate(name, c),
        lambda n, c: _meal_reject_reason(n, c),
    )


def workout_image_path(name: str) -> str | None:
    """Local path to a relevant, cached image of this workout, or None. The
    image is chosen by the workout's name — never by position/index."""
    slug = _workout_slug(name)
    if not slug:
        return None
    cached = _cached_path(WORKOUTS_IMG_DIR, slug)
    if cached:
        return cached
    return _resolve_photo(
        WORKOUTS_IMG_DIR, slug, name,
        build_workout_image_queries(name),
        lambda c: _score_workout_candidate(name, c),
        lambda n, c: _workout_reject_reason(n, c),
    )


# Startup cleanup: wipe any cached image the manifests do not vouch for; they
# will simply be re-searched on the next plan generation.
_MEALS_STARTUP_CLEANED = _cleanup_unverified_images(MEALS_IMG_DIR)
_WORKOUTS_STARTUP_CLEANED = _cleanup_unverified_images(WORKOUTS_IMG_DIR)
if _MEALS_STARTUP_CLEANED or _WORKOUTS_STARTUP_CLEANED:
    print(
        "[image-cache] startup cleanup: removed "
        f"{_MEALS_STARTUP_CLEANED} unverified meal photo(s), "
        f"{_WORKOUTS_STARTUP_CLEANED} unverified workout photo(s) "
        "(they will be re-searched on the next plan generation)"
    )


def _load_credits() -> dict:
    manifest_path = os.path.join(ASSETS_IMAGES_DIR, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                return json.load(f).get("categories", {})
        except Exception:
            pass
    return {}


_IMAGE_CREDITS = _load_credits()



# Strict-JSON LLM output + local BMR/TDEE math
# --------------------------------------------------------------------------
PLAN_SCHEMA = """{
  "bmi_summary": "string - one sentence describing the BMI category picture",
  "restaurants": [{"name": str, "cuisine": str, "why": str, "price": str}],
  "breakfasts":  [{"name": str, "kcal": int, "protein_g": int, "why": str}],
  "lunches":     [{"name": str, "kcal": int, "protein_g": int, "why": str}],
  "dinners":     [{"name": str, "kcal": int, "protein_g": int, "why": str}],
  "workouts":    [{"name": str, "sets": str, "reps": str, "target": str, "tips": str}],
  "weekly_split": [{"day": "Mon", "focus": str}],
  "notes": [str]
}"""


def _build_prompt(input_data: dict, constraints: str = "") -> str:
    """Build the strict-JSON prompt. `constraints` are hard safety guardrails."""
    guardrail = f"\nHARD SAFETY CONSTRAINTS (never violate):\n{constraints}\n" if constraints else ""
    return (
        "You are NutrFit, an expert diet and workout planner. Return your ENTIRE reply as "
        "STRICT JSON matching this schema exactly — no markdown fences, no prose, no text "
        "outside the JSON object:\n"
        f"{PLAN_SCHEMA}\n\n"
        "Person profile:\n"
        f"- name: {input_data.get('name', '')}\n"
        f"- age: {input_data.get('age', '')}\n"
        f"- gender: {input_data.get('gender', '')}\n"
        f"- weight: {input_data.get('weight', '')} kg\n"
        f"- height: {input_data.get('height', '')} cm\n"
        f"- diet: {input_data.get('veg_or_nonveg', '')}\n"
        f"- medical conditions: {input_data.get('disease', 'None')}\n"
        f"- region: {input_data.get('region', '')} ({input_data.get('state', '')})\n"
        f"- allergies: {input_data.get('allergics', 'None')}\n"
        f"- food preference: {input_data.get('foodtype', '')}{guardrail}\n\n"
        "Recommend at least: 6 restaurants, 6 breakfasts, 5 lunches, 5 dinners, 6 workouts, "
        "and a 7-day weekly split (Mon..Sun). For restaurants, use restaurants that would "
        "actually be available near the person's state/region. Keep each `why`/`tips` to "
        "one short sentence. Answer with valid JSON only."
    )


EMPTY_PLAN = {
    "bmi_summary": "",
    "restaurants": [],
    "breakfasts": [],
    "lunches": [],
    "dinners": [],
    "workouts": [],
    "weekly_split": [],
    "notes": [],
}


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of model text (strips markdown fences)."""
    text = (text or "").strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    text = "\n".join(lines)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def _normalize_plan(data: dict) -> dict:
    plan = {k: list(v) if isinstance(v, list) else v for k, v in EMPTY_PLAN.items()}
    for key in EMPTY_PLAN:
        value = data.get(key)
        if isinstance(EMPTY_PLAN[key], list):
            plan[key] = value if isinstance(value, list) else []
        else:
            plan[key] = value if isinstance(value, str) else ""
    return plan


# ── Local fallback catalogs (keyless) ─────────────────────────────────────────
_FALLBACK_BREAKFASTS = [
    {"name": "Masala oats", "kcal": 180, "protein_g": 7, "why": "Fiber-rich, low GI, keeps you full till lunch."},
    {"name": "Poha with peanuts", "kcal": 250, "protein_g": 8, "why": "Light, iron-rich and easy to digest."},
    {"name": "Idli sambar (2 pc)", "kcal": 220, "protein_g": 6, "why": "Steamed, low calorie, gut-friendly."},
    {"name": "Moong dal chilla", "kcal": 200, "protein_g": 11, "why": "Protein-packed, high-fiber savory pancake."},
    {"name": "Boiled eggs with toast", "kcal": 300, "protein_g": 18, "why": "Quick protein to start the day."},
    {"name": "Veggie sandwich", "kcal": 240, "protein_g": 9, "why": "Balanced carbs with colorful vegetables."},
    {"name": "Ragi porridge", "kcal": 210, "protein_g": 9, "why": "Calcium-rich millet, gives steady energy."},
    {"name": "Greek yogurt + fruit", "kcal": 200, "protein_g": 14, "why": "High protein, probiotics for digestion."},
]

_FALLBACK_LUNCHES = [
    {"name": "Dal + rice + salad", "kcal": 450, "protein_g": 14, "why": "Classic balanced meal, easy portion control."},
    {"name": "Roti + rajma (3 rotis)", "kcal": 480, "protein_g": 16, "why": "Plant protein + slow carbs, very filling."},
    {"name": "Vegetable khichdi + curd", "kcal": 420, "protein_g": 14, "why": "Light yet filling, gentle on digestion."},
    {"name": "Paneer sandwich", "kcal": 380, "protein_g": 20, "why": "Good protein hit with toasty veggies."},
    {"name": "Chole + 2 roti", "kcal": 450, "protein_g": 15, "why": "Fiber and protein rich, regional favorite."},
    {"name": "Veg pulao + raita", "kcal": 430, "protein_g": 12, "why": "Satiating one-pot meal, add salad on the side."},
    {"name": "Quinoa veg bowl", "kcal": 400, "protein_g": 16, "why": "Complete protein from quinoa + veggies."},
]

_FALLBACK_DINNERS = [
    {"name": "Grilled paneer + veggies", "kcal": 320, "protein_g": 22, "why": "Lean protein, low carb, easy to digest at night."},
    {"name": "Dal soup + 1 roti", "kcal": 300, "protein_g": 14, "why": "Warm, light and high fiber."},
    {"name": "Stir-fried tofu + quinoa", "kcal": 350, "protein_g": 24, "why": "Balanced amino acids, keeps recovery on track."},
    {"name": "Mushroom curry + rice", "kcal": 310, "protein_g": 12, "why": "Filling, savory and low in calories."},
    {"name": "Besan chilla + curd", "kcal": 280, "protein_g": 15, "why": "Protein-rich dinner, no heaviness."},
    {"name": "Paneer tikka + salad", "kcal": 290, "protein_g": 18, "why": "Chargrilled flavor without extra oil."},
    {"name": "Vegetable soup + grilled tofu", "kcal": 260, "protein_g": 16, "why": "Very light, ideal for later meals."},
]

_FALLBACK_WORKOUTS = [
    {"name": "Squats", "sets": "3", "reps": "15", "target": "Legs", "tips": "Push through heels, keep chest up."},
    {"name": "Push-ups", "sets": "3", "reps": "12", "target": "Chest", "tips": "Lower slowly for 2-3 seconds."},
    {"name": "Plank", "sets": "3", "reps": "45 sec", "target": "Core", "tips": "Squeeze glutes and brace your core."},
    {"name": "Lunges", "sets": "3", "reps": "10/leg", "target": "Legs", "tips": "Step long, knee stays above ankle."},
    {"name": "Dumbbell rows", "sets": "3", "reps": "12", "target": "Back", "tips": "Pull the elbow toward your hip."},
    {"name": "Glute bridge", "sets": "3", "reps": "15", "target": "Glutes", "tips": "Squeeze at the top for 1 second."},
    {"name": "Jumping jacks", "sets": "3", "reps": "30 sec", "target": "Cardio", "tips": "Land soft on the balls of your feet."},
    {"name": "Calf raises", "sets": "3", "reps": "20", "target": "Calves", "tips": "Rise tall, lower slowly."},
]

_FALLBACK_WEEKLY_SPLIT = [
    {"day": "Mon", "focus": "Legs"},
    {"day": "Tue", "focus": "Upper Body"},
    {"day": "Wed", "focus": "Core & Cardio"},
    {"day": "Thu", "focus": "Active Recovery"},
    {"day": "Fri", "focus": "Full Body"},
    {"day": "Sat", "focus": "Cardio + Mobility"},
    {"day": "Sun", "focus": "Rest"},
]

_WEEK_ORDER = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_str(value, default: str = "") -> str:
    return value if isinstance(value, str) else str(value or default)


def _clean_meal(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = _to_str(item.get("name")).strip()
    if not name:
        return None
    return {
        "name": name,
        "kcal": _to_int(item.get("kcal"), 0),
        "protein_g": _to_int(item.get("protein_g"), 0),
        "why": _to_str(item.get("why"))[:120],
    }


def _clean_workout(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    name = _to_str(item.get("name")).strip()
    if not name:
        return None
    return {
        "name": name,
        "sets": _to_str(item.get("sets"), "3"),
        "reps": _to_str(item.get("reps"), "12"),
        "target": _to_str(item.get("target")),
        "tips": _to_str(item.get("tips"))[:120],
    }


def _enrich_plan(plan: dict, state: str = "", region: str = "") -> dict:
    """Clean LLM output and pad every section from local fallbacks."""
    if state or region:
        plan["state"] = plan.get("state") or state
        plan["region"] = plan.get("region") or region
    for key, catalog, target in (
        ("breakfasts", _FALLBACK_BREAKFASTS, 5),
        ("lunches", _FALLBACK_LUNCHES, 4),
        ("dinners", _FALLBACK_DINNERS, 4),
    ):
        items = [m for m in (_clean_meal(i) for i in plan.get(key, [])) if m]
        used = {m["name"].lower() for m in items}
        for cand in catalog:
            if len(items) >= target:
                break
            if cand["name"].lower() not in used:
                items.append(dict(cand))
                used.add(cand["name"].lower())
        plan[key] = items[:target]

    workouts = [w for w in (_clean_workout(i) for i in plan.get("workouts", [])) if w]
    used = {w["name"].lower() for w in workouts}
    for cand in _FALLBACK_WORKOUTS:
        if len(workouts) >= 6:
            break
        if cand["name"].lower() not in used:
            workouts.append(dict(cand))
            used.add(cand["name"].lower())
    plan["workouts"] = workouts[:6]

    split = [
        {"day": _to_str(d.get("day")), "focus": _to_str(d.get("focus"))}
        for d in plan.get("weekly_split", [])
        if isinstance(d, dict) and _to_str(d.get("day")) and _to_str(d.get("focus"))
    ]
    present = {s["day"] for s in split}
    for cand in _FALLBACK_WEEKLY_SPLIT:
        if cand["day"] not in present:
            split.append(dict(cand))
            present.add(cand["day"])
    plan["weekly_split"] = sorted(split, key=lambda s: _WEEK_ORDER.get(s["day"], 99))[:7]

    restaurants = []
    for r in plan.get("restaurants", []):
        if isinstance(r, dict) and _to_str(r.get("name")).strip():
            restaurants.append({
                "name": _to_str(r.get("name")),
                "cuisine": _to_str(r.get("cuisine")),
                "why": _to_str(r.get("why"))[:120],
                "price": _to_str(r.get("price")),
            })
    plan["restaurants"] = restaurants[:6]
    if not plan["restaurants"]:
        plan["restaurants"] = _fallback_restaurants(
            plan.get("state", "") or state, plan.get("region", "") or region
        )

    notes = [n for n in plan.get("notes", []) if isinstance(n, str) and n.strip()]
    if not notes:
        notes = ["Drink 2-3 litres of water daily, keep workouts consistent, and prioritize sleep."]
    plan["notes"] = [n[:160] for n in notes[:6]]

    return plan


def _compute_metrics(height_cm: float, weight_kg: float, age: int, gender: str) -> dict:
    """Local Mifflin-St Jeor: BMI category, BMR, TDEE, and daily calorie target."""
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m**2)

    if bmi < 18.5:
        category, color, kcal = "Underweight", "blue", 1.0   # target = +300/day
    elif bmi < 25:
        category, color, kcal = "Normal weight", "green", 0.0
    elif bmi < 30:
        category, color, kcal = "Overweight", "yellow", 0.5  # target = -500/day
    else:
        category, color, kcal = "Obesity", "red", 0.5        # target = -500/day

    if gender.lower().startswith("f"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    tdee = bmr * 1.55  # moderately active
    if kcal == 1.0:
        target = tdee + 300
    elif kcal == 0.5:
        target = tdee - 500
    else:
        target = tdee

    return {
        "bmi": bmi,
        "bmi_category": category,
        "bmi_color": color,
        "bmr": bmr,
        "tdee": tdee,
        "daily_target": target,
    }


def _bmi_cat(bmi: float) -> tuple[str, str]:
    if bmi < 18.5:
        return "Underweight", "blue"
    elif bmi < 25:
        return "Normal weight", "green"
    elif bmi < 30:
        return "Overweight", "amber"
    return "Obesity", "red"


# ── UI constants (shared with the Streamlit frontend) ─────────────────────────

REGION_STATES = {
    "North": ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Rajasthan"],
    "South": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana"],
    "East": ["West Bengal", "Odisha", "Bihar", "Jharkhand"],
    "West": ["Maharashtra", "Gujarat", "Goa", "Madhya Pradesh"],
    "North-East": ["Assam", "Meghalaya", "Manipur", "Mizoram", "Nagaland", "Arunachal Pradesh", "Tripura", "Sikkim"],
    "International": ["USA", "UK", "UAE", "Canada", "Singapore", "Australia"],
}

DIET_OPTIONS = ["Veg", "Non-Veg", "Eggetarian"]
DISEASE_CHIPS = ["Diabetes", "Hypertension", "Thyroid", "PCOS", "Gastritis", "Other", "None"]
FOOD_TYPES = ["Home-cooked", "Street food", "Restaurant", "Meal-prep", "Mixed"]

# Local restaurant suggestions (keyless) — used whenever the LLM returns no
# restaurants, so "Restaurants nearby" always has something to show.
RESTAURANT_FALLBACKS = {
    "Delhi": [
        {"name": "Indian Coffee House", "cuisine": "South Indian", "why": "Light, budget-friendly dosa & filter coffee.", "price": "₹150-300"},
        {"name": "Karim's", "cuisine": "Mughlai", "why": "Grilled kebabs, high protein, iconic Old Delhi spot.", "price": "₹300-600"},
        {"name": "Amber", "cuisine": "Modern Indian", "why": "Central Delhi fine dining with regional thalis.", "price": "₹800-1500"},
    ],
    "Punjab": [
        {"name": "Bharawan Da Dhaba", "cuisine": "Punjabi", "why": "Dal makhani & tandoori rotis low on oil.", "price": "₹200-400"},
        {"name": "Chawla's", "cuisine": "North Indian", "why": "Tandoori chicken, grilled, protein-rich.", "price": "₹400-700"},
    ],
    "Haryana": [
        {"name": "Aap Ki Rasoi", "cuisine": "North Indian", "why": "Fresh home-style thalis with balanced portions.", "price": "₹200-400"},
        {"name": "Bikanerwala", "cuisine": "Rajasthani", "why": "Wholesome dal-bati tours and curd.", "price": "₹200-350"},
    ],
    "Uttar Pradesh": [
        {"name": "Indian Coffee House", "cuisine": "South Indian", "why": "Vegetarian sambhar-dosa, light and filling.", "price": "₹150-300"},
        {"name": "Tunday Kababi", "cuisine": "Awadhi", "why": "Famous Lucknow galouti, but pair with salad.", "price": "₹200-350"},
    ],
    "Rajasthan": [
        {"name": "Bikanervala", "cuisine": "Rajasthani", "why": "Balanced thali and dal-pakwan, veg-friendly.", "price": "₹200-350"},
        {"name": "Khandelwal Samosa", "cuisine": "Rajasthani", "why": "Baked samosas with mint chutney, portion-controlled.", "price": "₹50-150"},
    ],
    "Kerala": [
        {"name": "Azad Restaurant", "cuisine": "Kerala", "why": "Fish moilee & puttu, high omega-3.", "price": "₹300-500"},
        {"name": "Buhari", "cuisine": "South Indian", "why": "Biriyani & grilled meats, flavorful and filling.", "price": "₹250-450"},
    ],
    "Tamil Nadu": [
        {"name": "Saravana Bhavan", "cuisine": "South Indian", "why": "Idli-dosa, low GI carbs, veg-heavy.", "price": "₹150-300"},
        {"name": "Anandha Bhavan", "cuisine": "South Indian", "why": "Mini tiffin & curd rice — light and digestible.", "price": "₹150-300"},
    ],
    "Karnataka": [
        {"name": "Adyar Ananda Bhavan", "cuisine": "South Indian", "why": "Milkshakes & thali, portion-flexible.", "price": "₹200-400"},
        {"name": "MTR", "cuisine": "South Indian", "why": "Rava idli and kesari bath, classic Bengaluru.", "price": "₹250-450"},
    ],
    "Andhra Pradesh": [
        {"name": "New Andhra Kitchen", "cuisine": "Andhra", "why": "Grilled chicken & ragi roti, high protein.", "price": "₹300-500"},
        {"name": "A2B", "cuisine": "South Indian", "why": "Curd rice & vegetable meals, calorie-friendly.", "price": "₹200-350"},
    ],
    "Telangana": [
        {"name": "Taj Biryani House", "cuisine": "Hyderabadi", "why": "Biryani fix — go for a small portion with raita.", "price": "₹250-450"},
        {"name": "Chutneys", "cuisine": "South Indian", "why": "Paneer tikka & salads, balanced choices.", "price": "₹300-500"},
    ],
    "West Bengal": [
        {"name": "Bhojohori Manna", "cuisine": "Bengali", "why": "Fish curry bowl — omega-3 rich, steamed rice.", "price": "₹300-500"},
        {"name": "Kewpie's", "cuisine": "Continental", "why": "Grilled chicken steaks in a calm setting.", "price": "₹400-700"},
    ],
    "Odisha": [
        {"name": "Tunday Kebabi", "cuisine": "Odishi", "why": "Pakhala with grilled fish, hydrating and light.", "price": "₹200-400"},
    ],
    "Bihar": [
        {"name": "Bansal Sweets", "cuisine": "Bihari", "why": "Sattu sharbat & litti-chokha, high fiber.", "price": "₹150-300"},
    ],
    "Jharkhand": [
        {"name": "The Hawai Adda", "cuisine": "Multi-cuisine", "why": "Grilled options with salads, veg-friendly.", "price": "₹300-500"},
    ],
    "Maharashtra": [
        {"name": "Rajdhani", "cuisine": "Gujarati", "why": "Thali with balanced carbs, some protein options.", "price": "₹300-500"},
        {"name": "Irani Cafe (B Merwan)", "cuisine": "Parsi", "why": "Berry pulao & grilled sandwiches, satiating.", "price": "₹200-350"},
    ],
    "Gujarat": [
        {"name": "Rajdhani", "cuisine": "Gujarati", "why": "Rotli plus sabzi thali, portion-friendly.", "price": "₹300-500"},
        {"name": "Gordhan Thal", "cuisine": "Gujarati", "why": "Mixed thali with millet rotli options.", "price": "₹250-450"},
    ],
    "Goa": [
        {"name": "Fishka", "cuisine": "Goan", "why": "Grilled fish and brown rice, seaside freshness.", "price": "₹400-700"},
        {"name": "Viva Panjim", "cuisine": "Goan", "why": "Prawn curry with neer dosa, omega-3 rich.", "price": "₹400-700"},
    ],
    "Madhya Pradesh": [
        {"name": "Bansi Vihar", "cuisine": "Indore", "why": "Poha jalebi breakfast — light when portioned.", "price": "₹100-200"},
    ],
    "Assam": [
        {"name": "Machaan", "cuisine": "Assamese", "why": "Smoked fish & rice, low oil traditional meal.", "price": "₹300-500"},
    ],
    "Meghalaya": [
        {"name": "Cafe Shillong", "cuisine": "Khasi", "why": "Jadoh with pork/veg, local millet rice.", "price": "₹200-400"},
    ],
    "Manipur": [
        {"name": "Chanambam", "cuisine": "Manipuri", "why": "Steamed fish & rice, minimal oil.", "price": "₹150-300"},
    ],
    "Tripura": [
        {"name": "Hilltop Cafe", "cuisine": "Tripuri", "why": "Fermented dishes — gut friendly, high protein.", "price": "₹150-300"},
    ],
    "Nagaland": [
        {"name": "Kohima Kitchen", "cuisine": "Naga", "why": "Smoked pork and bamboo shoots, protein-rich.", "price": "₹200-400"},
    ],
}


def _fallback_restaurants(state: str, region: str) -> list[dict]:
    """State-aware restaurant suggestions so the section is never empty."""
    picks = RESTAURANT_FALLBACKS.get(state or "", [])
    if not picks:
        picks = RESTAURANT_FALLBACKS.get("Delhi", [])
    return list(picks)


# --------------------------------------------------------------------------
# Persistence (SQLite — same tables the Reflex build created)
# --------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nutrifit.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS planrow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL DEFAULT '',
            inputs_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS bmirow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bmi FLOAT NOT NULL DEFAULT 0,
            category TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT ''
        );
        """
    )
    return conn


def cache_key(input_data: dict) -> str:
    return hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()


def get_cached_plan(ckey: str, max_age_s: int = 86400) -> dict | None:
    """Return {plan, metrics} if a plan with this inputs hash exists and is fresh."""
    try:
        conn = _db()
        row = conn.execute(
            "SELECT created_at, result_json FROM planrow WHERE inputs_json = ? ORDER BY id DESC LIMIT 1",
            (ckey,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        created = row["created_at"] or ""
        if created:
            ts = datetime.fromisoformat(created)
            if (datetime.now(ts.tzinfo) - ts).total_seconds() > max_age_s:
                return None
        stored = json.loads(row["result_json"]) if isinstance(row["result_json"], str) else {}
        if isinstance(stored, dict) and stored.get("plan"):
            return {"plan": stored["plan"], "metrics": stored.get("metrics")}
        if isinstance(stored, dict):
            return {"plan": stored, "metrics": None}
    except Exception:
        pass
    return None


def save_plan(inputs: dict, ckey: str, plan: dict, metrics: dict) -> None:
    """Persist inputs + metrics + plan so saved plans reload fully and BMI history works."""
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        payload = {"inputs": inputs, "metrics": metrics, "plan": plan}
        conn = _db()
        conn.execute(
            "INSERT INTO planrow (user_name, inputs_json, result_json, created_at) VALUES (?, ?, ?, ?)",
            (inputs.get("name", ""), ckey, json.dumps(payload, default=str), now_str),
        )
        conn.execute(
            "INSERT INTO bmirow (bmi, category, created_at) VALUES (?, ?, ?)",
            (metrics.get("bmi", 0.0), metrics.get("bmi_category", ""), now_str),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def all_saved_plans() -> list[dict]:
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT id, user_name, created_at FROM planrow ORDER BY id DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_saved_plan(plan_id: int) -> dict | None:
    """Return {plan, metrics, user_name} for a saved row, or None."""
    try:
        conn = _db()
        row = conn.execute(
            "SELECT user_name, result_json FROM planrow WHERE id = ?", (int(plan_id),)
        ).fetchone()
        conn.close()
        if not row:
            return None
        stored = json.loads(row["result_json"]) if isinstance(row["result_json"], str) else {}
        if isinstance(stored, dict) and isinstance(stored.get("plan"), dict):
            return {
                "plan": stored["plan"],
                "metrics": stored.get("metrics"),
                "user_name": row["user_name"],
                "inputs": stored.get("inputs", {}),
            }
        if isinstance(stored, dict):
            return {
                "plan": stored,
                "metrics": None,
                "user_name": row["user_name"],
                "inputs": {},
            }
    except Exception:
        pass
    return None


def all_bmis() -> list[dict]:
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT bmi, category, created_at FROM bmirow ORDER BY id ASC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def clear_history() -> int:
    """Delete all saved plans and BMI rows. Returns number of plans deleted."""
    try:
        conn = _db()
        cur = conn.execute("DELETE FROM planrow")
        n = cur.rowcount
        conn.execute("DELETE FROM bmirow")
        conn.commit()
        conn.close()
        return n
    except Exception:
        return 0


# --------------------------------------------------------------------------
# PDF export
# --------------------------------------------------------------------------
_UNICODE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
]


def _find_unicode_font() -> str | None:
    for path in _UNICODE_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def pdf_bytes(plan: dict, metrics: dict | None = None) -> bytes:
    """Render the plan to a PDF (Unicode font when available). Returns bytes."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    uni_font = _find_unicode_font()
    has_uni = False
    if uni_font:
        try:
            pdf.add_font("AppUni", fname=uni_font)
            has_uni = True
        except Exception:  # noqa: BLE001
            has_uni = False

    def _cell(txt: str, w: int = 0, h: int = 8, style: str = "B", size: int = 10):
        if has_uni:
            pdf.set_font("AppUni", size=size)
        else:
            pdf.set_font("Helvetica", style, size)
        pdf.cell(w, h, txt.replace("₹", "Rs "), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def _mc(txt: str, h: int = 6, style: str = "", size: int = 10):
        if has_uni:
            pdf.set_font("AppUni", size=size)
        else:
            pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, h, txt.replace("₹", "Rs "), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.add_page()
    _cell("Diet & Workout Plan", h=10, style="B", size=16)
    if metrics:
        _cell(
            f"BMI {metrics['bmi']:.2f} ({metrics['bmi_category']})  |  "
            f"BMR {metrics['bmr']:.0f} kcal  |  TDEE {metrics['tdee']:.0f} kcal  |  "
            f"Daily target {metrics['daily_target']:.0f} kcal",
            style="",
        )
    pdf.ln(4)
    for section in ("breakfasts", "lunches", "dinners", "restaurants", "workouts"):
        items = plan.get(section, [])
        if not items:
            continue
        _cell(section.replace("_", " ").title() + ":", h=8, style="B", size=12)
        for it in items:
            line = f"- {it.get('name', '')}"
            if "kcal" in it:
                line += f"  ({it.get('kcal', '?')} kcal, {it.get('protein_g', '?')}g protein)"
            elif "sets" in it:
                line += f"  {it.get('sets', '')}x{it.get('reps', '')}  target: {it.get('target', '')}"
            elif "cuisine" in it:
                line += f"  - {it.get('cuisine', '')} {it.get('price', '')}"
            _mc(line)
            if "why" in it:
                _mc(f"  {it['why']}", h=5, size=9)
        pdf.ln(2)
    pdf.ln(4)
    _mc(
        "Generated by Diet & Workout Recommendation (Streamlit + NVIDIA NIM). "
        "Not a substitute for professional medical advice.",
        h=5,
        size=8,
    )
    return bytes(pdf.output())