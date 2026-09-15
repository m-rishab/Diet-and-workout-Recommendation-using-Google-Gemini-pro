# Diet-and-workout-Recommendation

[Live](https://m-rishab-diet-and-workout-recommendatio-appstreamlit-app-zpz7af.streamlit.app/)

> **Note:** The repository name "(using-Google-Gemini-pro)" is **legacy** — this
> app is powered by **NVIDIA NIM** (`nvidia/nemotron-3-super-120b-a12b` via `ChatNVIDIA`/LangChain), not Google Gemini. See [Legacy notebook](#legacy-notebook) for the original Gemini demo.

Elevate your health journey with our Diet & Workout Recommendation System powered by **NVIDIA NIM** (Nemotron 120B)! Personalized suggestions based on age, gender, height, weight, region, dietary preferences, allergies, and health conditions. Optimize your well-being effortlessly!

## Key Features

- **Personalized recommendations:** Generates diet (breakfast/lunch/dinner) and workout plans customized to age, gender, weight, height, region/state, diet type, allergies and medical conditions.
- **AI-powered insights:** Uses `NVIDIA NIM` (`nvidia/nemotron-3-super-120b-a12b`) with a strict-JSON prompt and local safety guardrails (allergy/medical constraints + medical disclaimer) — output is parsed and normalized before display.
- **Local health math:** BMI, BMR (Mifflin-St Jeor), TDEE (`×1.55`) and daily calorie target computed locally (`app/backend.py:_compute_metrics`), not by the LLM.
- **Modern UI:** Built with Streamlit — sidebar form, 6 tabs (Meals / Restaurants / Workouts / Weekly Split / Insights / Ask the Coach), save/load and clear-history flows.
- **Interactive charts:** BMI bar/gauge, macro targets and BMI-over-time trend via Plotly.
- **Keyless images:** Real dish/exercise photos are searched at runtime through DuckDuckGo (primary, via `ddgs`) with Bing and Wikimedia Commons as fallbacks — no image API keys needed. Each result is relevance-scored (prepared-dish vs raw ingredient, protein-contradiction filter, `_ACCEPT_MIN_SCORE=55`) and only downloaded after validation (`Pillow` magic-bytes + dimensions), then cached locally in `assets/images/meals|workouts/` with `manifest.json`. When nothing trustworthy is found the app shows **no image** rather than a wrong one (`CORRECT > NONE > WRONG`).
- **Robust fallbacks & persistence:** Local catalogs pad incomplete LLM sections, state-aware restaurant fallbacks ensure "Restaurants nearby" is never empty, and `SQLite` (`nutrifit.db`) persists saved plans/BMI history with 24h cache reuse. Thread-pooled image fetching has deadlines so photos never block plan generation.
- **Export & coach:** Download plan as JSON/PDF (`fpdf2`) and ask follow-up questions via the Coach chat (plan JSON as context).

## Technologies Used

- **Streamlit `1.49.0`:** Interactive web app (`app/streamlit_app.py`).
- **NVIDIA NIM:** Hosted inference for `nvidia/nemotron-3-super-120b-a12b` (no local GPU) — `langchain-nvidia-ai-endpoints==1.4.3` / `langchain_core==1.6.3`.
- **LangChain `ChatNVIDIA`:** Strict-JSON prompting, retry on transient `503/504` and JSON-repair loop.
- **Plotly `7.0.0`:** BMI gauge/bar and BMI trend charts.
- **SQLite:** Local `nutrifit.db` (`planrow`, `bmirow`) for saved plans and BMI history.
- **fpdf2 `2.8.8`:** PDF export; **python-dotenv `1.2.3`:** `.env` loading for `NVAPI_KEY`.
- **Image pipeline (stdlib + optional):** `urllib` search/scrape, `ddgs` (DuckDuckGo), `Pillow` (image validation) — both optional with graceful fallbacks.

## How it Works / Flowchart

```
Start
  → User fills sidebar form (name, age, gender, weight/height (kg_cm or Imperial), diet, medical condition, region/state, allergies, food preference)
  → Input normalization + local metrics: BMI/BMI-category, BMR, TDEE, daily target (_compute_metrics)
  → Build strict-JSON prompt (_build_prompt) + safety constraints (allergy/medical disclaimer)
  → Check SQLite cache (cache_key = md5(sorted inputs), 24h TTL) — if hit, enrich & render immediately
  → Send to NVIDIA NIM via ChatNVIDIA.stream() with retries; repair loop if JSON invalid (_extract_json → _normalize_plan)
  → Enrich & pad (_enrich_plan): clean items, fill missing meals/workouts/weekly_split from local catalogs, state-aware restaurant fallbacks, ensure notes
  → Image pipeline (parallel, deadline-bound): build_meal/workout_image_queries → search_web_images (DDG → Bing → Wikimedia) → score (|contradiction|= -inf, _ACCEPT_MIN_SCORE 55) → download + _validate_image → cache under assets/images/meals|workouts/manifest.json
  → Save plan + metrics to SQLite (save_plan) and render Streamlit tabs; persist BMI history for Insights trend
  → (Optional) Coach chat (plan JSON as context), JSON/PDF export, load saved plans, BMI trend, clear history
End
```

## Setup and Usage

1. Obtain a [NVIDIA NIM API key](https://build.nvidia.com) (free credits available).
2. Create a virtual environment and install required libraries:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # optional but recommended for images:
   pip install ddgs Pillow
   ```

3. Create `.env` from the template and add your key (never commit `.env`):

   ```bash
   cp .env.example .env   # then edit .env with your NVAPI_KEY
   ```

4. Run the Streamlit app (dish/workout images are resolved automatically and
   cached under `assets/images/` — no manual download step needed):

   ```bash
   streamlit run app/streamlit_app.py
   ```

   Open the URL shown in the terminal (default http://localhost:8501).

## Project structure

```
.
├── app/
│   ├── backend.py          # prompts, strict-JSON parsing, _compute_metrics (Mifflin-St Jeor),
│   │                       # _enrich_plan + fallback catalogs, restaurant fallbacks,
│   │                       # SQLite persistence (nutrifit.db), PDF export (fpdf2),
│   │                       # image pipeline: query builders, DDG/Bing/Wikimedia search,
│   │                       # relevance scoring with contradiction filtering, download + validation, caching
│   └── streamlit_app.py    # Streamlit UI: sidebar form, plan tabs, coach chat, insights, export
├── assets/images/
│   ├── hero_*.jpg / restaurant_*.jpg / yoga_*.jpg ...  # static generic images
│   ├── meals/ / workouts/  # runtime-cached dish/exercise photos + manifest.json
│   └── manifest.json
├── scripts/fetch_images.py # optional bulk pre-fetch for images
├── .streamlit/config.toml  # dark theme (#0A0A0A / #C6FF3E)
├── .env.example            # template for NVAPI_KEY
├── requirements.txt        # pinned deps (streamlit, langchain-nvidia, plotly, fpdf2, dotenv)
├── nutrifit.db             # SQLite DB (gitignored, created at runtime)
└── Diet recommendation system.ipynb  # legacy Gemini demo (archived, see below)
```

- `app/backend.py` — all logic: prompts, strict-JSON parsing, plan enrichment with
  keyless fallback catalogs, BMR/TDEE math, restaurant fallbacks, SQLite persistence,
  PDF export, and the image pipeline (query builders, DuckDuckGo/Bing/Wikimedia search,
  relevance scoring with contradiction filtering, download + validation, local caching).
- `app/streamlit_app.py` — the Streamlit UI (sidebar form, plan tabs, coach chat,
  insights, export).
- `.streamlit/config.toml` — dark theme.

## Legacy notebook

`Diet recommendation system.ipynb` is the original prototype using `langchain_google_genai` (`GoogleGenerativeAI(model="gemini-pro")` with `LLMChain`/`PromptTemplate`). It is **archived and not used by the Streamlit app** — kept only for reference. It contains a hardcoded `GOOGLE_API_KEY` example; do not reuse that key.
