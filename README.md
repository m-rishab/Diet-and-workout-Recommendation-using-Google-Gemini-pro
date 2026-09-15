# Diet-and-workout-Recommendation

[Live](https://m-rishab-diet-and-workout-recommendatio-appstreamlit-app-zpz7af.streamlit.app/)

> **Note:** The repository name "(using-Google-Gemini-pro)" is **legacy** — this
> app is powered by **NVIDIA NIM** (Nemotron-3-Super-120B via LangChain), not
> Google Gemini.

Elevate your health journey with our Diet &amp; Workout Recommendation System powered by **NVIDIA NIM** (Nemotron 120B)! Personalized suggestions based on age, gender, height, weight, region, dietary preferences, allergies, and health conditions. Optimize your well-being effortlessly!

## Key Features

- **Personalized recommendations:** Generates diet and workout plans customized to individual needs and preferences.
- **AI-powered insights:** Utilizes NVIDIA NIM's Nemotron-3-Super-120B language model to provide comprehensive and informative recommendations.
- **Modern UI:** Built with Streamlit for a fast, interactive, and responsive interface.
- **Interactive charts:** BMI gauge, macro targets and BMI-over-time trend via Plotly.
- **Keyless images:** Real dish/exercise photos are searched at runtime through DuckDuckGo (primary) with Bing and Wikimedia Commons as fallbacks — no image API keys needed. Each result is relevance-scored (prepared-dish vs raw ingredient, protein-contradiction filter) and only downloaded after validation, then cached locally in `assets/images/`. When nothing trustworthy is found the app shows **no image** rather than a wrong one.

## Technologies Used

- **Streamlit:** Python framework for building interactive web apps.
- **NVIDIA NIM:** Hosted inference for NVIDIA's Nemotron large language models (no local GPU needed).
- **Langchain:** Library for standardized interaction with the `ChatNVIDIA` model.
- **SQLite:** Local persistence for saved plans and BMI history.

## Flowchart

**Start**

**--> User inputs preferences** - Age - Weight - Food type - Gender - Veg or Non-Veg - Region - State

**--> Prepare request for NVIDIA NIM API** - Format user input into structured API request - Include prompts and parameters as needed

**--> Send request to NVIDIA NIM API** - Use Langchain `ChatNVIDIA` - Submit request with NVIDIA API key

**--> Receive response from NVIDIA NIM API** - API processes request - Generates text output with recommendations

**--> Parse and format response** - Extract relevant information: - Food suggestions - Workout - Fitness tips

**--> Present recommendations to user** - Display information in the Streamlit interface

**--> (Optional) Offer additional functionalities** - Adjust preferences - Refine recommendations - Track progress - Access other diet/workout features

**End**

## Setup and Usage

1. Obtain a [NVIDIA NIM API key](https://build.nvidia.com) (free credits available).
2. Create a virtual environment and install required libraries:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
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

- `app/backend.py` — all logic: prompts, strict-JSON parsing, plan enrichment with
  keyless fallback catalogs, BMR/TDEE math, restaurant fallbacks, SQLite persistence,
  PDF export, and the image pipeline (query builders, DuckDuckGo/Bing/Wikimedia search,
  relevance scoring with contradiction filtering, download + validation, local caching).
- `app/streamlit_app.py` — the Streamlit UI (sidebar form, plan tabs, coach chat,
  insights, export).
- `.streamlit/config.toml` — dark theme.
