# Diet-and-workout-Recommendation-using-Google-Gemini-pro

Elevate your health journey with our Diet &amp; Workout Recommendation System powered by **NVIDIA NIM** (Nemotron 120B)! Personalized suggestions based on age, gender, height, weight, region, dietary preferences, allergies, and health conditions. Optimize your well-being effortlessly!

## Key Features

- **Personalized recommendations:** Generates diet and workout plans customized to individual needs and preferences.
- **AI-powered insights:** Utilizes NVIDIA NIM's Nemotron-3-Super-120B language model to provide comprehensive and informative recommendations.
- **Modern UI:** Built with Reflex (Python → React) for a fast, interactive, and responsive interface.
- **Interactive charts:** 3D BMI visualization and API usage tracking via Plotly.

## Technologies Used

- **Reflex:** Python framework for building reactive web apps (React frontend, Python backend).
- **NVIDIA NIM:** Hosted inference for NVIDIA's Nemotron large language models (no local GPU needed).
- **Langchain:** Library for standardized interaction with the `ChatNVIDIA` model.

## Flowchart

**Start**

**--> User inputs preferences** - Age - Weight - Food type - Gender - Veg or Non-Veg - Region - State

**--> Prepare request for NVIDIA NIM API** - Format user input into structured API request - Include prompts and parameters as needed

**--> Send request to NVIDIA NIM API** - Use Langchain `ChatNVIDIA` - Submit request with NVIDIA API key

**--> Receive response from NVIDIA NIM API** - API processes request - Generates text output with recommendations

**--> Parse and format response** - Extract relevant information: - Food suggestions - Workout - Fitness tips

**--> Present recommendations to user** - Display information in the Reflex interface

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

3. Run the Reflex app:

   ```bash
   reflex run
   ```

   Open http://localhost:3000 in your browser.

## Deployment

Deploy to Reflex Cloud:

```bash
reflex login
reflex deploy
```

The NVIDIA API key must be provided via the `NVIDIA_API_KEY` environment variable (the app falls back to the key configured in `app/app.py`).