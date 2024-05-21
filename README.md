# Diet-and-workout-Recommendation-using-Google-Gemini-pro
Elevate your health journey with our Diet &amp; Workout Recommendation System on Google Gemini Pro! Personalized suggestions based on age, Gender, height, weight, region, dietary preferences, allergies, and health conditions. Optimize your well-being effortlessly!

## Key Features

- **Personalized recommendations:** Generates diet and workout plans customized to individual needs and preferences.
- **AI-powered insights:** Utilizes Google Gemini Pro's advanced language capabilities to provide comprehensive and informative recommendations.
- **User-friendly interface:** Streamlined interaction through a simple and intuitive interface.

## Technologies Used

- **Streamlit:** Framework for building web applications in Python.
- **Google Gemini Pro API:** Access to Google AI's text-generating capabilities.
- **Langchain:** Library for interacting with Google AI models.

## Flowchart: Using Google Gemini Pro API Key in Diet and Workout Recommendation Project

**Start**

**--> User inputs preferences** - Age - Weight -Food type - Gender - Veg or Non-Veg - Region - State

**--> Prepare request for Gemini Pro API** - Format user input into structured API request - Include prompts and parameters as needed

**--> Send request to Gemini Pro API** - Use Langchain library or other method - Submit request with API key

**--> Receive response from Gemini Pro API** - API processes request - Generates text output with recommendations

**--> Parse and format response** - Extract relevant information: - Food suggestions - Workout - Fitness tips.

**--> Present recommendations to user** - Display information in Streamlit interface

**--> (Optional) Offer additional functionalities** - Adjust preferences - Refine recommendations - Track progress - Access other diet/workout features

**End**


## Demo

![88bc7558-2a0e-41cf-98db-594cca9cbcaa](https://github.com/m-rishab/Diet-and-workout-Recommendation-using-Google-Gemini-pro/assets/113618652/d70340d6-e32b-486d-b7cb-dfaa17b62d7a)

## Live Project

[https://diet-and-workout-recommendation-using-app-gemini-pro.streamlit.app/](https://diet-and-workout-recommendation-using-app-gemini-pro.streamlit.app/)

## Setup and Usage

1. Obtain a Google Gemini Pro API key.
2. **Install required libraries:**

   ```bash
   pip install "library"

## Run the Streamlit app:

  ````Bash
  streamlit run app.py   
