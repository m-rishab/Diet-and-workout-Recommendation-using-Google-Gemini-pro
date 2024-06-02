import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
import plotly.graph_objs as go
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import langchain.globals as lcg

# Set verbose to True or False based on your requirements
lcg.set_verbose(True)  # Enable verbose mode if needed

# Set up the model and prompt template
os.environ["GOOGLE_API_KEY"] = 'AIzaSyA9WppajNxtoEErBLRkF1pisIJ8Ajs2rfY'
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template_resto = PromptTemplate(
    input_variables=['name', 'age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'state', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 6 restaurant names, 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person name: {name}\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person state: {state}\n"
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}."
)

# Create a Runnable chain
chain_resto = RunnableLambda(lambda inputs: prompt_template_resto.format(**inputs)) | model

# Usage Tracking
usage_data = []

# Initialize session state for usage tracking
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = []

# Capture Usage Data
def capture_usage():
    st.session_state.timestamp.append(datetime.now())
    usage_data.append({'timestamp': st.session_state.timestamp[-1]})

# Create Weekly Usage Tracking Graph
def create_usage_tracking_graph():
    # Generate dates from January 2024 to today
    start_date = date(2024, 1, 1)
    end_date = date.today()
    date_range = pd.date_range(start_date, end_date, freq='W-MON')

    # Placeholder data for demonstration
    # Replace with actual usage data aggregation logic
    weeks = [date.strftime("%Y-%m-%d") for date in date_range]
    usage_count = [10, 20, 15, 25, 30, 5, 10, 8, 15, 18, 8, 15, 10, 20, 22]  # Placeholder counts

    fig = go.Figure(data=[go.Bar(
        x=weeks,
        y=usage_count,
        text=usage_count,
        textposition='auto',
    )])

    fig.update_layout(
        title='Weekly Usage Tracking',
        xaxis_title='Week',
        yaxis_title='Usage Count',
    )

    st.plotly_chart(fig)

# Custom styling with background image
st.markdown(
    f"""
    <style>
        .title {{
            font-size: 32px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 20px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
        }}
        .subtitle {{
            font-size: 20px;
            font-family: 'Helvetica', sans-serif;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 30px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
        }}
        .form-label {{
            font-weight: bold;
            color: #FFFFFF;
        }}
        .form-input {{
            margin-bottom: 15px;
        }}
        .recommendations {{
            margin-top: 20px;
            font-family: 'Helvetica', sans-serif;
        }}
        .bmi {{
            font-size: 20px;
            font-weight: bold;
            color: #FFFFFF;
            margin-top: 10px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
        }}
        .stButton button {{
            transition: background-color 0.3s, transform 0.3s;
        }}
        .stButton button:hover {{
            background-color: #4CAF50;
            transform: scale(1.05);
        }}
        .form-container {{
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a Streamlit web app
st.markdown('<div class="title">Diet and Workout Recommendation Using Google Gemini-Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personalized recommendations based on your profile</div>', unsafe_allow_html=True)

# User input form
with st.form(key='user_input_form', clear_on_submit=True):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown('<div class="form-label">Name:</div>', unsafe_allow_html=True)
    name = st.text_input('Name', key='name', placeholder='Enter your name', label_visibility='hidden')

    st.markdown('<div class="form-label">Age:</div>', unsafe_allow_html=True)
    age = st.text_input('Age', key='age', placeholder='Enter your age', label_visibility='hidden')

    st.markdown('<div class="form-label">Gender:</div>', unsafe_allow_html=True)
    gender = st.selectbox('Gender', ['Male', 'Female'], key='gender', label_visibility='hidden')

    st.markdown('<div class="form-label">Weight (kg):</div>', unsafe_allow_html=True)
    weight = st.text_input('Weight (kg)', key='weight', placeholder='Enter your weight in kg', label_visibility='hidden')

    st.markdown('<div class="form-label">Height (cm):</div>', unsafe_allow_html=True)
    height = st.text_input('Height (cm)', key='height', placeholder='Enter your height in cm', label_visibility='hidden')

    st.markdown('<div class="form-label">Veg or Non-Veg:</div>', unsafe_allow_html=True)
    veg_or_nonveg = st.selectbox('Veg or Non-Veg', ['Veg', 'Non-Veg'], key='veg_or_nonveg', label_visibility='hidden')

    st.markdown('<div class="form-label">Disease:</div>', unsafe_allow_html=True)
    disease = st.text_input('Disease', key='disease', placeholder='Enter any generic disease', label_visibility='hidden')

    st.markdown('<div class="form-label">Region:</div>', unsafe_allow_html=True)
    region = st.text_input('Region', key='region', placeholder='Enter your region', label_visibility='hidden')

    st.markdown('<div class="form-label">State:</div>', unsafe_allow_html=True)
    state = st.text_input('State', key='state', placeholder='Enter your state', label_visibility='hidden')

    st.markdown('<div class="form-label">Allergics:</div>', unsafe_allow_html=True)
    allergics = st.text_input('Allergics', key='allergics', placeholder='Enter any allergies', label_visibility='hidden')

    st.markdown('<div class="form-label">Food Type:</div>', unsafe_allow_html=True)
    foodtype = st.text_input('Food Type', key='foodtype', placeholder='Enter your preferred food type', label_visibility='hidden')

    submit_button = st.form_submit_button(label='Get Recommendations')
    st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    # Check if all form fields are filled
    if all([name, age, gender, weight, height, veg_or_nonveg, disease, region, state, allergics, foodtype]):
        input_data = {
            'name': name,
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'veg_or_nonveg': veg_or_nonveg,
            'disease': disease,
            'region': region,
            'state': state,
            'allergics': allergics,
            'foodtype': foodtype
        }

        # Capture usage data
        capture_usage()

        # Run the chain and get the recommendations
        recommendations = chain_resto.invoke(input_data)

        st.markdown('<div class="subtitle">Recommendations:</div>', unsafe_allow_html=True)
        st.markdown('<div class="recommendations">', unsafe_allow_html=True)
        st.markdown(recommendations, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Calculate and display BMI
        height_m = float(height) / 100.0  # Convert height from cm to meters
        bmi = float(weight) / (height_m ** 2)

        # BMI categorization
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif 18.5 <= bmi < 24.9:
                return 'Normal weight'
            elif 25 <= bmi < 29.9:
                return 'Overweight'
            else:
                return 'Obesity'

        bmi_category = categorize_bmi(bmi)

        # Define color based on BMI category
        bmi_colors = {
            'Underweight': 'blue',
            'Normal weight': 'green',
            'Overweight': 'yellow',
            'Obesity': 'red'
        }
        color = bmi_colors[bmi_category]

        # Create enhanced 3D BMI visualization
        fig = go.Figure(data=[go.Scatter3d(
            x=[int(age)],
            y=[float(weight)],
            z=[bmi],
            mode='markers',
            marker=dict(
                size=12,
                color=color,
                opacity=0.8
            ),
            text=[f"Age: {age}<br>Weight: {weight} kg<br>BMI: {bmi:.2f}<br>Category: {bmi_category}"],
            hoverinfo='text'
        )])

        fig.update_layout(
            title='3D BMI Visualization',
            scene=dict(
                xaxis_title='Age',
                yaxis_title='Weight (kg)',
                zaxis_title='BMI',
                xaxis=dict(backgroundcolor="rgb(200, 200, 230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                yaxis=dict(backgroundcolor="rgb(230, 200,230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                zaxis=dict(backgroundcolor="rgb(230, 230,200)", gridcolor="white", showbackground=True, zerolinecolor="white"),
            ),
            margin=dict(r=10, l=10, b=10, t=30)
        )

        st.plotly_chart(fig)
        st.markdown(f'<div class="bmi">Your BMI is {bmi:.2f}, which falls under the category: {bmi_category}</div>', unsafe_allow_html=True)
    else:
        st.error("Please fill in all the form fields.")

# Render Weekly Usage Tracking Graph
st.markdown('<div class="subtitle">Weekly Usage Tracking</div>', unsafe_allow_html=True)
create_usage_tracking_graph()
