import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
import langchain.globals as lcg  # Import langchain.globals

# Set verbose to True or False based on your requirements
lcg.set_verbose(True)  # Enable verbose mode if needed

# Set up the model and prompt template
os.environ["GOOGLE_API_KEY"] = 'AIzaSyB0EZLaJdNmzVg-kfXMx2ljYNreoVxoudo'
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'state', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 6 restaurants names, 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person state: {state}\n"  # Include state in the prompt
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}."
)
chain_resto = LLMChain(llm=model, prompt=prompt_template_resto)

# Custom styling
st.markdown(
    """
    <style>
        .title {
            font-size: 25px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        .content {
            font-family: 'Helvetica', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a Streamlit web app
st.title('Diet and Workout Recommendation Using Google Gemini-Pro')

# User input form
age = st.text_input('Age')
gender = st.selectbox('Gender', ['Male', 'Female'])
weight = st.text_input('Weight (kg)')
height = st.text_input('Height (cm)')
veg_or_nonveg = st.selectbox('Veg or Non-Veg', ['Veg', 'Non-Veg'])
disease = st.text_input('Disease')
region = st.text_input('Region')
state = st.text_input('State')
allergics = st.text_input('Allergics')
foodtype = st.text_input('Food Type')

# Button to trigger recommendations
if st.button('Get Recommendations'):
    # Check if all form fields are filled
    if age and gender and weight and height and veg_or_nonveg and disease and region and state and allergics and foodtype:
        input_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'veg_or_nonveg': veg_or_nonveg,
            'disease': disease,
            'region': region,
            'state': state,  # Include state in input_data
            'allergics': allergics,
            'foodtype': foodtype
        }

        results = chain_resto.invoke(input_data)

        # Extract recommendations
        results_text = results['text']
        st.write("Generated Recommendations:")
        st.write(results_text)
    else:
        st.write("Sorry, you did not provide any information. Please fill in all the form fields.")
