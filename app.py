import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load machine learning model and dataset
@st.cache_resource
def load_model():
    return pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Car.csv')

model = load_model()
car = load_data()

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Predictor")
st.write("Enter the vehicle specifications below to calculate the estimated resale price.")

# Company Selection
companies = sorted(car['company'].unique())
selected_company = st.selectbox("Select Company", companies)

# Dynamic model filtering based on chosen company
filtered_models = sorted(car[car['company'] == selected_company]['name'].unique())
selected_model = st.selectbox("Select Model", filtered_models)

# Additional inputs
years = sorted(car['year'].unique(), reverse=True)
selected_year = st.selectbox("Select Year of Purchase", years)

fuel_types = car['fuel_type'].unique()
selected_fuel = st.selectbox("Select Fuel Type", fuel_types)

kms_driven = st.number_input("Select Number of Kms Travelled", min_value=0, value=10000, step=1000)

# Prediction trigger
if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame(
        [[selected_model, selected_company, selected_year, selected_fuel, kms_driven]],
        columns=['name', 'company', 'year', 'fuel_type', 'kms_driven']
    )
    prediction = model.predict(input_df)
    st.success(f"Estimated Resale Price: ₹{np.round(prediction[0], 2):,}")