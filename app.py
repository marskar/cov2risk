import numpy as np
import streamlit as st
from joblib import load

gender = st.sidebar.selectbox("Choose the Gender", ["Female", "Male"])
age = st.sidebar.number_input("Select the Age", min_value=0, value=38)
continent = st.sidebar.selectbox(
    "Choose the Continent",
    [
        "Africa",
        "Asia",
        "North America",
        "South America",
        "Australia",
        "Oceania"
    ]
)

logreg = load(
    "models/2020-10-21_vcf_logistic-regression-model_age-gender-region.pickle"
)

amms = load("models/age_min-max-scaler.pickle")

gender_map = {"Female": 0, "Male": 1}

continent_map = {
    "Africa": [0, 0, 0, 0, 0],
    "Asia": np.array([1, 0, 0, 0, 0]),
    "North America": [0, 1, 0, 0, 0],
    "South America": [0, 0, 1, 0, 0],
    "Australia": [0, 0, 0, 1, 0],
    "Oceania": [0, 0, 0, 0, 1]
}

scaled_age = amms.transform([[age]])

X = np.append(scaled_age, gender_map[gender])
X = np.append(X, continent_map[continent])
st.text(
    f"The predicted risk of severe outcome is {logreg.predict_proba([X])[0][1]:.2f}."
)
