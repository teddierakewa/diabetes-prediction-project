# diabetes_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load CSV dataset
# -----------------------------
csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
data = pd.read_csv(csv_path)

# Features and target
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = data['Outcome']

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Save model for later use
    joblib.dump(model, os.path.join(os.path.dirname(__file__), "diabetes_model.joblib"))
    return model, acc

model, accuracy = train_model(X, y)

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="💉",
    layout="centered",
)

# Dark mode styling
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stNumberInput>div>input { background-color: #1C1F26; color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App title and description
# -----------------------------
st.title("💉 Diabetes Prediction App")
st.write(f"Model trained with accuracy: {accuracy*100:.2f}%")
st.write("Enter your health details below:")

# -----------------------------
# Input fields
# -----------------------------
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, format="%.2f")
age = st.number_input("Age", 0, 120, 30)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of diabetes!")
    else:
        st.success("✅ Low risk of diabetes!")
