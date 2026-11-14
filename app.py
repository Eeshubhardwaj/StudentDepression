import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("model.pkl")
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Family_background",
    "Living_style",
    "Current_semester",
    "CGPA",
    "friend_circle",
    "Health_condition",
    "Cost_without_study",
    "Relationship_status_In a Relationship",
    "Relationship_status_Single",
    "Marital_status_Married",
    "Marital_status_Separated",
    "Marital_status_Unmarried",
]

st.set_page_config(page_title="Student Wellbeing Check", page_icon="📊", layout="wide")
st.title("Student Wellbeing Check")
st.write(
    "Enter the details below to get a model-generated depression-risk estimate. "
    "This tool is for educational purposes only and is not a diagnosis."
)

with st.form("student_details"):
    st.subheader("Personal and academic details")
    personal_col, academic_col = st.columns(2)

    with personal_col:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        family_background = st.selectbox(
            "Family background", ["Lower Middle class", "Middle class", "Rich class"]
        )
        living_style = st.selectbox("Living style", ["Bachelor", "Family"])
        relationship_status = st.selectbox(
            "Relationship status", ["Complicated", "In a Relationship", "Single"]
        )
        marital_status = st.selectbox("Marital status", ["Married", "Separated", "Unmarried"])

    with academic_col:
        current_semester = st.number_input(
            "Current semester", min_value=1, max_value=20, value=1, step=1
        )
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
        friend_circle = st.selectbox("Have a friend circle?", ["No", "Yes"])
        health_condition = st.selectbox(
            "Health condition", ["Good", "Normal", "Abnormal", "Critical"]
        )
        cost_without_study = st.number_input(
            "Monthly cost without study", min_value=0, value=10000, step=500
        )

    submitted = st.form_submit_button("Check estimate", type="primary")

if submitted:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    input_values = {
        "Gender": {"Male": 0, "Female": 1}[gender],
        "Age": age,
        "Family_background": {
            "Lower Middle class": 1,
            "Middle class": 2,
            "Rich class": 3,
        }[family_background],
        "Living_style": {"Bachelor": 0, "Family": 1}[living_style],
        "Current_semester": current_semester,
        "CGPA": cgpa,
        "friend_circle": {"No": 0, "Yes": 1}[friend_circle],
        "Health_condition": {"Good": 3, "Normal": 2, "Abnormal": 1, "Critical": 0}[
            health_condition
        ],
        "Cost_without_study": cost_without_study,
    }

    for option in ["Complicated", "In a Relationship", "Single"]:
        input_values[f"Relationship_status_{option}"] = int(relationship_status == option)
    for option in ["Married", "Separated", "Unmarried"]:
        input_values[f"Marital_status_{option}"] = int(marital_status == option)

    model_features = getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    missing_features = set(model_features) - input_values.keys()
    unexpected_features = input_values.keys() - set(model_features)
    if missing_features or unexpected_features:
        st.error("The saved model features do not match the notebook preprocessing.")
        st.stop()

    input_data = pd.DataFrame([[input_values[name] for name in model_features]], columns=model_features)
    prediction = int(model.predict(input_data)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if prediction in classes:
            probability = float(model.predict_proba(input_data)[0][classes.index(prediction)])

    st.subheader("Model estimate")
    if prediction == 1:
        st.warning("The model detected indicators associated with depression.")
    else:
        st.success("The model did not detect indicators associated with depression.")

    if probability is not None:
        st.metric("Model confidence for this estimate", f"{probability:.0%}")

    st.caption(
        "A prediction cannot determine whether someone has depression. "
        "For support, speak with a qualified mental-health professional."
    )