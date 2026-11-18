
import streamlit as st
import pandas as pd
import joblib
import json

# Load saved model
model = joblib.load("student_performance_model.pkl")

# Load feature columns
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Load input settings
with open("input_settings.json", "r") as f:
    settings = json.load(f)

df_options = settings["df_options"]
df_min = settings["df_min"]
df_max = settings["df_max"]
df_median = settings["df_median"]

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Application title
st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter student details below to predict academic performance."
)

st.divider()

st.subheader("Student Details")

categorical_columns = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering"
]

input_data = {}

# Generate input fields
for col in feature_columns:

    if col in categorical_columns:
        options = df_options[col]

        input_data[col] = st.selectbox(
            col,
            options=options,
            format_func=lambda x: str(x)
        )

    else:
        minimum = float(df_min[col])
        maximum = float(df_max[col])
        default = float(df_median[col])

        input_data[col] = st.number_input(
            col,
            min_value=minimum,
            max_value=maximum,
            value=default
        )

st.divider()

# Prediction button
if st.button("Predict Student Performance", type="primary"):

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    prediction = model.predict(input_df)

    st.success(
        f"Predicted Grade Class: {int(prediction[0])}"
    )

    st.info(
        "Prediction completed successfully!"
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        st.subheader("Prediction Confidence")

        for grade, probability in zip(
            model.classes_, probabilities
        ):
            st.write(
                f"Grade Class {grade}: {probability * 100:.2f}%"
            )

st.divider()

st.caption("Student Performance Prediction System | Machine Learning Project")
