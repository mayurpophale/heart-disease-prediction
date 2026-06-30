import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color:#f8f9fa;
}

h1{
    color:#e63946;
    text-align:center;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
    background-color:#e63946;
    color:white;
}

.stButton>button:hover{
    background-color:#c1121f;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Files ---------------- #

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ---------------- Sidebar ---------------- #

st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("---")

st.sidebar.write("""
### About

This application predicts the likelihood of Heart Disease using a Machine Learning model.

**Algorithm:** KNN

**Developer:** Mayur Pophale
""")

st.sidebar.markdown("---")

st.sidebar.success("Version 1.0")

# ---------------- Main Title ---------------- #

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Fill in the patient's medical information below and click **Predict**."
)

st.divider()

# ---------------- Input ---------------- #

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age",18,100,45)

    sex = st.selectbox(
        "Sex",
        ["Male","Female"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","TA","ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=700,
        value=200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar (>120 mg/dL)",
        [0,1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y","N"]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.0,
        1.0,
        0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"]
    )

st.divider()

predict = st.button("🔍 Predict Heart Disease")

# ---------------- Prediction ---------------- #

if predict:

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
    }

    raw_input[f"Sex_{sex}"] = 1
    raw_input[f"ChestPainType_{chest_pain}"] = 1
    raw_input[f"RestingECG_{resting_ecg}"] = 1
    raw_input[f"ExerciseAngina_{exercise_angina}"] = 1
    raw_input[f"ST_Slope_{st_slope}"] = 1

    input_df = pd.DataFrame([raw_input])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # Probability

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(scaled_input)[0][1]

        st.write("### Risk Probability")

        st.progress(int(probability * 100))

        st.metric(
            "Probability",
            f"{probability*100:.2f}%"
        )

    st.divider()

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Feature":[
            "Age",
            "Sex",
            "Chest Pain",
            "Resting BP",
            "Cholesterol",
            "Fasting BS",
            "Resting ECG",
            "Maximum HR",
            "Exercise Angina",
            "Oldpeak",
            "ST Slope"
        ],
        "Value":[
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope
        ]
    })

    st.table(summary)

    st.divider()

    if prediction == 1:

        st.warning("""
### Recommendations

• Consult a Cardiologist

• Exercise Regularly

• Maintain Healthy Diet

• Control Blood Pressure

• Monitor Cholesterol

• Avoid Smoking
""")

    else:

        st.success("""
### Recommendations

• Maintain Healthy Lifestyle

• Exercise Regularly

• Eat Balanced Diet

• Regular Health Checkups
""")

# ---------------- Footer ---------------- #

st.divider()

st.markdown(
"""
<center>

Made with ❤️ by <b>Mayur Pophale</b>

Government College of Engineering, Aurangabad

</center>
""",
unsafe_allow_html=True
)