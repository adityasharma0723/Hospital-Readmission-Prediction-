import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import importlib
if 'src.predict' in sys.modules:
    importlib.reload(sys.modules['src.predict'])
from src.predict import predict_readmission

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0rem !important;
        margin-top: -3rem !important;
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-size: 20px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2A3C 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid #3A3A4C;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(#eee, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Readmission Prediction")
st.markdown("<h5 style='text-align: center; color: #a0a0b0;'><i>An AI-powered tool combining structured clinical data with NLP text analysis to predict 30-day hospital readmission risk.</i></h5>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📋 Patient Demographics & Hospital Stay Details", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Demographics**")
        age = st.number_input("Age", min_value=0, max_value=120, value=72)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Newborn", "Not Available", "Trauma Center"])

    with col2:
        st.markdown("**🛏️ Current Stay**")
        time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=14, value=8)
        num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=3)
        num_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=18)
        
    with col3:
        st.markdown("**🩺 Medical History**")
        num_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=9)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=200, value=65)
        st.markdown("**🔄 Prior Visits (Past Year)**")
        col3a, col3b, col3c = st.columns(3)
        number_emergency = col3a.number_input("Emergency", min_value=0, max_value=50, value=2)
        number_inpatient = col3b.number_input("Inpatient", min_value=0, max_value=50, value=3)
        number_outpatient = col3c.number_input("Outpatient", min_value=0, max_value=50, value=5)

st.markdown("<h3>✍️ Clinical Diagnosis Notes</h3>", unsafe_allow_html=True)
sample_text = (
    "Patient diagnosed with congestive heart failure and type 2 diabetes. "
    "Presenting symptoms include shortness of breath and chest pain. "
    "Multiple comorbidities complicate treatment plan. "
    "Previous admission within 30 days noted."
)
diagnosis_text = st.text_area("Enter raw clinical notes or doctor's evaluation here:", value=sample_text, height=140)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🚀 Analyze & Predict Readmission Risk", type="primary")

if predict_clicked:
    st.markdown("---")
    with st.spinner("🧠 AI Models analyzing patient data and clinical notes..."):
        try:
            patient_data = {
                "age": age,
                "gender": gender,
                "admission_type": admission_type,
                "num_medications": num_medications,
                "num_procedures": num_procedures,
                "num_diagnoses": num_diagnoses,
                "time_in_hospital": time_in_hospital,
                "num_lab_procedures": num_lab_procedures,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient,
                "number_outpatient": number_outpatient,
            }
            
            result = predict_readmission(patient_data, diagnosis_text)
            
            st.markdown("<h2 style='text-align: center;'>📊 Prediction Results</h2>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric(label="Readmission Prediction", value=result["prediction"])
                st.markdown("</div>", unsafe_allow_html=True)
                
            prob = result["probability"]
            with res_col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric(label="Probability of Readmission", value=f"{prob:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            risk = result["risk_level"]
            
            if risk == "High":
                st.error(f"### ⚠️ Risk Level: **{risk}**")
                st.progress(prob)
                st.markdown("""
                > **Recommendation:** High risk of readmission within 30 days. Intensive post-discharge monitoring 
                and immediate follow-up appointment scheduling is highly recommended.
                """)
            elif risk == "Medium":
                st.warning(f"### ⚡ Risk Level: **{risk}**")
                st.progress(prob)
                st.markdown("""
                > **Recommendation:** Medium risk of readmission. Ensure standard follow-up protocol is strictly 
                adhered to, with an outbound check-in call scheduled 48 hours post-discharge.
                """)
            else:
                st.success(f"### ✅ Risk Level: **{risk}**")
                st.progress(prob)
                st.markdown("""
                > **Recommendation:** Low risk of readmission. Standard discharge procedures apply. 
                Patient exhibits stable markers.
                """)
                
        except FileNotFoundError as e:
            st.error(f"Model artifacts missing: {str(e)}\nPlease run `python main.py` first to train and save the models.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
