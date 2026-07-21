import json
import requests
import streamlit as st


st.set_page_config(page_title="VitalRiskAI",layout="centered")
st.title("VitalRiskAI")
with st.form("stroke_form"):

    gender = st.selectbox("Gender",["Male", "Female"])
    age = st.number_input("Age",min_value=0.0,max_value=120.0)
    hypertension = st.selectbox("Hypertension",[0, 1])
    heart_disease = st.selectbox("Heart Disease",[0, 1])
    ever_married = st.selectbox("Ever Married",["Yes", "No"])
    work_type = st.selectbox("Work Type",["Private","Self-employed","Govt_job","children","Never_worked"])
    residence_type = st.selectbox("Residence Type",["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level",min_value=0.0)
    bmi = st.number_input("BMI",min_value=0.0)
    smoking_status = st.selectbox("Smoking Status",["never smoked","formerly smoked","smokes","Unknown"])
    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}
    
    with open("app/patient_inputs/patient_input.json","w") as file:
        json.dump(input_data,file,indent=4)


        response = requests.post("https://vitalriskai.onrender.com",json=input_data)

    if response.status_code == 200:
        result = response.json()
        prob = result["probability"]
        if result["stroke_risk"] == 1:
            st.error("Stroke Risk Detected")
        else:
            st.success("No Immediate Stroke Risk Detected")
        st.write(f"Probability : {prob:.2%}")
        if prob < 0.30:
            st.success("Low Stroke Risk")
        elif prob < 0.60:
            st.warning("Moderate Stroke Risk")
        else:
            st.error("High Stroke Risk")
        st.markdown("### AI Health Explanation")
        st.write(result["explanation"])
    else:
        st.error("Prediction API failed")