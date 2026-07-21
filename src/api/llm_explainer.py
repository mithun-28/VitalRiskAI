import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")
def generate_explanation(patient_data,probability,stroke_risk):
    risk_level = (
        "HIGH"
        if probability >= 0.60
        else "MODERATE"
        if probability >= 0.30
        else "LOW"
    )

    prompt = f"""
You are a healthcare risk explanation assistant.

Patient Details:
{patient_data}

Prediction:
Stroke Risk = {stroke_risk}
Probability = {probability:.2%}
Risk Level = {risk_level}

Generate a response with:

1. Simple explanation
2. Likely contributing factors
3. Preventive measures
4. Reassuring and professional tone
5. Do not diagnose disease
6. Mention this is only an AI prediction

Keep it under 150 words.
"""
    response = model.generate_content(prompt)

    return response.text