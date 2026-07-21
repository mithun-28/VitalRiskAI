import joblib
import pandas as pd
import shap

pipeline = joblib.load("models/stroke_pipeline.pkl")
background_data = joblib.load("models/background_data.pkl")
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]
background_processed = preprocessor.transform(background_data)
explainer = shap.LinearExplainer(model,background_processed)
def clean_feature_name(feature_name):
    feature_name = feature_name.replace("", "")
    feature_name = feature_name.replace("", "")

    feature_name = feature_name.replace("smoking_status_smokes","Smoking Status = Smokes")
    feature_name = feature_name.replace("smoking_status_formerly smoked","Smoking Status = Formerly Smoked")
    feature_name = feature_name.replace("smoking_status_never smoked","Smoking Status = Never Smoked")
    feature_name = feature_name.replace("ever_married_Yes","Married")
    feature_name = feature_name.replace("ever_married_No","Not Married")
    feature_name = feature_name.replace("Residence_type_Urban","Urban Residence")
    feature_name = feature_name.replace("Residence_type_Rural","Rural Residence")
    feature_name = feature_name.replace("_"," ")
    return feature_name.title()


def generate_explanation(probability, top_factors):

    factor_names = [
        factor["feature"]
        for factor in top_factors
    ]

    if probability < 0.30:

        risk_level = "LOW"

        recommendation = (
            "Your current stroke risk appears low. "
            "Continue maintaining a healthy lifestyle, "
            "exercise regularly, eat a balanced diet, "
            "and attend routine health checkups."
        )

    elif probability < 0.60:

        risk_level = "MODERATE"

        recommendation = (
            "Your stroke risk appears moderate. "
            "Consider monitoring blood pressure, "
            "maintaining healthy glucose levels, "
            "improving physical activity, and "
            "consulting a healthcare professional if needed."
        )

    else:
        risk_level = "HIGH"
        recommendation = (
            "Your stroke risk appears high. "
            "Medical consultation is strongly recommended. "
            "Carefully monitor blood pressure, blood sugar, "
            "weight management, smoking habits, and overall cardiovascular health."
        )

    explanation = (
        f"Risk Level: {risk_level}. "
        f"The most influential factors were: "
        f"{', '.join(factor_names)}. "
        f"{recommendation}"
    )
    return explanation

def PredictStroke(data):
    input_df = pd.DataFrame([data])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    processed_input = preprocessor.transform(input_df)
    shap_values = explainer.shap_values(processed_input)
    feature_names = preprocessor.get_feature_names_out()
    importance = list(zip(feature_names,shap_values[0]))
    importance.sort(key=lambda x: abs(x[1]),reverse=True)
    top_factors = []
    for feature, value in importance[:5]:
        top_factors.append({"feature": clean_feature_name(feature),"impact": round(float(value), 4)})
    explanation = generate_explanation(probability,top_factors)
    return {"stroke_risk": int(prediction),"probability": round(float(probability), 4),"top_factors": top_factors,"explanation": explanation}