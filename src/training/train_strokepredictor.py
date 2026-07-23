import os
import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pipeline import get_preprocessor

print(f"Scikit-Learn Version: {sklearn.__version__}")

data = pd.read_csv( r"C:\Users\slpmi\OneDrive\Desktop\Healthcare\data\StrokeData_Unprocessed.csv")
X = data.drop(columns=["stroke"])
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
os.makedirs("models", exist_ok=True)

background_data = X_train.sample(n=min(100, len(X_train)),random_state=42)
joblib.dump(background_data,"models/background_data.pkl")
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count
print(f"\nScale Pos Weight: {scale_pos_weight:.2f}\n")

models = {"logistic_regression": LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42),
          "random_forest": RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1),
          "lightgbm": LGBMClassifier(n_estimators=200,class_weight="balanced",random_state=42,verbose=-1),
          "xgboost": XGBClassifier(n_estimators=200,random_state=42,scale_pos_weight=scale_pos_weight,eval_metric="logloss")}

def evaluate_model(model_name, y_true, y_pred, y_prob):
    return {"model": model_name,"accuracy": accuracy_score(y_true, y_pred),"precision": precision_score(y_true, y_pred, zero_division=0),"recall": recall_score(y_true, y_pred, zero_division=0),"f1": f1_score(y_true, y_pred, zero_division=0),"roc_auc": roc_auc_score(y_true, y_prob)}

results = []

best_pipeline = None
best_auc = 0
best_model_name = ""

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    pipeline = Pipeline([("preprocessor", get_preprocessor()),("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(model_name,y_test,y_pred,y_prob)
    results.append(metrics)

    print(classification_report(y_test,y_pred,zero_division=0))
    if metrics["roc_auc"] > best_auc:
        best_auc = metrics["roc_auc"]
        best_pipeline = pipeline
        best_model_name = model_name
results_df = pd.DataFrame(results)

print("\nModel Comparison:\n")
print(results_df.sort_values(by="roc_auc",ascending=False))
joblib.dump(best_pipeline,"models/stroke_pipeline.pkl")

print(f"\nBest Model Saved: {best_model_name}")
print(f"Best ROC-AUC: {best_auc:.4f}")
loaded_pipeline = joblib.load("models/stroke_pipeline.pkl")
loaded_model = loaded_pipeline.named_steps["model"]
print("\nVerification:")
print(type(loaded_model))
print("Has multi_class:", hasattr(loaded_model, "multi_class"))
sample_prediction = loaded_pipeline.predict(X_test.head())
print("Sample Predictions:", sample_prediction)