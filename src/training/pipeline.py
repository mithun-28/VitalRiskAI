from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

categorical_features = ["gender","ever_married","work_type","Residence_type","smoking_status"]
numerical_features = ["age","avg_glucose_level","bmi","heart_disease","hypertension"]

num_imputation_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler",StandardScaler())])
cat_imputation_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encode", OneHotEncoder(handle_unknown= "ignore"))])

preprocess_pipeline = ColumnTransformer([("num", num_imputation_pipeline, numerical_features), ("cat", cat_imputation_pipeline, categorical_features)])


def get_preprocessor():
    return preprocess_pipeline