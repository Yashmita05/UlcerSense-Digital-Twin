import joblib
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load Tabular Data
df = pd.read_csv("digital_twin_patient_data.csv")

feature_cols = ["toes_temp", "midfoot_temp", "heel_temp", "age", "hba1c", "diabetes_years"]
X = df[feature_cols]
y = df["ulcer_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    scale_pos_weight=(len(y_train) - sum(y_train)) / max(sum(y_train), 1),
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Initialize and save TreeExplainer
explainer = shap.TreeExplainer(model)

joblib.dump(model, "xgb_ulcer_model.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
joblib.dump(feature_cols, "feature_names.pkl")
print("Model, SHAP explainer, and feature metadata saved successfully.")