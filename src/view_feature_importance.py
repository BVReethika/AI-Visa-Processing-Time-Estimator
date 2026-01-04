import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")

feature_names = [
    "applicant_country",
    "visa_type",
    "processing_center",
    "visa_status",
    "application_month"
]

importance = model.feature_importances_

df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(df)
