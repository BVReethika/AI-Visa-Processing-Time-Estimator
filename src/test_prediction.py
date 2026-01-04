import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")

sample_input = pd.DataFrame([{
    "applicant_country": 2,
    "visa_type": 1,
    "processing_center": 3,
    "visa_status": 0,
    "application_month": 5
}])

prediction = model.predict(sample_input)

print(f"Predicted Processing Time: {int(prediction[0])} days")
