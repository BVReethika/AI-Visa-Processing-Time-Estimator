import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load processed data
DATA_PATH = "data/processed/visa_cleaned_data.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Encode categorical columns
# ---------------------------
categorical_cols = [
    "applicant_country",
    "visa_type",
    "processing_center",
    "visa_status"
]

encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Extract month from application_date (if present)
if "application_date" in df.columns:
    df["application_date"] = pd.to_datetime(df["application_date"])
    df["application_month"] = df["application_date"].dt.month
    df.drop(columns=["application_date"], inplace=True)

# Drop non-useful columns
drop_cols = ["application_id", "decision_date"]
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ---------------------------
# Split features & target
# ---------------------------
X = df.drop(columns=["processing_time_days"])
y = df["processing_time_days"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Random Forest Model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Model Performance")
print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")
print(f"R²   : {r2:.2f}")

# ---------------------------
# Save model
# ---------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_model.pkl")

print("Model saved successfully")

