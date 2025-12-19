import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/visa_raw_data.csv")

# Convert date columns
df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

# Remove records with missing decision date
df = df.dropna(subset=["decision_date"])

# Create target variable
df["processing_time_days"] = (
    df["decision_date"] - df["application_date"]
).dt.days

# Remove invalid values
df = df[df["processing_time_days"] > 0]

# Save cleaned dataset
df.to_csv("data/processed/visa_cleaned_data.csv", index=False)

print("Milestone 1 preprocessing completed successfully")
