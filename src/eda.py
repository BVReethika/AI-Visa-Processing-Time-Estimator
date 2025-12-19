import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/visa_cleaned_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.describe())


import matplotlib.pyplot as plt

plt.plot([1,2,3],[4,5,6])
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/visa_cleaned_data.csv")

#  Distribution of Processing Time
plt.figure()
sns.histplot(df["processing_time_days"], kde=True)
plt.title("Distribution of Visa Processing Time")
plt.xlabel("Processing Time (Days)")
plt.ylabel("Frequency")
plt.show()


# Processing Time by Visa Type
plt.figure()
sns.boxplot(x="visa_type", y="processing_time_days", data=df)
plt.title("Processing Time by Visa Type")
plt.xlabel("Visa Type")
plt.ylabel("Processing Time (Days)")
plt.xticks(rotation=30)
plt.show()

#  Processing Time by Applicant Country
plt.figure()
sns.boxplot(x="applicant_country", y="processing_time_days", data=df)
plt.title("Processing Time by Applicant Country")
plt.xlabel("Applicant Country")
plt.ylabel("Processing Time (Days)")
plt.xticks(rotation=45)
plt.show()

# Seasonal Trend Analysis
df["application_date"] = pd.to_datetime(df["application_date"])
df["month"] = df["application_date"].dt.month

plt.figure()
sns.boxplot(x="month", y="processing_time_days", data=df)
plt.title("Seasonal Impact on Visa Processing Time")
plt.xlabel("Application Month")
plt.ylabel("Processing Time (Days)")
plt.show()

# Processing Time by Processing Center
plt.figure()
sns.boxplot(x="processing_center", y="processing_time_days", data=df)
plt.title("Processing Time by Processing Center")
plt.xlabel("Processing Center")
plt.ylabel("Processing Time (Days)")
plt.xticks(rotation=30)
plt.show()

# Feature Importance Analysis
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

encoded_df = df.copy()
le = LabelEncoder()

for col in ["applicant_country", "visa_type", "processing_center"]:
    encoded_df[col] = le.fit_transform(encoded_df[col])

X = encoded_df[["applicant_country", "visa_type", "processing_center"]]
y = encoded_df["processing_time_days"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance for Visa Processing Time")
plt.show()

importance_df.to_csv("data/processed/feature_importance.csv", index=False)




