import joblib

model = joblib.load("models/random_forest_model.pkl")

print("Model loaded successfully!")
print(model)
