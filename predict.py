import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/titanic_model.pkl")

print("Model Loaded Successfully!")


# Prediction function
def predict_survival(passenger_data):

    prediction = model.predict(passenger_data)

    probability = model.predict_proba(passenger_data)

    return prediction[0], probability[0]