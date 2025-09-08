import numpy as np
import pandas as pd
from joblib import load

# Load saved models & scaler
cls_model = load("models/occurrence_rf.joblib")
reg_model = load("models/frp_rf.joblib")
scaler    = load("models/feature_scaler.joblib")

# Define features in the correct order (same as training)
FEATURES = [
    'fire_weather_index',
    'humidity_min',
    'temp_mean',
    'temp_range',
    'wind_speed_max',
    'dewpoint_mean',
    'cloud_cover_mean',
    'evapotranspiration_total'
]

def predict_fire(input_data):
    """
    input_data: dict with keys matching FEATURES
    Example:
    {
        "fire_weather_index": 6.2,
        "humidity_min": 30,
        "temp_mean": 25.0,
        "temp_range": 12.0,
        "wind_speed_max": 15.0,
        "dewpoint_mean": 10.0,
        "cloud_cover_mean": 20.0,
        "evapotranspiration_total": 5.0
    }
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([input_data], columns=FEATURES)

    # Scale
    X_scaled = scaler.transform(df)

    # Predict fire occurrence
    fire_occurrence = cls_model.predict(X_scaled)[0]
    fire_prob = cls_model.predict_proba(X_scaled)[0][1]

    # Predict fire intensity (FRP)
    frp_pred = reg_model.predict(X_scaled)[0]

    return {
    "fire_occurrence": int(fire_occurrence),
    "fire_probability": round(float(fire_prob), 3),
    "predicted_frp": round(float(frp_pred), 2)
}

if __name__ == "__main__":
    # Example input
    sample_input = {
        "fire_weather_index": 6.2,
        "humidity_min": 30,
        "temp_mean": 25.0,
        "temp_range": 12.0,
        "wind_speed_max": 15.0,
        "dewpoint_mean": 10.0,
        "cloud_cover_mean": 20.0,
        "evapotranspiration_total": 5.0
    }

    result = predict_fire(sample_input)
    print("🔥 Prediction Result:", result)