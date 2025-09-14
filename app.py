import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from joblib import load
import os

st.set_page_config(
    page_title="Wildfire Smoke Spread Prediction",
    page_icon="🔥",
    layout="wide",
)

# ---------- Meta ----------
st.title("🔥 Wildfire Smoke Spread Prediction")
st.write(
    "Predict **wildfire occurrence** (Yes/No) and estimate **fire intensity (FRP)** "
    "based on environmental and weather conditions."
)

MODELS_DIR = "models"
CLS_MODEL_PATH = os.path.join(MODELS_DIR, "occurrence_rf.joblib")
REG_MODEL_PATH = os.path.join(MODELS_DIR, "frp_rf.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")

FEATURES = [
    "fire_weather_index",
    "humidity_min",
    "temp_mean",
    "temp_range",
    "wind_speed_max",
    "dewpoint_mean",
    "cloud_cover_mean",
    "evapotranspiration_total",
]

# ---------- Load models ----------
@st.cache_resource
def load_models():
    try:
        cls = load(CLS_MODEL_PATH) if os.path.exists(CLS_MODEL_PATH) else None
    except Exception:
        cls = None
    try:
        reg = load(REG_MODEL_PATH) if os.path.exists(REG_MODEL_PATH) else None
    except Exception:
        reg = None
    try:
        scaler = load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    except Exception:
        scaler = None
    return cls, reg, scaler

cls_model, reg_model, scaler = load_models()

# ---------- Sidebar inputs ----------
st.sidebar.header("🌦️ Input Weather Conditions")
default_values = {
    "fire_weather_index": 6.0,
    "humidity_min": 30.0,
    "temp_mean": 25.0,
    "temp_range": 10.0,
    "wind_speed_max": 10.0,
    "dewpoint_mean": 12.0,
    "cloud_cover_mean": 20.0,
    "evapotranspiration_total": 5.0,
}
user_inputs = {}
for f in FEATURES:
    val = st.sidebar.number_input(
        f.replace("_", " ").title(),
        min_value=0.0,
        max_value=1000.0,
        value=float(default_values.get(f, 0.0)),
        step=0.1,
        format="%.2f",
    )
    user_inputs[f] = val
st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("Demo mode (if models missing)", value=False)

# ---------- Prepare input dataframe ----------
input_df = pd.DataFrame([user_inputs], columns=FEATURES)

# ---------- Prediction ----------
def run_prediction(df):
    X = df.copy()
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = (X - X.mean()) / (X.std().replace(0, 1))
        X_scaled = np.nan_to_num(X_scaled)

    results = {}
    if cls_model is not None:
        prob = cls_model.predict_proba(X_scaled)[0][1]
        occ = int(cls_model.predict(X_scaled)[0])
        results["fire_occurrence"] = occ
        results["fire_probability"] = float(prob)
    else:
        results["fire_occurrence"] = None
        results["fire_probability"] = None

    if reg_model is not None:
        frp_pred = reg_model.predict(X_scaled)[0]
        results["predicted_frp"] = float(frp_pred)
    else:
        results["predicted_frp"] = None
    return results

# ---------- Predict button ----------
if st.button("🔮 Predict"):
    if (cls_model is None or reg_model is None or scaler is None) and not demo_mode:
        st.warning("⚠️ Models not found in `/models`. Enable demo mode or retrain.")
    else:
        if demo_mode and (cls_model is None or reg_model is None or scaler is None):
            score = (
                0.25 * (user_inputs["fire_weather_index"] / 10.0)
                + 0.25 * (1 - (user_inputs["humidity_min"] / 100.0))
                + 0.25 * (user_inputs["wind_speed_max"] / 20.0)
                + 0.25 * (user_inputs["temp_mean"] / 40.0)
            )
            prob = float(np.clip(score, 0.0, 1.0))
            occ = int(prob > 0.5)
            frp = float(
                np.clip(
                    (user_inputs["fire_weather_index"] * 2.0 + user_inputs["wind_speed_max"]) / 2.0,
                    0,
                    200,
                )
            )
            res = {"fire_occurrence": occ, "fire_probability": prob, "predicted_frp": frp}
        else:
            res = run_prediction(input_df)

        # ---------- Display results ----------
        st.subheader("📊 Prediction Results")
        if res["fire_occurrence"] is not None:
            if res["fire_occurrence"] == 1:
                st.error(f"⚠️ Fire Likely — Probability: {res['fire_probability']:.2%}")
            else:
                st.success(f"✅ Fire Unlikely — Probability: {res['fire_probability']:.2%}")

            st.progress(min(max(res["fire_probability"], 0), 1))
            st.metric("🔥 Predicted Fire Radiative Power (FRP)", f"{res['predicted_frp']:.2f}")
        else:
            st.warning("Prediction unavailable (missing model).")

# ---------- Insights ----------
st.markdown("---")
st.subheader("📈 Model Insights")

with st.expander("Classification Feature Importance"):
    if cls_model is not None:
        imp = pd.Series(cls_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        imp.plot(kind="barh", ax=ax, color="skyblue")
        ax.set_title("Which features influence fire occurrence?")
        st.pyplot(fig)
    else:
        st.info("Classification model not loaded.")

with st.expander("Regression Feature Importance"):
    if reg_model is not None:
        imp = pd.Series(reg_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        imp.plot(kind="barh", ax=ax, color="skyblue")
        ax.set_title("Which features influence FRP?")
        st.pyplot(fig)
    else:
        st.info("Regression model not loaded.")

with st.expander("Confusion Matrix (Classification)"):
    if cls_model is not None and scaler is not None:
        try:
            df = pd.read_csv("data/final_dataset.csv")
            df_eval = df[FEATURES + ["occured"]].dropna()
            X_eval = scaler.transform(df_eval[FEATURES])
            y_true = df_eval["occured"].astype(int)
            y_pred = cls_model.predict(X_eval)

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["No Fire", "Fire"],
                yticklabels=["No Fire", "Fire"],
                ax=ax,
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not display confusion matrix: {e}")
    else:
        st.info("Confusion matrix unavailable (need model + dataset).")

# ---------- Footer ----------
st.markdown("---")
st.caption("Models are loaded from `/models`. Run `wildfire.ipynb` to retrain models.")