import os
import pandas as pd
import streamlit as st
from datetime import datetime
import joblib

from predict import predict_disease

# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYMPTOM_COLUMNS_PATH = os.path.join(BASE_DIR, "model", "symptom_columns.pkl")
LOG_PATH = os.path.join(BASE_DIR, "history", "prediction_log.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD SYMPTOMS
# =========================
if not os.path.exists(SYMPTOM_COLUMNS_PATH):
    st.error("Model symptom file not found. Please train the model first.")
    st.stop()

symptom_columns = joblib.load(SYMPTOM_COLUMNS_PATH)

# Create display mapping (spaces and Title Casing)
display_to_original = {sym.replace("_", " ").title(): sym for sym in symptom_columns}
original_to_display = {sym: sym.replace("_", " ").title() for sym in symptom_columns}

# =========================
# HEADER
# =========================
st.title("🩺 Smart AI Disease Prediction System")
st.markdown(
    """
This system predicts possible diseases based on selected symptoms using trained machine learning models.  
It is **for educational purposes only** and **not a replacement for a real doctor**.
"""
)

st.warning(
    "Medical disclaimer: This prediction system is not a clinical diagnostic tool. "
    "Always consult a qualified doctor for proper diagnosis and treatment."
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Patient Input")

selected_displays = st.sidebar.multiselect(
    "Select symptoms:",
    options=list(display_to_original.keys())
)

# Convert display names back to model feature names
selected_symptoms = [display_to_original[disp] for disp in selected_displays]

predict_button = st.sidebar.button("Predict Disease")

if st.sidebar.button("Clear Selection"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.write("**Algorithms trained:**")
st.sidebar.write("- Bernoulli Naive Bayes")
st.sidebar.write("- Random Forest")
st.sidebar.write("- Logistic Regression")
st.sidebar.write("**Final deployed model:** Best-performing saved model (Logistic Regression)")
st.sidebar.write("**Features:** Symptom-based disease prediction")
st.sidebar.write("**Dataset:** Kaggle symptom-disease dataset")

# =========================
# HELPER: SAVE HISTORY
# =========================
def save_prediction_history(selected_symptoms, result):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms": ", ".join([original_to_display[sym] for sym in selected_symptoms]),
        "predicted_disease": result.get("disease", "N/A"),
        "severity": result.get("severity", "N/A"),
        "risk_level": result.get("risk_level", "Unknown"),
        "is_emergency": str(result.get("is_emergency", False))
    }

    log_df = pd.DataFrame([log_entry])

    file_exists_and_not_empty = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0

    if file_exists_and_not_empty:
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)

# =========================
# MAIN PREDICTION
# =========================
if predict_button:
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        try:
            result = predict_disease(selected_symptoms)

            # -------------------------
            # TOP RESULT CARDS
            # -------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"### Predicted Disease: {result.get('disease', 'N/A')}")
                st.info(f"### Severity Level: {result.get('severity', 'N/A')}")
                st.warning(f"### Risk Level: {result.get('risk_level', 'Unknown')}")

            with col2:
                st.subheader("Top 3 Predictions")
                probabilities = result.get("probabilities", [])
                if probabilities:
                    for item in probabilities:
                        disease = item.get("disease", "N/A")
                        prob = item.get("probability", 0.0)
                        st.write(f"**{disease}** — {prob}%")
                else:
                    st.write("No probability predictions available.")

            # -------------------------
            # DISEASE DETAILS
            # -------------------------
            st.markdown("---")
            st.subheader("📖 Disease Description")
            st.write(result.get("description", "No description available."))

            st.subheader("🛡️ Precautions")
            st.write(result.get("precautions", "No precautions available."))

            st.subheader("👨‍⚕️ Recommended Specialist")
            st.write(result.get("doctor", "General Physician"))

            # -------------------------
            # EMERGENCY WARNING
            # -------------------------
            if result.get("is_emergency", False):
                st.error(
                    "⚠️ Emergency Warning: Some selected symptoms may indicate a serious condition. "
                    "Please seek immediate medical attention."
                )

            # -------------------------
            # SHOW SELECTED SYMPTOMS
            # -------------------------
            st.markdown("---")
            st.subheader("Selected Symptoms")
            st.write(", ".join([original_to_display[sym] for sym in selected_symptoms]))

            # -------------------------
            # SAVE HISTORY
            # -------------------------
            save_prediction_history(selected_symptoms, result)
            st.success("Prediction saved to history.")

        except Exception as e:
            st.error(f"An error occurred while making prediction: {e}")

# =========================
# SHOW HISTORY BELOW
# =========================
st.markdown("---")
def show_prediction_history():
    if not os.path.exists(LOG_PATH):
        st.info("No prediction history yet.")
        return

    if os.path.getsize(LOG_PATH) == 0:
        st.info("No prediction history yet.")
        return

    try:
        history_df = pd.read_csv(LOG_PATH)

        if history_df.empty:
            st.info("No prediction history yet.")
        else:
            st.subheader("📜 Prediction History")
            st.dataframe(history_df.tail(10), use_container_width=True)

    except pd.errors.EmptyDataError:
        st.info("No prediction history yet.")
    except Exception:
        st.info("Prediction history could not be loaded yet.")

show_prediction_history()