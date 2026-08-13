import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "disease_model.pkl")
SYMPTOM_COLUMNS_PATH = os.path.join(BASE_DIR, "model", "symptom_columns.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "data", "disease_info.csv")

# Global cache for loaded model and data artifacts
_model = None
_label_encoder = None
_symptom_columns = None
_disease_info_df = None


def load_artifacts():
    global _model, _label_encoder, _symptom_columns, _disease_info_df
    if _model is None or _label_encoder is None or _symptom_columns is None or _disease_info_df is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            _model = None

        try:
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        except Exception:
            _label_encoder = None

        try:
            _symptom_columns = joblib.load(SYMPTOM_COLUMNS_PATH)
        except Exception:
            _symptom_columns = []

        try:
            if os.path.exists(DISEASE_INFO_PATH):
                df_info = pd.read_csv(DISEASE_INFO_PATH)
                df_info.columns = [c.strip().lower() for c in df_info.columns]

                # Map legacy columns to the new expected schema
                if "doctor_advice" in df_info.columns and "doctor" not in df_info.columns:
                    df_info = df_info.rename(columns={"doctor_advice": "doctor"})
                if "severity" in df_info.columns and "risk_level" not in df_info.columns:
                    df_info = df_info.rename(columns={"severity": "risk_level"})

                required_cols = ["disease", "description", "precautions", "doctor", "risk_level"]
                for col in required_cols:
                    if col not in df_info.columns:
                        df_info[col] = ""

                df_info["disease"] = (
                    df_info["disease"].astype(str).str.strip().str.lower()
                )
                _disease_info_df = df_info
            else:
                _disease_info_df = pd.DataFrame(
                    columns=["disease", "description", "precautions", "doctor", "risk_level"]
                )
        except Exception:
            _disease_info_df = pd.DataFrame(
                columns=["disease", "description", "precautions", "doctor", "risk_level"]
            )

    return _model, _label_encoder, _symptom_columns, _disease_info_df


def prepare_input(selected_symptoms, symptom_columns):
    row = {symptom: 0 for symptom in symptom_columns}

    # Normalize input symptoms
    normalized_selected = [str(sym).strip().lower() for sym in selected_symptoms]

    for sym in normalized_selected:
        if sym in row:
            row[sym] = 1

    df_input = pd.DataFrame([row])
    # Ensure exact same column ordering as in the training dataset
    if len(symptom_columns) > 0:
        df_input = df_input[symptom_columns]
    return df_input


def is_valid_value(val):
    if pd.isna(val):
        return False
    if str(val).strip().lower() in ["", "nan", "none"]:
        return False
    return True


def get_disease_info(disease_name):
    global _disease_info_df
    load_artifacts()

    disease_name = str(disease_name).strip().lower()

    default_info = {
        "description": "No description available.",
        "precautions": "No precautions available.",
        "doctor": "General Physician",
        "risk_level": "Unknown"
    }

    if _disease_info_df is None or _disease_info_df.empty:
        return default_info

    if "disease" not in _disease_info_df.columns:
        return default_info

    row = _disease_info_df[_disease_info_df["disease"] == disease_name]

    if row.empty:
        return default_info

    row = row.iloc[0]

    return {
        "description": row["description"] if is_valid_value(row["description"]) else default_info["description"],
        "precautions": row["precautions"] if is_valid_value(row["precautions"]) else default_info["precautions"],
        "doctor": row["doctor"] if is_valid_value(row["doctor"]) else default_info["doctor"],
        "risk_level": row["risk_level"] if is_valid_value(row["risk_level"]) else default_info["risk_level"],
    }


def calculate_severity(selected_symptoms):
    CRITICAL_SYMPTOMS = {
        "chest pain",
        "shortness of breath",
        "fainting",
        "confusion",
        "severe bleeding",
        "paralysis",
        "seizure",
        "coughing blood"
    }
    MODERATE_SYMPTOMS = {
        "fever",
        "vomiting",
        "severe headache",
        "dehydration",
        "rapid heartbeat",
        "abdominal pain"
    }

    score = 0
    has_critical = False

    for sym in selected_symptoms:
        sym_lower = str(sym).strip().lower()
        if sym_lower in CRITICAL_SYMPTOMS:
            score += 3
            has_critical = True
        elif sym_lower in MODERATE_SYMPTOMS:
            score += 2
        else:
            score += 1

    if score >= 8:
        severity = "High"
    elif score >= 4:
        severity = "Medium"
    else:
        severity = "Low"

    is_emergency = (severity == "High") or has_critical

    return severity, is_emergency


def predict_disease(selected_symptoms):
    model, label_encoder, symptom_columns, disease_info_df = load_artifacts()

    if not selected_symptoms:
        raise ValueError("Please select at least one symptom.")

    if model is None or label_encoder is None:
        raise ValueError("Model artifacts are not loaded properly.")

    input_df = prepare_input(selected_symptoms, symptom_columns)

    pred_encoded = model.predict(input_df)[0]
    predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]

    top_predictions = []
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            class_names = label_encoder.classes_
            top_indices = probabilities.argsort()[::-1][:3]
            top_predictions = [
                {
                    "disease": class_names[i],
                    "probability": float(round(probabilities[i] * 100, 2))
                }
                for i in top_indices
            ]
        except Exception:
            top_predictions = []

    disease_info = get_disease_info(predicted_disease)
    severity, is_emergency = calculate_severity(selected_symptoms)

    return {
        "disease": predicted_disease,
        "description": disease_info["description"],
        "precautions": disease_info["precautions"],
        "doctor": disease_info["doctor"],
        "risk_level": disease_info["risk_level"],
        "severity": severity,
        "is_emergency": is_emergency,
        "probabilities": top_predictions
    }

