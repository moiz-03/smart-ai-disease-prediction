# 🩺 Smart AI Disease Prediction System

A machine learning-powered web application that predicts possible diseases based on selected symptoms. Built with **Streamlit**, trained on a large Kaggle symptom-disease dataset, and backed by three competing ML models — all wrapped in a clean, informative UI.

> ⚠️ **Medical Disclaimer:** This system is for educational purposes only and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 🚀 Features

- 🔍 **Symptom-based prediction** — select from 377 real-world symptoms
- 🏆 **Top 3 disease predictions** with confidence probabilities
- 📋 **Disease details** — description, precautions, and recommended specialist
- 🚨 **Emergency detection** — flags critical symptom combinations automatically
- 📜 **Prediction history** — logs every prediction with timestamp, severity, and risk level
- 🧠 **Three ML models compared** — best-performing model deployed automatically

---

## 📊 Model Performance

Three classifiers were trained and evaluated on the dataset. The best-performing model is automatically saved and deployed.

| Model | Accuracy |
|---|---|
| Bernoulli Naive Bayes | 90.95% |
| Random Forest | — |
| **Logistic Regression** ✅ | **92.92%** |

> **Deployed model:** Logistic Regression — 92.92% accuracy  
> **Dataset:** 157 diseases · 377 symptom features

---

## 🗂️ Project Structure

```
disease-predictor/
│
├── app.py                  # Streamlit web application (main entry point)
├── predict.py              # Prediction engine (model loading, inference, severity)
├── train_model.py          # Model training script
│
├── data/
│   ├── disease_dataset.csv # Training dataset (symptom–disease pairs)
│   └── disease_info.csv    # Disease metadata (descriptions, precautions, doctors)
│
├── model/
│   ├── disease_model.pkl   # Trained Logistic Regression model
│   ├── label_encoder.pkl   # Label encoder for disease class names
│   └── symptom_columns.pkl # Ordered list of symptom feature columns
│
├── outputs/
│   └── model_metrics.txt   # Full classification report for all 3 trained models
│
├── history/
│   └── prediction_log.csv  # Auto-generated prediction history log
│
└── tests/
    ├── test_predict.py      # Unit tests for prediction logic
    ├── test_train_model.py  # Integration tests for training pipeline
    └── test_app.py          # UI tests using Streamlit AppTest
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd disease-predictor
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn pandas joblib
```

### 3. (Optional) Retrain the model

Only required if you want to retrain from scratch. Pre-trained model artifacts are already included in `model/`.

```bash
python train_model.py
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧪 Running Tests

The project includes a full test suite covering the prediction engine, training pipeline, and Streamlit UI.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Results:** 32 tests — all passing ✅

| Test File | Coverage | Tests |
|---|---|---|
| `test_predict.py` | Severity scoring, input prep, disease lookup, prediction flow | 7 |
| `test_train_model.py` | Model artifacts, label encoder, metrics file output | 8 |
| `test_app.py` | Page state, form validation, prediction UI, history, emergency alert | 17 |

---

## 🧠 How It Works

1. **Select symptoms** from the multi-select dropdown in the sidebar (377 options available).
2. Click **Predict Disease** — the app encodes your symptoms as a binary feature vector and runs them through the trained Logistic Regression model.
3. The top predicted disease is shown alongside its **description**, **precautions**, and **recommended specialist**.
4. The **Top 3 Predictions** panel shows alternative diagnoses with their confidence percentages.
5. A **severity score** is calculated based on the symptom set — critical symptoms (e.g. chest pain, shortness of breath) automatically trigger an emergency warning.
6. Every prediction is saved to `history/prediction_log.csv` for review.

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| Web Framework | [Streamlit](https://streamlit.io) |
| ML Models | scikit-learn (Logistic Regression, Random Forest, Bernoulli NB) |
| Data Processing | pandas, NumPy |
| Model Serialisation | joblib |
| Testing | Python `unittest` + Streamlit `AppTest` |
| Dataset | Kaggle symptom–disease dataset |

---

## 📄 License

This project is for educational purposes. Always consult a qualified medical professional for health concerns.
