import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
DATA_PATH = "data/disease_dataset.csv"
MODEL_DIR = "model"
OUTPUT_DIR = "outputs"
TARGET_COL = "diseases"

MAX_ROWS = 50000       # adjust if needed
CHUNK_SIZE = 10000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset in chunks...")

# =========================
# 1) LOAD DATA IN CHUNKS
# =========================
chunks = []
rows_loaded = 0

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    chunk.columns = [col.strip().lower() for col in chunk.columns]

    if TARGET_COL not in chunk.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    chunks.append(chunk)
    rows_loaded += len(chunk)
    print(f"Loaded {rows_loaded} rows...")

    if rows_loaded >= MAX_ROWS:
        break

df = pd.concat(chunks, ignore_index=True)
df = df.iloc[:MAX_ROWS].copy()

print(f"\nUsing {len(df)} rows before cleaning.")

# =========================
# 2) CLEAN DATA
# =========================
df = df.drop_duplicates()

y = df[TARGET_COL].astype(str).str.strip()
X = df.drop(columns=[TARGET_COL])

print(f"Rows after duplicate removal: {len(df)}")
print("Original feature shape:", X.shape)

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Convert to binary 0/1
X = (X > 0).astype("uint8")

# =========================
# 3) REMOVE RARE CLASSES
# =========================
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index

mask = y.isin(valid_classes)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"Rows after removing rare diseases: {len(y)}")
print(f"Remaining diseases: {y.nunique()}")

# =========================
# 4) ENCODE LABELS
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Processed feature shape:", X.shape)
print("Feature dtype:", X.dtypes.iloc[0])
print("Number of diseases:", len(label_encoder.classes_))
print("Number of symptoms/features:", X.shape[1])

# =========================
# 5) TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# 6) TRAIN MODELS
# =========================
models = {
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
reports = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    results[model_name] = {
        "model": model,
        "accuracy": acc
    }
    reports[model_name] = report

    print(f"{model_name} Accuracy: {acc * 100:.2f}%")

# =========================
# 7) PICK BEST MODEL
# =========================
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]
best_accuracy = results[best_model_name]["accuracy"]

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print("==============================")

# =========================
# 8) SAVE BEST MODEL + ARTIFACTS
# =========================
joblib.dump(best_model, os.path.join(MODEL_DIR, "disease_model.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "symptom_columns.pkl"))

print("\nBest model and artifacts saved successfully in /model")

# =========================
# 9) SAVE METRICS TO FILE
# =========================
metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("SMART AI DISEASE PREDICTION SYSTEM - MODEL COMPARISON\n")
    f.write("=" * 60 + "\n\n")

    for model_name in results:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {results[model_name]['accuracy'] * 100:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(reports[model_name])
        f.write("\n" + "=" * 60 + "\n\n")

    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write(f"BEST ACCURACY: {best_accuracy * 100:.2f}%\n")

print(f"Model metrics saved to: {metrics_path}")
print("\nTraining complete.")