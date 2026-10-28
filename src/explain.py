from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

DATA_DIR = Path("data/MachineLearningCSV")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

model = tf.keras.models.load_model(MODEL_DIR / "cnn_lstm.keras")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

# Read a small set of rows to construct a representative explanation example.
files = sorted(DATA_DIR.rglob("*.csv"))
if not files:
    raise FileNotFoundError("No CICIDS2017 CSV files found.")

parts = []
for file in files:
    sample = pd.read_csv(file, nrows=1000, low_memory=False)
    sample.columns = sample.columns.str.strip()
    if "Label" in sample.columns:
        parts.append(sample)

if not parts:
    raise ValueError("No CSV data with a Label column was found.")

data = pd.concat(parts, ignore_index=True)
data["Label"] = data["Label"].astype(str).str.strip()

X = data.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.reindex(columns=feature_columns)
valid = X.notna().all(axis=1)
X = X.loc[valid].reset_index(drop=True)
labels = data.loc[valid, "Label"].reset_index(drop=True)

if len(X) == 0:
    raise ValueError("No valid rows remain after cleaning.")

# Transform exactly as during training.
scaled = np.clip(scaler.transform(X.to_numpy(dtype=np.float32)), 0, 1)
sample_idx = 0
row = scaled[sample_idx]
prediction = model.predict(row.reshape(1, -1, 1), verbose=0)[0]
predicted_id = int(np.argmax(prediction))
predicted_label = str(encoder.classes_[predicted_id])

# LIME operates on 2D tabular features; wrap the model to preserve the
# CNN-LSTM input shape used during training.
def predict_proba(batch):
    batch = np.asarray(batch, dtype=np.float32)
    batch = np.clip(batch, 0, 1)
    return model.predict(batch.reshape(-1, batch.shape[1], 1), verbose=0)

# Use a bounded background sample for performance.
background = scaled[:min(500, len(scaled))]
explainer = LimeTabularExplainer(
    training_data=background,
    feature_names=feature_columns,
    class_names=list(encoder.classes_),
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

explanation = explainer.explain_instance(
    row,
    predict_proba,
    num_features=10,
    num_samples=1000,
    top_labels=1
)

explanation.save_to_file(str(OUT_DIR / "lime_explanation.html"))
explained_label = next(iter(explanation.local_exp))
fig = explanation.as_pyplot_figure(label=explained_label)
fig.tight_layout()
fig.savefig(OUT_DIR / "lime_explanation.png", dpi=160, bbox_inches="tight")
plt.close(fig)

result = {
    "predicted_label": predicted_label,
    "confidence": float(prediction[predicted_id]),
    "source_label": str(labels.iloc[sample_idx]),
    "explanation": [
        {"feature": feature, "weight": float(weight)}
        for feature, weight in explanation.as_list(label=explained_label)
    ]
}
with open(OUT_DIR / "lime_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("Actual label:", result["source_label"])
print("Predicted label:", result["predicted_label"])
print("Prediction confidence:", round(result["confidence"], 4))
print("\nTop LIME feature contributions:")
for item in result["explanation"]:
    print(f"  {item['feature']}: {item['weight']:+.4f}")

print("\nSaved:")
print("  outputs/lime_explanation.html")
print("  outputs/lime_explanation.png")
print("  outputs/lime_result.json")


