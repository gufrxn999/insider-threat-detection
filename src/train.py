from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
MAX_PER_CLASS = 2000
CHUNK_SIZE = 50000

ROOT = Path("data/MachineLearningCSV")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

files = sorted(ROOT.rglob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files found under {ROOT}")

# Collect a bounded, reproducible sample from each label.
samples = {}
for file in files:
    print(f"Scanning: {file.name}", flush=True)
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing in {file.name}")

        chunk["Label"] = chunk["Label"].astype(str).str.strip()
        for label, group in chunk.groupby("Label", sort=False):
            remaining = MAX_PER_CLASS - len(samples.get(label, []))
            if remaining <= 0:
                continue

            take = group.sample(
                n=min(remaining, len(group)),
                random_state=SEED
            )
            samples.setdefault(label, []).append(take)

frames = [
    pd.concat(parts, ignore_index=True)
    for parts in samples.values() if parts
]
data = pd.concat(frames, ignore_index=True)
print(f"\nSample rows collected: {len(data):,}")
print("Class counts before cleaning:")
print(data["Label"].value_counts().to_string())

labels = data["Label"].copy()
X = data.drop(columns=["Label"]).copy()

# Convert features to numeric and remove invalid values.
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that have no usable numeric values.
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    raise ValueError("No usable numeric features found.")

# Remove rows with missing features; don't invent values for corrupted rows.
valid = X.notna().all(axis=1)
X = X.loc[valid].reset_index(drop=True)
labels = labels.loc[valid].reset_index(drop=True)

# Remove classes with fewer than 2 usable samples.
counts = labels.value_counts()
keep_labels = counts[counts >= 2].index
valid = labels.isin(keep_labels)
X = X.loc[valid].reset_index(drop=True)
labels = labels.loc[valid].reset_index(drop=True)

print(f"\nRows after cleaning: {len(X):,}")
print("Usable classes:")
print(labels.value_counts().to_string())

if labels.nunique() < 2:
    raise ValueError("Need at least two usable classes.")

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
class_counts = np.bincount(y)
if class_counts.min() < 2:
    raise ValueError("A class has fewer than two usable rows.")

X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(dtype=np.float32),
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Fit the scaler only on training data to avoid test-data leakage.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.clip(X_train, 0, 1)
X_test = np.clip(X_test, 0, 1)

# CNN-LSTM over the feature axis: (samples, 78 features, 1 channel).
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

num_classes = len(encoder.classes_)
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(64),
    layers.Dropout(0.5),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

probabilities = model.predict(X_test, batch_size=256, verbose=0)
predictions = probabilities.argmax(axis=1)
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test, predictions,
    labels=np.arange(num_classes),
    target_names=encoder.classes_,
    zero_division=0,
    output_dict=True
)

model.save(MODEL_DIR / "cnn_lstm.keras")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

with open(OUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": float(accuracy),
        "test_samples": int(len(y_test)),
        "classes": encoder.classes_.tolist(),
        "feature_count": int(X.shape[1]),
        "sample_cap_per_class": MAX_PER_CLASS,
        "seed": SEED
    }, f, indent=2)

cm = confusion_matrix(y_test, predictions, labels=np.arange(num_classes))
fig, ax = plt.subplots(figsize=(13, 11))
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=encoder.classes_
).plot(ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Model saved to: {MODEL_DIR / 'cnn_lstm.keras'}")
print(f"Metrics saved to: {OUT_DIR / 'metrics.json'}")
print(f"Report saved to: {OUT_DIR / 'classification_report.json'}")
print(f"Confusion matrix saved to: {OUT_DIR / 'confusion_matrix.png'}")
