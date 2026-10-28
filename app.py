from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_DIR / "cnn_lstm.keras")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    features = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, scaler, encoder, features

st.title("🛡️ Insider Threat Detection")
st.caption("CICIDS2017 · CNN–LSTM classification · LIME explainability")

try:
    model, scaler, encoder, feature_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model artifacts: {exc}")
    st.stop()

metrics_path = OUT_DIR / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out test accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Input features", metrics["feature_count"])
    c3.metric("Classes", len(metrics["classes"]))
    st.caption(
        "Metrics are from the saved sample-based evaluation, "
        "not a full-dataset benchmark."
    )

st.divider()
st.subheader("Predict from a CSV")
st.write(
    "Upload CICIDS2017-style flow data with the same feature columns "
    "used during training. A Label column is optional."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded, low_memory=False)
        data.columns = data.columns.str.strip()
        source_labels = data["Label"].astype(str) if "Label" in data.columns else None

        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            st.error(
                f"Your CSV is missing {len(missing)} required features. "
                f"Examples: {missing[:8]}"
            )
            st.stop()

        X = data[feature_columns].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        valid = X.notna().all(axis=1)

        if not valid.any():
            st.error("No rows have complete numeric values for all required features.")
            st.stop()

        X_valid = X.loc[valid]
        scaled = np.clip(scaler.transform(X_valid.to_numpy(dtype=np.float32)), 0, 1)
        probs = model.predict(
            scaled.reshape(-1, scaled.shape[1], 1),
            batch_size=256,
            verbose=0
        )
        ids = probs.argmax(axis=1)

        result = pd.DataFrame({
            "Predicted label": encoder.inverse_transform(ids),
            "Confidence": probs.max(axis=1)
        }, index=X_valid.index)

        if source_labels is not None:
            result.insert(0, "Original label", source_labels.loc[X_valid.index])

        st.success(f"Predictions generated for {len(result):,} valid rows.")
        st.dataframe(result.head(100), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            result.to_csv(index=True).encode("utf-8"),
            file_name="intrusion_predictions.csv",
            mime="text/csv"
        )

        st.subheader("Prediction summary")
        st.bar_chart(result["Predicted label"].value_counts())

        st.subheader("Explain one prediction with LIME")
        selected_index = st.selectbox(
            "Choose a row from the valid uploaded records",
            options=list(X_valid.index)
        )

        if st.button("Generate LIME explanation"):
            with st.spinner("Generating explanation..."):
                background = scaled[:min(500, len(scaled))]
                explainer = LimeTabularExplainer(
                    training_data=background,
                    feature_names=feature_columns,
                    class_names=list(encoder.classes_),
                    mode="classification",
                    random_state=42
                )

                row_pos = list(X_valid.index).index(selected_index)
                row = scaled[row_pos]

                def predict_proba(batch):
                    batch = np.asarray(batch, dtype=np.float32)
                    batch = np.clip(batch, 0, 1)
                    return model.predict(
                        batch.reshape(-1, batch.shape[1], 1),
                        verbose=0
                    )

                explanation = explainer.explain_instance(
                    row,
                    predict_proba,
                    num_features=10,
                    num_samples=1000,
                    top_labels=1
                )
                explained_label = int(np.argmax(probs[row_pos]))
                fig = explanation.as_pyplot_figure(label=explained_label)
                st.pyplot(fig)
                plt.close(fig)

                st.write(
                    "Prediction:",
                    encoder.classes_[explained_label],
                    "· Confidence:",
                    f"{probs[row_pos, explained_label]:.2%}"
                )

    except Exception as exc:
        st.exception(exc)

st.divider()
st.subheader("Saved evaluation artifacts")

report_path = OUT_DIR / "classification_report.json"
cm_path = OUT_DIR / "confusion_matrix.png"
lime_path = OUT_DIR / "lime_explanation.png"

if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.write("Classification report")
    st.json({
        "macro avg": report.get("macro avg", {}),
        "weighted avg": report.get("weighted avg", {})
    })

if cm_path.exists():
    st.write("Confusion matrix")
    st.image(str(cm_path), use_container_width=True)

if lime_path.exists():
    st.write("Previously generated LIME example")
    st.image(str(lime_path), use_container_width=True)

st.caption(
    "Research prototype for educational demonstration. "
    "Predictions are model outputs, not guaranteed security determinations."
)
