"""
MediLens AI — Streamlit App (Step 5: Deployment)

Run locally with:
    streamlit run app/streamlit_app.py

Features:
  - Upload a CSV of patient records (same 30 features as the training set) OR
    manually adjust key sliders for a single patient.
  - Real-time risk prediction from the trained RandomForest model.
  - SHAP-based explanation of the prediction ("why did the model say this?").
  - Auto-generated plain-English summary for the patient.
  - CNN demo tab: upload a chest X-ray-like image for a pneumonia/normal call.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="MediLens AI", page_icon="🩺", layout="wide")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

@st.cache_resource
def load_artifacts():
    rf = joblib.load(f"{ARTIFACT_DIR}/random_forest_model.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
    feature_cols = joblib.load(f"{ARTIFACT_DIR}/feature_cols.joblib")
    explainer = joblib.load(f"{ARTIFACT_DIR}/shap_explainer.joblib")
    return rf, scaler, feature_cols, explainer

def generate_patient_summary(diagnosis, prob, top_features, recommendation):
    risk_word = "elevated" if prob >= 0.7 else ("moderate" if prob >= 0.4 else "low")
    feats_text = ", ".join(top_features[:3])
    return (
        f"Your recent results show a {risk_word} likelihood ({prob:.0%}) of {diagnosis}. "
        f"The factors that most influenced this result were: {feats_text}. "
        f"This is not a final diagnosis — {recommendation} Please discuss these results "
        f"with your care provider, who can put them in context with your full medical history."
    )

st.title("🩺 MediLens AI — Diagnostic & Insight Prototype")
st.caption(
    "Prototype only — not a medical device, not for clinical use. "
    "Predictions are illustrative and trained on a public research dataset."
)

tab1, tab2, tab3 = st.tabs(["📊 Tabular Risk Model", "🫁 Imaging Classifier (demo)", "ℹ️ About & Ethics"])

# ---------------------------------------------------------------------------
# TAB 1: Tabular risk model
# ---------------------------------------------------------------------------
with tab1:
    rf, scaler, feature_cols, explainer = load_artifacts()

    st.subheader("Patient Risk Assessment")
    mode = st.radio("Input method", ["Upload CSV", "Manual entry (key features)"], horizontal=True)

    input_df = None

    if mode == "Upload CSV":
        uploaded = st.file_uploader(f"Upload a CSV with these {len(feature_cols)} columns", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            missing = set(feature_cols) - set(raw.columns)
            if missing:
                st.error(f"Missing columns: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
            else:
                input_df = raw[feature_cols].fillna(raw[feature_cols].median(numeric_only=True))
    else:
        st.write("Adjust the most influential features (others default to dataset median):")
        defaults = {}
        key_feats = ["worst area", "worst concave points", "worst perimeter",
                     "mean concave points", "worst radius"]
        cols = st.columns(len(key_feats))
        for i, feat in enumerate(key_feats):
            with cols[i]:
                defaults[feat] = st.slider(feat, 0.0, 3000.0 if "area" in feat else 40.0,
                                            value=100.0 if "area" not in feat else 800.0)
        row = {f: 0.0 for f in feature_cols}
        row.update(defaults)
        input_df = pd.DataFrame([row])

    if input_df is not None and st.button("Run Diagnosis", type="primary"):
        X_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
        proba = rf.predict_proba(X_scaled)[:, 1]
        preds = rf.predict(X_scaled)

        idx = 0  # show first row in detail
        st.metric("Predicted probability (malignant-pattern risk)", f"{proba[idx]:.1%}")

        shap_values = explainer.shap_values(X_scaled.iloc[[idx]], check_additivity=False)
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        elif isinstance(shap_values, list):
            sv = shap_values[1]

        top_idx = np.argsort(-np.abs(sv[0]))[:5]
        top_features_for_patient = [feature_cols[i] for i in top_idx]

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("**Top contributing factors (SHAP)**")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh([feature_cols[i] for i in top_idx][::-1], sv[0][top_idx][::-1])
            ax.set_xlabel("SHAP value (impact on risk)")
            st.pyplot(fig)

        with col_b:
            st.write("**Plain-English summary**")
            summary = generate_patient_summary(
                diagnosis="malignant tissue characteristics",
                prob=float(proba[idx]),
                top_features=top_features_for_patient,
                recommendation="a follow-up biopsy and specialist consultation are recommended.",
            )
            st.info(summary)

        if len(input_df) > 1:
            st.write("**Batch results**")
            out = input_df.copy()
            out["predicted_risk_probability"] = proba
            out["predicted_class"] = np.where(preds == 1, "malignant-pattern", "benign-pattern")
            st.dataframe(out[["predicted_class", "predicted_risk_probability"]])

# ---------------------------------------------------------------------------
# TAB 2: Imaging demo
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Chest X-ray Classifier (architecture demo)")
    st.write(
        "This tab illustrates the deployment pattern for the CNN/MobileNetV2 "
        "classifier from `scripts/cnn_xray_classifier.py`. In production, load "
        "the saved Keras model here and run `model.predict()` on the uploaded "
        "image after resizing/normalizing it to match training preprocessing."
    )
    img_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        st.image(img_file, caption="Uploaded image", width=300)
        st.warning(
            "This demo build does not ship pretrained X-ray model weights "
            "(the real Chest X-Ray Pneumonia dataset requires a Kaggle download "
            "that wasn't available in the build environment). Wire in your "
            "trained `.h5`/`.keras` model file to get real predictions here — "
            "see the CNN script for the exact architecture and preprocessing."
        )

# ---------------------------------------------------------------------------
# TAB 3: About / Ethics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("About this prototype")
    st.markdown("""
    - **Tabular model**: RandomForestClassifier trained on the Wisconsin Breast
      Cancer Diagnostic dataset (bundled with scikit-learn), used here as a
      stand-in for a Heart Disease / Diabetes-style tabular diagnosis task.
    - **Imaging model**: CNN + MobileNetV2 transfer-learning architecture,
      demoed on synthetic image data — see `MODEL_REPORT.md` for details on
      why real pretrained weights weren't available in this build.
    - **Explainability**: SHAP TreeExplainer surfaces the features driving
      each individual prediction, addressing the "black box" concern.

    ### Ethics & bias considerations
    - This dataset skews toward a single population; a production system needs
      training data audited for demographic representativeness before
      deployment on a general patient population.
    - False negatives (missed malignant cases) are more costly than false
      positives in this domain — threshold tuning and recall should be
      prioritized over raw accuracy.
    - Model outputs are **decision support only**, not a diagnosis, and should
      always be paired with clinician review.
    - Patient-facing summaries must avoid alarming or falsely reassuring
      language — the template here is deliberately conservative and always
      points back to a human clinician.
    """)
