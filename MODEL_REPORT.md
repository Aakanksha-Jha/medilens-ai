# MediLens AI — Model Report

**Prototype build — not a medical device, not validated for clinical use.**

## 1. Tabular Diagnostic Risk Model (Steps 1–2)

**Dataset:** Wisconsin Breast Cancer Diagnostic dataset (569 patients, 30
features), bundled with scikit-learn. Used as a stand-in for the brief's
suggested Heart Disease / Diabetes UCI datasets — same problem shape
(multivariate tabular → binary diagnosis) — because this build environment's
network is restricted to package registries and can't reach the UCI
Repository or Kaggle. Swapping the data source is a one-line change (see
`scripts/run_eda_ml.py` docstring).

**Preprocessing:** simulated realistic missingness (1% MCAR), median
imputation, `StandardScaler` feature scaling. 74 of 569 rows contained at
least one feature more than 3 standard deviations from its mean (flagged,
not automatically dropped — clinically these outliers can be genuine, so
they went into a held-out-review bucket rather than being silently removed).

### Model comparison (80/20 stratified split, test set n=113)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.947 | 0.974 | 0.881 | 0.925 | 0.978 |
| Random Forest | 0.956 | 1.000 | 0.881 | 0.937 | 0.991 |
| XGBoost | 0.947 | 1.000 | 0.857 | 0.923 | 0.994 |

Random Forest 5-fold cross-validated F1: **0.942 ± 0.032** — confirms the
held-out split result wasn't a lucky draw.

**Chosen model for deployment: Random Forest.** Best F1/accuracy trade-off,
and tree-based SHAP explanations are cheap to compute in real time for the
Streamlit app.

### PCA — dimensionality reduction impact

| Components | F1 | Fit time (s) | Explained variance |
|---|---|---|---|
| 30 (all) | 0.925 | 0.521 | 100% |
| 10 | 0.925 | 0.393 | 94.9% |
| 5 | **0.950** | 0.342 | 84.2% |
| 2 | 0.927 | 0.296 | 62.9% |

Reducing to 5 principal components matched or slightly beat the full
feature set on F1 while training ~34% faster — a reasonable trade-off if
inference latency or storage becomes a constraint at larger scale. Below 5
components, accuracy degrades as too much discriminative variance is lost.

### Explainability (SHAP)

Top global risk drivers by mean absolute SHAP value: **worst area**, **worst
concave points**, **worst perimeter**, **mean concave points**, **worst
radius**. These align with known clinical intuition for this dataset (larger,
more irregularly-shaped cell nuclei correlate with malignancy), which is a
useful sanity check that the model learned real signal rather than an
artifact.

## 2. Computer Vision — CNN + Transfer Learning (Step 3)

**Data note:** the real target dataset (Chest X-Ray Pneumonia, Kaggle) could
not be downloaded in this build sandbox (no general internet egress, only
package registries). Two architectures were built and trained end-to-end on
synthetic procedurally-generated "lung-like" images so the code path is
proven to run correctly:

| Model | Test Loss | Test Accuracy |
|---|---|---|
| Small CNN (trained from scratch, 3 conv blocks) | 0.001 | 100%* |
| MobileNetV2 (transfer-learning architecture) | 0.693 | 50%* |

*Both numbers are on synthetic data and **not representative of real-world
performance** — the small CNN essentially memorized an easy synthetic
signal, while MobileNetV2 shows no learning at chance-level accuracy because
its ImageNet pretrained weights could not be downloaded
(`storage.googleapis.com` was unreachable), so it ran with random
initialization and a frozen (untrainable) backbone — expected behavior for
an untrained frozen network. **With normal internet access, `weights="imagenet"`
loads automatically and this becomes genuine transfer learning**, which is
the intended production configuration. Architecture, preprocessing, and
training loop are otherwise production-ready — point
`image_dataset_from_directory` at the real dataset folder and no other code
changes are needed.

## 3. Sequence Modeling — LSTM Vitals Forecast (Step 3)

Trained on synthetic per-patient daily heart-rate series (200 patients, 60
days, with a 25% chance of an injected deterioration event) to forecast the
next 3 days from a 14-day window.

- **Test MAE:** 1.84 bpm
- **Test MSE (normalized):** 0.099

This is a real, executed result on realistic-shaped synthetic vitals data.
The same architecture applies directly to real EHR/wearable vitals streams.

## 4. NLP — Note Classification & Plain-English Summaries (Step 4)

**TF-IDF + Logistic Regression** classified patient notes into
`medication_side_effect`, `appointment_logistics`, `symptom_report`,
`positive_feedback` — macro F1 = **0.59** on a small illustrative dataset
(~32 hand-written examples, held-out n=10). This is a toy-scale demo; a
production system would train on thousands of triaged historical notes and
should comfortably exceed this.

**Plain-English summaries:** the production design uses a HuggingFace
`transformers.pipeline("summarization")` call (e.g. `distilbart-cnn-12-6`)
to turn structured model output into patient-facing language. Since this
sandbox can't reach `huggingface.co` to download model weights, a
template-based generator was used instead so the pipeline runs end-to-end —
functionally equivalent in structure, minus the generative phrasing. Sample
output (using this run's real Random Forest prediction + SHAP features):

> "Your recent results show a elevated likelihood (82%) of malignant tissue
> characteristics. The factors that most influenced this result were: worst
> area, worst concave points, worst perimeter. This is not a final
> diagnosis — a follow-up biopsy and specialist consultation are
> recommended. Please discuss these results with your care provider, who
> can put them in context with your full medical history."

## 5. Ethics & Bias Considerations

- **Population representativeness:** the Wisconsin dataset (like most public
  medical benchmark datasets) is not demographically audited here. Before
  any real deployment, training data must be checked for representativeness
  across age, sex, ethnicity, and comorbidity profiles — a model that looks
  accurate in aggregate can still fail badly for underrepresented subgroups.
- **Asymmetric error costs:** in diagnostic settings, a false negative
  (missed disease) is typically far more costly than a false positive
  (unnecessary follow-up test). All three models here were optimized on F1,
  but a production system should tune the decision threshold toward higher
  recall and explicitly report the resulting precision/recall trade-off to
  clinicians.
- **Black-box risk:** addressed directly via SHAP — every individual
  prediction can be traced to the features that drove it, which is a
  prerequisite for clinician trust and regulatory scrutiny (e.g. FDA SaMD
  guidance increasingly expects explainability).
- **Synthetic-data caveat:** the CNN, LSTM, and NLP components in this
  specific report ran on synthetic or toy-scale data due to sandbox network
  restrictions, not on the real target datasets. Metrics for those three
  components should be read as *proof the code runs correctly*, not as
  evidence of real-world clinical accuracy. Before deployment, retrain on
  the actual Chest X-Ray Pneumonia dataset, real EHR vitals, and a larger
  corpus of triaged clinical notes.
- **Communication risk:** patient-facing summaries must avoid both
  needlessly alarming and falsely reassuring language. The template/LLM
  output here is deliberately conservative and always redirects the patient
  to a human clinician rather than asserting a diagnosis.
- **Not a medical device:** this prototype has not gone through any clinical
  validation, IRB review, or regulatory clearance and must not be used for
  actual patient care decisions.

## 6. Deployment

`app/streamlit_app.py` wraps the trained Random Forest + StandardScaler +
SHAP explainer + summary generator into a working UI:
- CSV upload or manual slider input for a single patient
- Real-time risk probability + SHAP-based "why" explanation
- Auto-generated plain-English summary
- A CNN imaging tab illustrating the deployment pattern for the X-ray model

Run locally with:
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
