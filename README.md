# MediLens AI — Prototype

An end-to-end AI diagnostic & insight system prototype: tabular risk
prediction, computer vision (CNN + transfer learning), sequence modeling
(LSTM), NLP note classification, generative plain-English summaries, SHAP
explainability, and a Streamlit deployment app.

## ⚠️ Scope & honesty notes (read this first)

This was built in a sandboxed environment with **no general internet
access** — only package registries (PyPI, npm, GitHub) were reachable, not
Kaggle, the UCI ML Repository, or huggingface.co. So:

| Component | What the brief suggested | What was actually used | Why |
|---|---|---|---|
| Tabular model | Heart Disease / Diabetes UCI dataset | scikit-learn's built-in Wisconsin Breast Cancer dataset | UCI Repository unreachable; this is a real public dataset, same problem shape, no download needed |
| CNN images | Real chest X-rays (pneumonia vs normal) | Procedurally-generated synthetic images | Kaggle unreachable |
| MobileNetV2 weights | Pretrained ImageNet weights | Random initialization (documented fallback) | `storage.googleapis.com` blocked by sandbox network policy |
| LSTM vitals | Real EHR/wearable vitals | Synthetic per-patient time series | No real vitals dataset available offline |
| Summarization LLM | HuggingFace `distilbart`/GPT/BERT | Template-based generator (real HF pipeline code included, commented as the production path) | `huggingface.co` unreachable to download weights |
| GitHub repo link | Hosted repo URL | This folder, ready to `git init && push` | No ability to create/host a repo from this environment |
| Hosted demo link | Live Streamlit/HF Spaces URL | Local `streamlit run` app, verified to boot successfully | No ability to deploy a public URL from this environment |

Every script **actually runs** and produces **real numbers** — nothing here
is fabricated output. Where real datasets weren't reachable, the code is
still production-ready: swapping in the real data source is a one-to-a-few
line change per script, called out explicitly in each file's docstring.

## Project structure

```
MediLens_AI/
├── notebooks/
│   └── 01_MediLens_AI_EDA_and_Modeling.ipynb   # Steps 1-2 + summaries of 3-4, with real embedded outputs
├── scripts/
│   ├── run_eda_ml.py                # Steps 1-2: EDA, LogReg/RF/XGBoost, PCA, SHAP (executed)
│   ├── cnn_xray_classifier.py       # Step 3: CNN + MobileNetV2 transfer learning (executed)
│   ├── lstm_vitals_forecast.py      # Step 3: LSTM vitals forecasting (executed)
│   ├── nlp_notes_and_summary.py     # Step 4: TF-IDF classification + summary generation (executed)
│   └── _build_notebook.py           # Builds the .ipynb from real captured outputs
├── app/
│   ├── streamlit_app.py             # Step 5: deployment app (verified to boot)
│   └── model_artifacts/             # Saved RandomForest, scaler, SHAP explainer
├── reports/                         # All generated plots + metrics JSON
├── requirements.txt
├── MODEL_REPORT.md                  # Accuracy/precision + ethics write-up
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results

```bash
python scripts/run_eda_ml.py              # ~10s, writes reports/ + app/model_artifacts/
python scripts/cnn_xray_classifier.py      # ~2-3 min on CPU
python scripts/lstm_vitals_forecast.py     # ~30s
python scripts/nlp_notes_and_summary.py    # ~5s
python scripts/_build_notebook.py          # rebuilds the notebook from reports/
```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

## Push to your own GitHub repo

```bash
cd MediLens_AI
git init
git add .
git commit -m "MediLens AI prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/medilens-ai.git
git push -u origin main
```

## Deploy a live demo (choose one)

- **Streamlit Community Cloud** (free): push to GitHub, then at
  share.streamlit.io connect the repo and point it at `app/streamlit_app.py`.
- **Hugging Face Spaces**: create a new Space (SDK: Streamlit), push this
  repo's contents to it.
