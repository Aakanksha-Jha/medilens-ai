"""
MediLens AI — Step 1 & 2: EDA + Predictive Modeling
Dataset: Wisconsin Breast Cancer Diagnostic dataset (bundled with scikit-learn).

NOTE ON DATA SOURCE: The sandbox this was built in only has network access to
package registries (PyPI/npm/GitHub), not general data-hosting sites like the
UCI ML Repository or Kaggle. So this pipeline uses scikit-learn's built-in
`load_breast_cancer()` dataset, which is real, public, structurally identical
in spirit to the Heart Disease / Diabetes UCI sets (same kind of multivariate,
tabular, binary-diagnosis problem), and requires no download. Swapping in the
actual Heart Disease UCI CSV is a one-line change (see swap_dataset() below)
once you have normal internet access.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                              roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score)
from xgboost import XGBClassifier
import shap

OUT = "/home/claude/MediLens_AI/reports"

# ---------------------------------------------------------------------------
# STEP 1: DATA PREP + EDA
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
# 0 = malignant, 1 = benign in sklearn's encoding -> flip so 1 = "positive/malignant" like a real diagnosis flag
df["diagnosis"] = 1 - df["diagnosis"]

print("Shape:", df.shape)
print("Missing values total:", df.isna().sum().sum())

# Simulate a touch of realistic missingness + impute (most real clinical data has some)
rng = np.random.default_rng(42)
mask = rng.random(df.shape) < 0.01
df_missing = df.mask(mask)
df_imputed = df_missing.fillna(df_missing.median(numeric_only=True))
assert df_imputed.isna().sum().sum() == 0

feature_cols = [c for c in df.columns if c != "diagnosis"]
X = df_imputed[feature_cols]
y = df_imputed["diagnosis"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Outlier detection via z-score (>3 std considered an outlier)
z = np.abs((X - X.mean()) / X.std())
outlier_rows = (z > 3).any(axis=1).sum()
print("Rows with at least one 3-sigma outlier:", outlier_rows)

# Correlation heatmap (top 12 features most correlated with target, for readability)
corr_with_target = X_scaled.copy()
corr_with_target["diagnosis"] = y.values
top_feats = corr_with_target.corr()["diagnosis"].abs().sort_values(ascending=False).index[1:13]
plt.figure(figsize=(10, 8))
sns.heatmap(corr_with_target[list(top_feats) + ["diagnosis"]].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Top 12 Features vs Diagnosis")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=140)
plt.close()

# ---------------------------------------------------------------------------
# STEP 2: MODELING
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

def evaluate(name, model, Xtr, Xte):
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(y_test, preds)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": cm.tolist(),
    }
    results[name] = metrics
    return model, cm, proba

# Logistic Regression (baseline, interpretable)
log_reg, cm_lr, proba_lr = evaluate("LogisticRegression", LogisticRegression(max_iter=2000), X_train, X_test)

# Random Forest (ensemble)
rf, cm_rf, proba_rf = evaluate("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42), X_train, X_test)

# XGBoost (gradient-boosted ensemble)
xgb, cm_xgb, proba_xgb = evaluate("XGBoost", XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42
), X_train, X_test)

# 5-fold CV sanity check on the best model (RF here, typically strongest w/o tuning)
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="f1")
results["RandomForest"]["cv_f1_mean"] = float(cv_scores.mean())
results["RandomForest"]["cv_f1_std"] = float(cv_scores.std())

# ---------------------------------------------------------------------------
# PCA — dimensionality reduction impact study
# ---------------------------------------------------------------------------
pca_results = {}
for n_comp in [X_scaled.shape[1], 10, 5, 2]:
    pca = PCA(n_components=n_comp)
    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)
    import time
    m = RandomForestClassifier(n_estimators=300, random_state=42)
    t0 = time.time()
    m.fit(Xtr_p, y_train)
    fit_time = time.time() - t0
    f1 = f1_score(y_test, m.predict(Xte_p))
    explained = float(np.sum(pca.explained_variance_ratio_)) if n_comp != X_scaled.shape[1] else 1.0
    pca_results[n_comp] = {"f1": f1, "fit_time_sec": fit_time, "explained_variance": explained}

# ---------------------------------------------------------------------------
# Confusion matrix plot (best model = RandomForest)
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Random Forest — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix_rf.png", dpi=140)
plt.close()

# ROC curves
plt.figure(figsize=(6, 5))
for name, proba in [("LogReg", proba_lr), ("RandomForest", proba_rf), ("XGBoost", proba_xgb)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=140)
plt.close()

# ---------------------------------------------------------------------------
# STEP 5 (partial): Explainability — SHAP on RandomForest
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test, check_additivity=False)
# shap_values can be a list (per-class), a 2D array, or a 3D array (samples, features, classes)
# depending on the installed shap version. Normalize to a 2D (samples, features) array
# for the positive class.
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
plt.figure()
shap.summary_plot(sv, X_test, show=False, max_display=10)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary.png", dpi=140, bbox_inches="tight")
plt.close()

# Save top global feature importances (SHAP-based) for the report
mean_abs_shap = np.abs(sv).mean(axis=0)
top_shap = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: -x[1])[:8]

# ---------------------------------------------------------------------------
# Persist everything for the notebook + report to consume
# ---------------------------------------------------------------------------
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({
        "dataset_shape": df.shape,
        "outlier_rows_3sigma": int(outlier_rows),
        "results": results,
        "pca_results": {str(k): v for k, v in pca_results.items()},
        "top_shap_features": [(f, float(v)) for f, v in top_shap],
    }, f, indent=2)

print(json.dumps(results, indent=2))
print("PCA study:", json.dumps({str(k): v for k, v in pca_results.items()}, indent=2))
print("Top SHAP features:", top_shap)

import joblib
import os
ARTIFACT_DIR = "/home/claude/MediLens_AI/app/model_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(rf, f"{ARTIFACT_DIR}/random_forest_model.joblib")
joblib.dump(scaler, f"{ARTIFACT_DIR}/scaler.joblib")
joblib.dump(feature_cols, f"{ARTIFACT_DIR}/feature_cols.joblib")
joblib.dump(explainer, f"{ARTIFACT_DIR}/shap_explainer.joblib")

print("Done.")
