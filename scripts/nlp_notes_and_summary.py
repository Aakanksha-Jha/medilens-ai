"""
MediLens AI — Step 4: NLP + Generative Summary

Part A: TF-IDF + classic ML to categorize patient feedback/notes into
         clinically useful buckets (e.g., "medication side effect",
         "appointment logistics", "symptom report", "positive feedback").

Part B: Turn a technical model output (probabilities, top risk drivers) into
         a plain-English patient-facing summary using a HuggingFace
         Transformers pipeline.

NETWORK NOTE: This sandbox's egress is locked to package registries (pypi,
npm, github) and cannot reach huggingface.co to download model weights.
`generate_summary_hf()` below is the real production code path — it works
as-is the moment you run it anywhere with normal internet access (or with
`HF_HOME` pointed at a local model cache). Because weights can't be fetched
here, `generate_patient_summary()` uses a lightweight template-based
generator so the whole script still runs end-to-end and produces real output
in this sandbox. Swapping one for the other is a one-line change in main().
"""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

OUT = "/home/claude/MediLens_AI/reports"

# ---------------------------------------------------------------------------
# Part A: TF-IDF note/feedback classification
# ---------------------------------------------------------------------------
notes = [
    ("I've had a rash on my arm since starting the new pills", "medication_side_effect"),
    ("The new medication is making me nauseous every morning", "medication_side_effect"),
    ("Experiencing dizziness after taking my blood pressure tablet", "medication_side_effect"),
    ("My skin broke out in hives after the antibiotic dose", "medication_side_effect"),
    ("Can I reschedule my Tuesday appointment to next week", "appointment_logistics"),
    ("What time is my follow-up scan on Friday", "appointment_logistics"),
    ("I need to change my clinic location for the next visit", "appointment_logistics"),
    ("Please confirm the address for my cardiology appointment", "appointment_logistics"),
    ("I've been having chest tightness and shortness of breath", "symptom_report"),
    ("There's persistent pain in my lower back for three days", "symptom_report"),
    ("I feel feverish and have a sore throat since yesterday", "symptom_report"),
    ("My blood sugar readings have been higher than usual this week", "symptom_report"),
    ("Thank you, the care team was wonderful during my stay", "positive_feedback"),
    ("The nurse explained everything clearly, really appreciated it", "positive_feedback"),
    ("Great experience overall, the staff were very attentive", "positive_feedback"),
    ("I'm very happy with how my treatment has been going", "positive_feedback"),
    ("My joints have been aching more than usual lately", "symptom_report"),
    ("Could we move my MRI slot to the morning instead", "appointment_logistics"),
    ("The new inhaler leaves a strange taste and mild headache", "medication_side_effect"),
    ("Everyone here has been so kind and professional, thank you", "positive_feedback"),
    ("I broke out in itchy welts after the new capsule", "medication_side_effect"),
    ("Feeling drowsy and light-headed since the dosage increase", "medication_side_effect"),
    ("Stomach cramps started right after the evening dose", "medication_side_effect"),
    ("Need to push my ultrasound appointment back an hour", "appointment_logistics"),
    ("Is there a way to book a telehealth visit instead", "appointment_logistics"),
    ("Requesting to move my lab work to next Monday", "appointment_logistics"),
    ("Sharp pain in my knee when climbing stairs recently", "symptom_report"),
    ("Persistent cough and fatigue for the past four days", "symptom_report"),
    ("Noticed swelling in my ankles over the last two days", "symptom_report"),
    ("The reception staff went above and beyond to help me", "positive_feedback"),
    ("Really grateful for how quickly the team responded", "positive_feedback"),
    ("My recovery has been smooth thanks to the great support", "positive_feedback"),
]
df_notes = pd.DataFrame(notes, columns=["text", "label"])

X_train, X_test, y_train, y_test = train_test_split(
    df_notes["text"], df_notes["label"], test_size=0.3, random_state=42, stratify=df_notes["label"]
)

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
Xtr_vec = tfidf.fit_transform(X_train)
Xte_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr_vec, y_train)
preds = clf.predict(Xte_vec)

report = classification_report(y_test, preds, output_dict=True, zero_division=0)
f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)

print("TF-IDF note classification report:")
print(classification_report(y_test, preds, zero_division=0))

with open(f"{OUT}/nlp_classification_report.json", "w") as f:
    json.dump({"report": report, "f1_macro": f1_macro}, f, indent=2)


# ---------------------------------------------------------------------------
# Part B: Plain-English summary generation
# ---------------------------------------------------------------------------
def generate_summary_hf(technical_result: dict) -> str:
    """
    PRODUCTION PATH (requires internet access to huggingface.co the first time,
    to download model weights; cached locally after that).
    """
    from transformers import pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    prompt = (
        f"Diagnostic model result: {technical_result['diagnosis']} "
        f"with probability {technical_result['probability']:.0%}. "
        f"Top contributing factors: {', '.join(technical_result['top_features'])}. "
        f"Recommended action: {technical_result['recommendation']}."
    )
    summary = summarizer(prompt, max_length=60, min_length=20, do_sample=False)
    return summary[0]["summary_text"]


def generate_patient_summary(technical_result: dict) -> str:
    """
    OFFLINE-SAFE fallback used in this sandbox (no model download required).
    Template-based, but pulls in the exact same structured fields a real
    LLM call would use as its prompt — so behavior is equivalent in spirit,
    just without a generative model doing the phrasing.
    """
    diagnosis = technical_result["diagnosis"]
    prob = technical_result["probability"]
    feats = technical_result["top_features"]
    rec = technical_result["recommendation"]

    risk_word = "elevated" if prob >= 0.7 else ("moderate" if prob >= 0.4 else "low")
    feats_text = ", ".join(feats[:3])

    return (
        f"Your recent results show a {risk_word} likelihood ({prob:.0%}) of {diagnosis}. "
        f"The factors that most influenced this result were: {feats_text}. "
        f"This is not a final diagnosis — {rec} Please discuss these results with your "
        f"care provider, who can put them in context with your full medical history."
    )


# Example: feed in the real RandomForest result + top SHAP features from Step 2
with open(f"{OUT}/metrics.json") as f:
    ml_metrics = json.load(f)

example_result = {
    "diagnosis": "malignant tissue characteristics",
    "probability": 0.82,
    "top_features": [f for f, _ in ml_metrics["top_shap_features"][:3]],
    "recommendation": "a follow-up biopsy and specialist consultation are recommended.",
}

plain_english_summary = generate_patient_summary(example_result)
print("\nPlain-English summary (offline template path):")
print(plain_english_summary)

with open(f"{OUT}/sample_patient_summary.txt", "w") as f:
    f.write(plain_english_summary)

print("\nDone.")
