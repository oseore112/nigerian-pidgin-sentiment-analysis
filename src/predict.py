import os
import sys
import joblib
from scipy.sparse import hstack

# ============================================================
# LOAD FINAL ARTIFACTS
# (paths are built from this file's own location, so this
# work no matter which folder you run the progress from)
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models",
                    "final_logistic_regression.pkl"))
word_vectorizer = joblib.load(os.path.join(
    BASE_DIR, "models", "final_word_tfidf_vectorizer.pkl"))
char_vectorizer = joblib.load(os.path.join(
    BASE_DIR, "models", "final_char_tfidf_vectorizer.pkl"))

LABELS = model.classes_  # e.g. ['negative' 'neutral' 'positive']

# ============================================================
# TEXT CLEANING — MUST MATCH TRAINING PREPROCESSING EXACTLY
# ============================================================


def clean_tweet(text):
    # <-- PASTE YOUR ACTUAL tweet_clean LOGIC HERE -->
    # placeholder only — replace before using
    return text.strip().lower()


# ============================================================
# PREDICT
# ============================================================

def predict_sentiment(text):
    cleaned = clean_tweet(text)

    word_feat = word_vectorizer.transform([cleaned])
    char_feat = char_vectorizer.transform([cleaned])
    features = hstack([word_feat, char_feat])

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    prob_dict = dict(zip(LABELS, probs))
    confidence = prob_dict[pred]

    return pred, confidence, prob_dict


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter Nigerian Pidgin text: ")

    label, confidence, probs = predict_sentiment(text)

    print(f"\nSentiment: {label.upper()}")
    print(f"Confidence: {confidence * 100:.1f}%\n")
    for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"{cls:10s} {p*100:5.1f}%")
