import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.sparse import hstack


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned datasets...")

train = pd.read_csv("data/processed/train_clean.csv")
validation = pd.read_csv("data/processed/validation_clean.csv")

print("Training records:", len(train))
print("Validation records:", len(validation))


# ============================================================
# PREPARE TEXT
# ============================================================

X_train_text = train["tweet_clean"].fillna("")
X_val_text = validation["tweet_clean"].fillna("")

y_train = train["label"]
y_val = validation["label"]


# ============================================================
# WORD TF-IDF
# ============================================================

print("\nCreating Word + Character TF-IDF features...")


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_val_word = word_vectorizer.transform(X_val_text)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=16000,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_val_char = char_vectorizer.transform(X_val_text)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_train = hstack([X_train_word, X_train_char])
X_val = hstack([X_val_word, X_val_char])

print("Training feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = {
    "Neutral 3.5x": {
        "negative": 1.0,
        "neutral": 3.5,
        "positive": 1.0
    },

    "Neutral 4x": {
        "negative": 1.0,
        "neutral": 4.0,
        "positive": 1.0
    },

    "Neutral 5x": {
        "negative": 1.0,
        "neutral": 5.0,
        "positive": 1.0
    },

    "Neutral 6x": {
        "negative": 1.0,
        "neutral": 6.0,
        "positive": 1.0
    },

    "Neutral 7x": {
        "negative": 1.0,
        "neutral": 7.0,
        "positive": 1.0
    }
}


results = []


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for name, weights in experiments.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("\nClass weights:", weights)

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight=weights,
        solver="liblinear"
    )

    print("\nTraining...")

    model.fit(X_train, y_train)

    print("Making predictions...")

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted"
    )

    report = classification_report(
        y_val,
        predictions,
        output_dict=True,
        zero_division=0
    )

    negative_f1 = report["negative"]["f1-score"]
    neutral_f1 = report["neutral"]["f1-score"]
    positive_f1 = report["positive"]["f1-score"]

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("Negative F1:", round(negative_f1, 4))
    print("Neutral F1:", round(neutral_f1, 4))
    print("Positive F1:", round(positive_f1, 4))

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0
        )
    )

    results.append({
        "Experiment": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Negative F1": negative_f1,
        "Neutral F1": neutral_f1,
        "Positive F1": positive_f1
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n")
print("=" * 70)
print("NEUTRAL WEIGHT TUNING RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "reports/neutral_weight_results.csv",
    index=False
)

print("\nResults saved to:")
print("reports/neutral_weight_results.csv")

print("\n")
print("=" * 70)
print("NEUTRAL WEIGHT TUNING COMPLETE")
print("=" * 70)
