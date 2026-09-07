import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading cleaned datasets...")

train = pd.read_csv(
    "data/processed/train_clean.csv"
)

validation = pd.read_csv(
    "data/processed/validation_clean.csv"
)


print("Training records:", len(train))
print("Validation records:", len(validation))


# ==========================================
# 2. PREPARE DATA
# ==========================================

X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==========================================
# 3. BEST CONFIGURATION
# ==========================================

print("\nUsing best configuration...")

print("Word n-gram: (1, 2)")
print("Character n-gram: (3, 5)")
print("C: 1.0")


# ==========================================
# 4. CREATE WORD TF-IDF
# ==========================================

word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# ==========================================
# 5. CREATE CHARACTER TF-IDF
# ==========================================

char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)


# ==========================================
# 6. COMBINE FEATURES
# ==========================================

features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])


print("\nCreating combined TF-IDF features...")


X_train_features = features.fit_transform(
    X_train
)

X_validation_features = features.transform(
    X_validation
)


print(
    "Training feature shape:",
    X_train_features.shape
)

print(
    "Validation feature shape:",
    X_validation_features.shape
)


# ==========================================
# 7. TRAIN BEST MODEL
# ==========================================

print("\nTraining best Logistic Regression model...")


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    random_state=42
)


model.fit(
    X_train_features,
    y_train
)


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

print("\nMaking validation predictions...")


predictions = model.predict(
    X_validation_features
)


probabilities = model.predict_proba(
    X_validation_features
)


confidence = probabilities.max(
    axis=1
)


# ==========================================
# 9. BASIC PERFORMANCE
# ==========================================

accuracy = accuracy_score(
    y_validation,
    predictions
)


print("\n")
print("=" * 60)
print("BEST MODEL EVALUATION")
print("=" * 60)


print(
    f"\nValidation Accuracy: {accuracy:.4f}"
)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")


print(
    classification_report(
        y_validation,
        predictions,
        digits=4
    )
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")


cm = confusion_matrix(
    y_validation,
    predictions,
    labels=[
        "negative",
        "neutral",
        "positive"
    ]
)


print(
    pd.DataFrame(
        cm,
        index=[
            "Actual Negative",
            "Actual Neutral",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Neutral",
            "Predicted Positive"
        ]
    )
)


# ==========================================
# 12. ADD PREDICTIONS TO DATAFRAME
# ==========================================

results = validation.copy()


results["predicted_label"] = predictions

results["confidence"] = confidence


results["correct"] = (
    results["label"]
    ==
    results["predicted_label"]
)


# ==========================================
# 13. ERROR COUNT
# ==========================================

errors = results[
    results["correct"] == False
]


print("\nTotal incorrect predictions:")

print(
    len(errors)
)


# ==========================================
# 14. CONFUSION PAIRS
# ==========================================

print("\n")
print("=" * 60)
print("COMMON CONFUSIONS")
print("=" * 60)


confusions = (
    errors
    .groupby(
        [
            "label",
            "predicted_label"
        ]
    )
    .size()
    .sort_values(
        ascending=False
    )
)


print(
    confusions
)


# ==========================================
# 15. LOW-CONFIDENCE PREDICTIONS
# ==========================================

print("\n")
print("=" * 60)
print("LOW-CONFIDENCE PREDICTIONS")
print("=" * 60)


low_confidence = (
    results
    .sort_values(
        "confidence"
    )
    .head(30)
)


for _, row in low_confidence.iterrows():

    print("\nTWEET:")

    print(
        row["tweet"]
    )

    print("\nACTUAL:")

    print(
        row["label"]
    )

    print("\nPREDICTED:")

    print(
        row["predicted_label"]
    )

    print("\nCONFIDENCE:")

    print(
        f"{row['confidence']:.4f}"
    )

    print(
        "-" * 50
    )


# ==========================================
# 16. SAVE ALL PREDICTIONS
# ==========================================

results.to_csv(
    "reports/best_model_predictions.csv",
    index=False
)


# ==========================================
# 17. SAVE ERRORS
# ==========================================

errors.to_csv(
    "reports/best_model_errors.csv",
    index=False
)


print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)


print(
    "\nAll predictions saved to:"
)

print(
    "reports/best_model_predictions.csv"
)


print(
    "\nErrors saved to:"
)

print(
    "reports/best_model_errors.csv"
)
