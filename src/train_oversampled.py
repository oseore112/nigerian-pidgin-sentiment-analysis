import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import RandomOverSampler


# ==================================================
# 1. LOAD DATA
# ==================================================

print("Loading cleaned datasets...")

train = pd.read_csv(
    "data/processed/train_clean.csv"
)

validation = pd.read_csv(
    "data/processed/validation_clean.csv"
)


print("Training records:", len(train))
print("Validation records:", len(validation))


# ==================================================
# 2. SHOW ORIGINAL DISTRIBUTION
# ==================================================

print("\n")
print("=" * 60)
print("ORIGINAL TRAINING DISTRIBUTION")
print("=" * 60)

print(
    train["label"].value_counts()
)


# ==================================================
# 3. PREPARE DATA
# ==================================================

X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==================================================
# 4. CREATE FEATURES
# ==================================================

print("\nCreating Word + Character TF-IDF features...")


word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)


features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])


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


# ==================================================
# 5. OVERSAMPLE TRAINING DATA ONLY
# ==================================================

print("\nOversampling minority classes...")


oversampler = RandomOverSampler(
    random_state=42
)


X_train_resampled, y_train_resampled = (
    oversampler.fit_resample(
        X_train_features,
        y_train
    )
)


print("\n")
print("=" * 60)
print("RESAMPLED TRAINING DISTRIBUTION")
print("=" * 60)


print(
    pd.Series(
        y_train_resampled
    ).value_counts()
)


print(
    "\nResampled feature shape:",
    X_train_resampled.shape
)


# ==================================================
# 6. TRAIN MODEL
# ==================================================

print("\nTraining Logistic Regression model...")


model = LogisticRegression(
    max_iter=2000,
    C=1.0,
    random_state=42
)


model.fit(
    X_train_resampled,
    y_train_resampled
)


# ==================================================
# 7. MAKE PREDICTIONS
# ==================================================

print("\nMaking validation predictions...")


predictions = model.predict(
    X_validation_features
)


# ==================================================
# 8. EVALUATE
# ==================================================

accuracy = accuracy_score(
    y_validation,
    predictions
)


macro_f1 = f1_score(
    y_validation,
    predictions,
    average="macro"
)


weighted_f1 = f1_score(
    y_validation,
    predictions,
    average="weighted"
)


print("\n")
print("=" * 60)
print("OVERSAMPLED MODEL RESULTS")
print("=" * 60)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print(
    f"Macro F1: {macro_f1:.4f}"
)


print(
    f"Weighted F1: {weighted_f1:.4f}"
)


print("\nClassification Report:\n")


print(
    classification_report(
        y_validation,
        predictions,
        digits=4,
        zero_division=0
    )
)


# ==================================================
# 9. CONFUSION MATRIX
# ==================================================

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


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results = pd.DataFrame({
    "tweet": validation["tweet"],
    "tweet_clean": validation["tweet_clean"],
    "actual_label": y_validation,
    "predicted_label": predictions
})


results.to_csv(
    "reports/oversampled_predictions.csv",
    index=False
)


summary = pd.DataFrame({
    "Model": [
        "Best Tuned Model",
        "Oversampled Model"
    ],
    "Accuracy": [
        0.6767,
        accuracy
    ],
    "Macro F1": [
        0.5515,
        macro_f1
    ],
    "Weighted F1": [
        0.6765,
        weighted_f1
    ]
})


summary.to_csv(
    "reports/oversampling_comparison.csv",
    index=False
)


print("\n")
print("=" * 60)
print("OVERSAMPLING EXPERIMENT COMPLETE")
print("=" * 60)


print(
    "\nPredictions saved to:"
)

print(
    "reports/oversampled_predictions.csv"
)


print(
    "\nComparison saved to:"
)

print(
    "reports/oversampling_comparison.csv"
)
