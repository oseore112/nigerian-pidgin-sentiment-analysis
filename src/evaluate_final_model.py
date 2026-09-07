from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train = pd.read_csv("data/processed/train_clean.csv")
test = pd.read_csv("data/processed/test_clean.csv")

print(f"Training records: {len(train)}")
print(f"Test records: {len(test)}")


# ============================================================
# TEXT AND LABELS
# ============================================================

X_train_text = train["tweet_clean"].fillna("")
y_train = train["label"]

X_test_text = test["tweet_clean"].fillna("")
y_test = test["label"]


# ============================================================
# TF-IDF CONFIGURATION
# ============================================================

print("\nCreating final Word + Character TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

print("Fitting vectorizers on training data...")

X_train_word = word_vectorizer.fit_transform(X_train_text)
X_test_word = word_vectorizer.transform(X_test_text)

X_train_char = char_vectorizer.fit_transform(X_train_text)
X_test_char = char_vectorizer.transform(X_test_text)


# ============================================================
# COMBINE FEATURES
# ============================================================

X_train = hstack([X_train_word, X_train_char])
X_test = hstack([X_test_word, X_test_char])

print(f"Training feature shape: {X_train.shape}")
print(f"Test feature shape: {X_test.shape}")


# ============================================================
# LOAD / TRAIN FINAL MODEL
# ============================================================


print("\nTraining final Logistic Regression model...")

model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    class_weight=None,
    solver="lbfgs"
)

model.fit(X_train, y_train)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("Making predictions on untouched test set...")

y_pred = model.predict(X_test)


# ============================================================
# FINAL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

joblib.dump(
    model,
    "models/final_logistic_regression.pkl"
)

joblib.dump(
    word_vectorizer,
    "models/final_word_tfidf_vectorizer.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/final_char_tfidf_vectorizer.pkl"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

results = test.copy()

results["predicted_label"] = y_pred

results.to_csv(
    "reports/final_test_predictions.csv",
    index=False
)


print("\nFiles saved:")
print("models/final_logistic_regression.pkl")
print("models/final_word_tfidf_vectorizer.pkl")
print("models/final_char_tfidf_vectorizer.pkl")
print("reports/final_test_predictions.csv")

print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION COMPLETE")
print("=" * 70)
