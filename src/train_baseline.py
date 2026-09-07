import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
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

train = pd.read_csv("data/processed/train_clean.csv")
validation = pd.read_csv("data/processed/validation_clean.csv")

print("Training records:", len(train))
print("Validation records:", len(validation))


# ==========================================
# 2. SEPARATE TEXT AND LABELS
# ==========================================

X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==========================================
# 3. CREATE TF-IDF VECTORIZER
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# ==========================================
# 4. FIT TF-IDF ON TRAINING DATA ONLY
# ==========================================

X_train_tfidf = vectorizer.fit_transform(X_train)

X_validation_tfidf = vectorizer.transform(X_validation)

print("Training feature shape:", X_train_tfidf.shape)
print("Validation feature shape:", X_validation_tfidf.shape)


# ==========================================
# 5. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 6. VALIDATION PREDICTIONS
# ==========================================

print("\nMaking validation predictions...")

validation_predictions = model.predict(X_validation_tfidf)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print("\n==============================")
print("BASELINE MODEL RESULTS")
print("==============================")

print(f"Validation Accuracy: {accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions,
        digits=4
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_predictions,
        labels=["negative", "neutral", "positive"]
    )
)


# ==========================================
# 9. SAVE MODEL
# ==========================================

print("\nSaving model...")

joblib.dump(
    model,
    "models/logistic_regression_baseline.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("Model saved to:")
print("models/logistic_regression_baseline.pkl")

print("Vectorizer saved to:")
print("models/tfidf_vectorizer.pkl")


print("\n==============================")
print("BASELINE TRAINING COMPLETE")
print("==============================")
