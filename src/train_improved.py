import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


print("Loading cleaned datasets...")

train = pd.read_csv("data/processed/train_clean.csv")
validation = pd.read_csv("data/processed/validation_clean.csv")

print("Training records:", len(train))
print("Validation records:", len(validation))


# ==========================================
# PREPARE TEXT AND LABELS
# ==========================================

X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==========================================
# CREATE WORD + CHARACTER TF-IDF FEATURES
# ==========================================

print("\nCreating combined TF-IDF features...")


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


print("Fitting TF-IDF features...")

X_train_features = features.fit_transform(X_train)
X_validation_features = features.transform(X_validation)


print("Training feature shape:", X_train_features.shape)
print("Validation feature shape:", X_validation_features.shape)


# ==========================================
# TRAIN IMPROVED LOGISTIC REGRESSION
# ==========================================

print("\nTraining improved Logistic Regression model...")


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=2.0
)


model.fit(X_train_features, y_train)


# ==========================================
# VALIDATION PREDICTIONS
# ==========================================

print("\nMaking validation predictions...")

validation_predictions = model.predict(X_validation_features)


# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


print("\n==============================")
print("IMPROVED MODEL RESULTS")
print("==============================")

print(
    f"Validation Accuracy: {accuracy:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_predictions
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

print("\nSaving improved model...")


joblib.dump(
    model,
    "models/logistic_regression_improved.pkl"
)


joblib.dump(
    features,
    "models/tfidf_combined_vectorizer.pkl"
)


print("Model saved to:")
print("models/logistic_regression_improved.pkl")

print("Vectorizer saved to:")
print("models/tfidf_combined_vectorizer.pkl")


print("\n==============================")
print("IMPROVED TRAINING COMPLETE")
print("==============================")
