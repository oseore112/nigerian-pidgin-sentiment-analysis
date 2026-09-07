import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD VALIDATION DATA
# ==========================================

print("Loading validation dataset...")

validation = pd.read_csv(
    "data/processed/validation_clean.csv"
)

print("Validation records:", len(validation))


# ==========================================
# 2. LOAD MODEL AND VECTORIZER
# ==========================================

print("\nLoading trained model...")

model = joblib.load(
    "models/logistic_regression_baseline.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================================
# 3. CREATE FEATURES
# ==========================================

X_validation = vectorizer.transform(
    validation["tweet_clean"]
)

y_validation = validation["label"]


# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(X_validation)


# ==========================================
# 5. ADD PREDICTIONS TO DATAFRAME
# ==========================================

validation["predicted_label"] = predictions

validation["correct"] = (
    validation["label"]
    == validation["predicted_label"]
)


# ==========================================
# 6. OVERALL ACCURACY
# ==========================================

accuracy = accuracy_score(
    validation["label"],
    validation["predicted_label"]
)

print("\n==============================")
print("ERROR ANALYSIS")
print("==============================")

print(
    f"Validation Accuracy: {accuracy:.4f}"
)


# ==========================================
# 7. TOTAL ERRORS
# ==========================================

errors = validation[
    validation["correct"] == False
].copy()

print(
    "\nTotal incorrect predictions:",
    len(errors)
)


# ==========================================
# 8. MOST COMMON CONFUSIONS
# ==========================================

print("\n==============================")
print("COMMON CONFUSIONS")
print("==============================")

confusions = (
    errors
    .groupby(
        ["label", "predicted_label"]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(confusions)


# ==========================================
# 9. SHOW NEGATIVE → POSITIVE ERRORS
# ==========================================

print("\n==============================")
print("NEGATIVE → POSITIVE ERRORS")
print("==============================")

negative_positive = errors[
    (errors["label"] == "negative")
    &
    (errors["predicted_label"] == "positive")
]

for _, row in negative_positive.head(20).iterrows():

    print("\nTWEET:")
    print(row["tweet"])

    print("CLEANED:")
    print(row["tweet_clean"])

    print("ACTUAL:")
    print(row["label"])

    print("PREDICTED:")
    print(row["predicted_label"])

    print("-" * 60)


# ==========================================
# 10. POSITIVE → NEGATIVE ERRORS
# ==========================================

print("\n==============================")
print("POSITIVE → NEGATIVE ERRORS")
print("==============================")

positive_negative = errors[
    (errors["label"] == "positive")
    &
    (errors["predicted_label"] == "negative")
]

for _, row in positive_negative.head(20).iterrows():

    print("\nTWEET:")
    print(row["tweet"])

    print("CLEANED:")
    print(row["tweet_clean"])

    print("ACTUAL:")
    print(row["label"])

    print("PREDICTED:")
    print(row["predicted_label"])

    print("-" * 60)


# ==========================================
# 11. NEUTRAL ERRORS
# ==========================================

print("\n==============================")
print("NEUTRAL ERRORS")
print("==============================")

neutral_errors = errors[
    errors["label"] == "neutral"
]

for _, row in neutral_errors.head(20).iterrows():

    print("\nTWEET:")
    print(row["tweet"])

    print("CLEANED:")
    print(row["tweet_clean"])

    print("ACTUAL:")
    print(row["label"])

    print("PREDICTED:")
    print(row["predicted_label"])

    print("-" * 60)


# ==========================================
# 12. SAVE ALL ERRORS
# ==========================================

errors.to_csv(
    "data/interim/baseline_errors.csv",
    index=False
)

print("\n==============================")
print("ERROR ANALYSIS COMPLETE")
print("==============================")

print(
    "\nSaved errors to:"
)

print(
    "data/interim/baseline_errors.csv"
)
