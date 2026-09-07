import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression


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
# 2. PREPARE DATA
# ==================================================

X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==================================================
# 3. CREATE FEATURES
# ==================================================

print("\nCreating Word + Character TF-IDF...")


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


# ==================================================
# 4. TRAIN CURRENT BEST MODEL
# ==================================================

print("\nTraining current best model...")


model = LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight="balanced",
    random_state=42
)


model.fit(
    X_train_features,
    y_train
)


# ==================================================
# 5. PREDICTIONS
# ==================================================

print("\nMaking predictions...")


predictions = model.predict(
    X_validation_features
)

probabilities = model.predict_proba(
    X_validation_features
)

confidence = probabilities.max(
    axis=1
)


# ==================================================
# 6. CREATE RESULTS
# ==================================================

results = validation.copy()

results["predicted_label"] = predictions

results["confidence"] = confidence

results["correct"] = (
    results["label"]
    ==
    results["predicted_label"]
)


# ==================================================
# 7. FILTER NEUTRAL EXAMPLES
# ==================================================

neutral = results[
    results["label"] == "neutral"
].copy()


neutral = neutral.sort_values(
    "confidence"
)


# ==================================================
# 8. DISPLAY SUMMARY
# ==================================================

print("\n")
print("=" * 70)
print("NEUTRAL CLASS ANALYSIS")
print("=" * 70)


print(
    "\nTotal neutral validation examples:",
    len(neutral)
)


print(
    "Correctly classified:",
    neutral["correct"].sum()
)


print(
    "Incorrectly classified:",
    (~neutral["correct"]).sum()
)


print("\nPredicted labels for actual neutral tweets:")

print(
    neutral["predicted_label"]
    .value_counts()
)


# ==================================================
# 9. DISPLAY ALL NEUTRAL EXAMPLES
# ==================================================

print("\n")
print("=" * 70)
print("ALL NEUTRAL VALIDATION EXAMPLES")
print("=" * 70)


for _, row in neutral.iterrows():

    print("\nTWEET:")
    print(row["tweet"])

    print("\nPREDICTED:")
    print(row["predicted_label"])

    print("\nCONFIDENCE:")
    print(
        f"{row['confidence']:.4f}"
    )

    print("\nCORRECT:")
    print(row["correct"])

    print("-" * 60)


# ==================================================
# 10. SAVE REPORT
# ==================================================

neutral.to_csv(
    "reports/neutral_analysis.csv",
    index=False
)


print("\n")
print("=" * 70)
print("NEUTRAL ANALYSIS COMPLETE")
print("=" * 70)


print(
    "\nSaved to:"
)

print(
    "reports/neutral_analysis.csv"
)
