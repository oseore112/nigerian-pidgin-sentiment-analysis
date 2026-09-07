import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


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
# 3. CREATE WORD + CHARACTER TF-IDF
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
# 4. CLASS WEIGHT EXPERIMENTS
# ==================================================

experiments = [

    {
        "name": "Equal Weights",
        "weights": {
            "negative": 1.0,
            "neutral": 1.0,
            "positive": 1.0
        }
    },

    {
        "name": "Neutral 1.5x",
        "weights": {
            "negative": 1.0,
            "neutral": 1.5,
            "positive": 1.0
        }
    },

    {
        "name": "Neutral 2x",
        "weights": {
            "negative": 1.0,
            "neutral": 2.0,
            "positive": 1.0
        }
    },

    {
        "name": "Neutral 3x",
        "weights": {
            "negative": 1.0,
            "neutral": 3.0,
            "positive": 1.0
        }
    },

    {
        "name": "Balanced",
        "weights": "balanced"
    }
]


results = []


# ==================================================
# 5. RUN EXPERIMENTS
# ==================================================

for experiment in experiments:

    print("\n")
    print("=" * 60)

    print(
        experiment["name"]
    )

    print("=" * 60)

    print(
        "\nClass weights:",
        experiment["weights"]
    )

    # ----------------------------------------------
    # TRAIN MODEL
    # ----------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight=experiment["weights"],
        random_state=42
    )

    print("\nTraining...")

    model.fit(
        X_train_features,
        y_train
    )

    # ----------------------------------------------
    # PREDICT
    # ----------------------------------------------

    print("Making predictions...")

    predictions = model.predict(
        X_validation_features
    )

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

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

    report = classification_report(
        y_validation,
        predictions,
        output_dict=True,
        zero_division=0
    )

    neutral_f1 = report[
        "neutral"
    ]["f1-score"]

    negative_f1 = report[
        "negative"
    ]["f1-score"]

    positive_f1 = report[
        "positive"
    ]["f1-score"]

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print(
        f"Negative F1: {negative_f1:.4f}"
    )

    print(
        f"Neutral F1: {neutral_f1:.4f}"
    )

    print(
        f"Positive F1: {positive_f1:.4f}"
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

    # ----------------------------------------------
    # SAVE RESULT
    # ----------------------------------------------

    results.append({

        "Experiment":
            experiment["name"],

        "Accuracy":
            accuracy,

        "Macro F1":
            macro_f1,

        "Weighted F1":
            weighted_f1,

        "Negative F1":
            negative_f1,

        "Neutral F1":
            neutral_f1,

        "Positive F1":
            positive_f1
    })


# ==================================================
# 6. RESULTS TABLE
# ==================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)


print("\n")
print("=" * 70)
print("CLASS WEIGHT EXPERIMENT RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ==================================================
# 7. SAVE RESULTS
# ==================================================

results_df.to_csv(
    "reports/class_weight_results.csv",
    index=False
)


print("\nResults saved to:")

print(
    "reports/class_weight_results.csv"
)


print("\n")
print("=" * 70)
print("CLASS WEIGHT TUNING COMPLETE")
print("=" * 70)
