import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
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


X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==========================================
# 2. EXPERIMENT CONFIGURATIONS
# ==========================================

experiments = [

    {
        "name": "Experiment 1",
        "word_ngram": (1, 2),
        "char_ngram": (3, 5),
        "C": 1.0
    },

    {
        "name": "Experiment 2",
        "word_ngram": (1, 2),
        "char_ngram": (3, 5),
        "C": 2.0
    },

    {
        "name": "Experiment 3",
        "word_ngram": (1, 2),
        "char_ngram": (3, 5),
        "C": 4.0
    },

    {
        "name": "Experiment 4",
        "word_ngram": (1, 3),
        "char_ngram": (3, 5),
        "C": 2.0
    },

    {
        "name": "Experiment 5",
        "word_ngram": (1, 2),
        "char_ngram": (2, 5),
        "C": 2.0
    },

    {
        "name": "Experiment 6",
        "word_ngram": (1, 3),
        "char_ngram": (3, 6),
        "C": 4.0
    }

]


# ==========================================
# 3. RUN EXPERIMENTS
# ==========================================

results = []


for experiment in experiments:

    print("\n")
    print("=" * 60)

    print(
        experiment["name"]
    )

    print("=" * 60)

    print(
        "Word n-gram:",
        experiment["word_ngram"]
    )

    print(
        "Character n-gram:",
        experiment["char_ngram"]
    )

    print(
        "C:",
        experiment["C"]
    )

    # --------------------------------------
    # WORD TF-IDF
    # --------------------------------------

    word_tfidf = TfidfVectorizer(

        ngram_range=experiment["word_ngram"],

        min_df=2,

        max_df=0.95,

        sublinear_tf=True

    )

    # --------------------------------------
    # CHARACTER TF-IDF
    # --------------------------------------

    char_tfidf = TfidfVectorizer(

        analyzer="char",

        ngram_range=experiment["char_ngram"],

        min_df=2,

        max_features=30000,

        sublinear_tf=True

    )

    # --------------------------------------
    # COMBINE FEATURES
    # --------------------------------------

    features = FeatureUnion([

        ("word", word_tfidf),

        ("char", char_tfidf)

    ])

    print("\nCreating features...")

    X_train_features = features.fit_transform(
        X_train
    )

    X_validation_features = features.transform(
        X_validation
    )

    print(
        "Feature shape:",
        X_train_features.shape
    )

    # --------------------------------------
    # MODEL
    # --------------------------------------

    model = LogisticRegression(

        max_iter=2000,

        class_weight="balanced",

        C=experiment["C"],

        random_state=42

    )

    print("Training model...")

    model.fit(
        X_train_features,
        y_train
    )

    # --------------------------------------
    # PREDICTIONS
    # --------------------------------------

    print("Making predictions...")

    predictions = model.predict(
        X_validation_features
    )

    # --------------------------------------
    # METRICS
    # --------------------------------------

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    report = classification_report(
        y_validation,
        predictions,
        output_dict=True
    )

    macro_f1 = report[
        "macro avg"
    ][
        "f1-score"
    ]

    weighted_f1 = report[
        "weighted avg"
    ][
        "f1-score"
    ]

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    results.append({

        "Experiment":
        experiment["name"],

        "Word Ngram":
        str(experiment["word_ngram"]),

        "Char Ngram":
        str(experiment["char_ngram"]),

        "C":
        experiment["C"],

        "Accuracy":
        accuracy,

        "Macro F1":
        macro_f1,

        "Weighted F1":
        weighted_f1

    })


# ==========================================
# 4. RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(

    by="Macro F1",

    ascending=False

)


print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 5. SAVE RESULTS
# ==========================================

results_df.to_csv(

    "reports/tuning_results.csv",

    index=False

)


print("\nResults saved to:")

print(
    "reports/tuning_results.csv"
)


print("\n")
print("=" * 70)
print("TUNING COMPLETE")
print("=" * 70)
