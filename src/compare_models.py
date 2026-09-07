import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

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


# ==========================================
# 2. PREPARE DATA
# ==========================================

X_train = train["tweet_clean"]
y_train = train["label"]

X_validation = validation["tweet_clean"]
y_validation = validation["label"]


# ==========================================
# 3. CREATE WORD TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_validation_tfidf = vectorizer.transform(
    X_validation
)

print(
    "Training feature shape:",
    X_train_tfidf.shape
)

print(
    "Validation feature shape:",
    X_validation_tfidf.shape
)


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=1.0
    )
}


# ==========================================
# 5. TRAIN AND EVALUATE
# ==========================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

    print("Training...")

    model.fit(
        X_train_tfidf,
        y_train
    )

    print("Making predictions...")

    predictions = model.predict(
        X_validation_tfidf
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    report = classification_report(
        y_validation,
        predictions,
        output_dict=True
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        classification_report(
            y_validation,
            predictions,
            digits=4
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    })


# ==========================================
# 6. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)


print("\nResults saved to:")
print("reports/model_comparison.csv")

print("\n")
print("=" * 60)
print("MODEL COMPARISON COMPLETE")
print("=" * 60)
