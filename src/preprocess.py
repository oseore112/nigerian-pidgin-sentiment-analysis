from pathlib import Path
import html
import re
import unicodedata

import pandas as pd


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TEXT PREPROCESSING FUNCTION
# ============================================================

def clean_pidgin_text(text):
    """
    Clean Nigerian Pidgin social-media text while preserving
    important linguistic information such as emojis, negation,
    slang, and repeated words.
    """

    # Handle missing values safely
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter/X-style mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove zero-width and other invisible characters
    text = re.sub(
        r"[\u200b-\u200f\u202a-\u202e\ufeff]",
        "",
        text
    )

    # Normalize repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 3. LOAD THE THREE DATASETS
# ============================================================

print("Loading processed datasets...")

train = pd.read_csv(INPUT_DIR / "train.csv")
validation = pd.read_csv(INPUT_DIR / "validation.csv")
test = pd.read_csv(INPUT_DIR / "test.csv")

print(f"Train records: {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Test records: {len(test)}")


# ============================================================
# 4. APPLY PREPROCESSING
# ============================================================

print("\nCleaning text...")

train["tweet_clean"] = train["tweet"].apply(clean_pidgin_text)
validation["tweet_clean"] = validation["tweet"].apply(clean_pidgin_text)
test["tweet_clean"] = test["tweet"].apply(clean_pidgin_text)


# ============================================================
# 5. CHECK FOR EMPTY CLEANED TEXT
# ============================================================

print("\nEmpty cleaned tweets:")

print(
    "Train:",
    (train["tweet_clean"].str.len() == 0).sum()
)

print(
    "Validation:",
    (validation["tweet_clean"].str.len() == 0).sum()
)

print(
    "Test:",
    (test["tweet_clean"].str.len() == 0).sum()
)


# ============================================================
# 6. SAVE CLEANED DATASETS
# ============================================================

train.to_csv(
    OUTPUT_DIR / "train_clean.csv",
    index=False
)

validation.to_csv(
    OUTPUT_DIR / "validation_clean.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "test_clean.csv",
    index=False
)


# ============================================================
# 7. SHOW BEFORE/AFTER EXAMPLES
# ============================================================

print("\n==============================")
print("BEFORE / AFTER EXAMPLES")
print("==============================")

examples = train.sample(
    10,
    random_state=42
)

for _, row in examples.iterrows():

    print("\nORIGINAL:")
    print(row["tweet"])

    print("\nCLEANED:")
    print(row["tweet_clean"])

    print("\nLABEL:")
    print(row["label"])

    print("-" * 60)


print("\n==============================")
print("PREPROCESSING COMPLETE")
print("==============================")
