from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

INPUT_FILE = Path("data/interim/clean_master.csv")
OUTPUT_DIR = Path("data/processed")


# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load clean master dataset
# --------------------------------------------------

print("Loading clean master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total clean records: {len(df)}")


# --------------------------------------------------
# 3. Remove the original split column
# --------------------------------------------------

if "original_split" in df.columns:
    df = df.drop(columns=["original_split"])


# --------------------------------------------------
# 4. First split:
#    70% training
#    30% temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# --------------------------------------------------
# 5. Second split:
#    15% validation
#    15% test
# --------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# --------------------------------------------------
# 6. Save the three datasets
# --------------------------------------------------

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n==============================")
print("SPLIT COMPLETE")
print("==============================")

print(f"\nTraining records:   {len(train_df)}")
print(f"Validation records: {len(validation_df)}")
print(f"Test records:       {len(test_df)}")


print("\nTraining distribution:")
print(train_df["label"].value_counts())

print("\nValidation distribution:")
print(validation_df["label"].value_counts())

print("\nTest distribution:")
print(test_df["label"].value_counts())


print("\nFiles saved to:")
print(OUTPUT_DIR)
