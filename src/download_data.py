from pathlib import Path
from urllib.request import urlretrieve


# Project directory
RAW_DATA_DIR = Path("data/raw")

# Create the directory if it doesn't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Official NaijaSenti GitHub repository
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "hausanlp/NaijaSenti/main/data/annotated_tweets/pcm/"
)


FILES = {
    "train.tsv": "naijasenti_pcm_train.tsv",
    "dev.tsv": "naijasenti_pcm_dev.tsv",
    "test.tsv": "naijasenti_pcm_test.tsv",
}

print("Downloading official NaijaSenti Nigerian Pidgin dataset...")
print("Language: Nigerian Pidgin (pcm)")
print()


for source_file, output_file in FILES.items():

    url = BASE_URL + source_file
    output_path = RAW_DATA_DIR / output_file

    print(f"Downloading {source_file}...")

    urlretrieve(url, output_path)

    print(f"Saved: {output_path}")
    print()


print("========================================")
print("DOWNLOAD COMPLETE")
print("========================================")
print()
print("Files downloaded to:")
print(RAW_DATA_DIR)
