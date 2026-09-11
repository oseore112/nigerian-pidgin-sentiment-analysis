# 🇳🇬 Why Nigerian Pidgin?

Nigerian Pidgin is an important language variety in Nigeria and is heavily represented in informal communication and social media.

However, NLP systems often struggle with it because the language contains:

- Nigerian-specific slang
- Non-standard spellings
- Local expressions
- Borrowed English words
- Code-switching
- Contextual meanings
- Informal grammar
- Humor and sarcasm

For example:

> "E dey give cancer"

does not necessarily contain obvious English sentiment keywords that a simple keyword-based system would understand correctly.

This makes Nigerian Pidgin an interesting and challenging NLP problem.

---

# 📊 Dataset

The project uses the **NaijaSenti** dataset, specifically the Nigerian Pidgin sentiment data.

The original data was processed and prepared for machine learning.

After cleaning and preparation, the modelling dataset contained:

```text
Total clean records: 8,453
```

The dataset was divided into:

| Dataset    |   Records |
| ---------- | --------: |
| Training   |     5,917 |
| Validation |     1,268 |
| Test       |     1,268 |
| **Total**  | **8,453** |

The sentiment distribution in the training dataset was:

| Sentiment | Training Records |
| --------- | ---------------: |
| Negative  |            3,593 |
| Positive  |            1,977 |
| Neutral   |              347 |

This shows a significant class imbalance, particularly for the Neutral class.

---

# 🧹 Data Preparation

Several preprocessing and data-quality steps were performed before model training.

The preprocessing pipeline included:

- Loading the raw dataset
- Cleaning text
- Removing unnecessary noise
- Preparing sentiment labels
- Checking empty cleaned records
- Preparing training, validation, and test datasets
- Checking dataset split integrity
- Reviewing problematic/conflicting records
- Saving processed datasets for subsequent modelling

The cleaned datasets contained no empty cleaned tweets.

The final modelling split was:

```text
Training:    5,917
Validation:  1,268
Test:        1,268
```

The test dataset was kept separate from the training and validation process and was used for final evaluation.

---

# 🔍 Data Quality Audit

Data quality was treated as an important part of the project rather than immediately jumping into model training.

The audit considered:

- Missing/empty text
- Text cleaning
- Label consistency
- Duplicate/problematic records
- Dataset distribution
- Train/validation/test separation
- Potential data leakage

The final cleaned dataset contained **8,453 records**.

The project also revealed an important issue with the dataset:

### Class imbalance

The Neutral class contained substantially fewer examples than the Negative and Positive classes.

```text
Negative → 3,593
Positive → 1,977
Neutral  →   347
```

This imbalance became one of the major challenges during model development.

---

# 🧠 Model Development

The project followed an incremental machine learning workflow.

Rather than immediately building a complicated model, a baseline model was first created and then improved through controlled experiments.

The primary approach was:

```text
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning Classifier
   ↓
Sentiment Prediction
```

---

# 🧮 Feature Engineering

The project experimented with both **word-level and character-level TF-IDF features**.

### Word-level TF-IDF

Word n-grams were used to capture meaningful word combinations.

Example:

```text
"dey sweet"
"no like"
"very good"
```

The final configuration used:

```text
Word n-grams: (1, 2)
```

This means the model considered:

- Unigrams
- Bigrams

---

### Character-level TF-IDF

Character n-grams were also introduced to help capture:

- Nigerian Pidgin spelling variations
- Informal spelling
- Slang
- Word fragments
- Misspellings
- Morphological patterns

The final configuration used:

```text
Character n-grams: (3, 5)
```

The word-level and character-level features were combined into a single feature representation.

The resulting feature space contained approximately:

```text
46,002 features
```

---

# 🤖 Models Tested

Several machine learning approaches were evaluated.

## 1. Logistic Regression

The first baseline model used:

```text
TF-IDF
+
Logistic Regression
```

Baseline validation accuracy:

```text
65.69%
```

This provided a starting point for further experimentation.

---

## 2. Linear SVM

A Linear Support Vector Machine was also tested.

Validation accuracy:

```text
67.51%
```

Although the SVM achieved slightly higher accuracy than the initial Logistic Regression baseline, its Neutral-class performance remained weak.

---

## 3. Multinomial Naive Bayes

Multinomial Naive Bayes was tested as another traditional NLP baseline.

Validation accuracy:

```text
63.25%
```

Its performance was weaker than both Logistic Regression and Linear SVM.

The model also struggled significantly with the Neutral class.

---

# 🧪 Model Experiments

The project included several controlled experiments involving:

- Word n-gram ranges
- Character n-gram ranges
- Logistic Regression regularization strength (`C`)
- Word + character TF-IDF combinations
- Oversampling
- Class weighting
- Error analysis
- Confidence analysis

---

# 📈 Hyperparameter Tuning

Several configurations were tested.

Examples included:

| Experiment | Word N-gram | Character N-gram |   C |   Accuracy |
| ---------- | ----------- | ---------------- | --: | ---------: |
| 1          | (1,2)       | (3,5)            | 1.0 |     67.67% |
| 2          | (1,2)       | (3,5)            | 2.0 | **68.38%** |
| 3          | (1,2)       | (3,5)            | 4.0 |     67.90% |
| 4          | (1,3)       | (3,5)            | 2.0 | **68.38%** |
| 5          | (1,2)       | (2,5)            | 2.0 |     67.51% |
| 6          | (1,3)       | (3,6)            | 4.0 |     67.74% |

The strongest validation configuration selected during experimentation was based on:

```text
Word n-grams:      (1, 2)
Character n-grams: (3, 5)
C:                 2.0
Classifier:        Logistic Regression
```

---

# ⚖️ Class Imbalance Experiments

Because the Neutral class was significantly underrepresented, additional experiments were performed.

Two approaches were investigated:

### Oversampling

The minority classes were oversampled so that each class contained:

```text
3,593 records
```

This produced:

```text
Negative → 3,593
Positive → 3,593
Neutral  → 3,593
```

The oversampled model achieved:

```text
Accuracy: 68.14%
Macro F1: 54.36%
Weighted F1: 67.70%
```

However, oversampling did not produce a major improvement in Neutral-class performance.

---

### Class Weighting

Different class-weight configurations were tested, including:

```text
Equal weights
Neutral 1.5x
Neutral 2x
Neutral 3x
Balanced
```

The experiments demonstrated that simply increasing the Neutral class weight did not solve the underlying classification problem.

The balanced configuration produced the strongest overall macro F1 among the tested weighting strategies:

```text
Macro F1: 55.15%
```

However, Neutral remained substantially weaker than the other classes.

---

# 🔬 Error Analysis

An error-analysis stage was implemented to understand where the model was making mistakes.

The baseline model produced:

```text
Validation Accuracy: 65.69%
Incorrect predictions: 435
```

Common errors included:

```text
Negative → Positive
Positive → Negative
Negative → Neutral
Neutral → Negative
Neutral → Positive
Positive → Neutral
```

The most common confusion was between:

```text
Negative ↔ Positive
```

This is understandable for Nigerian Pidgin because sentiment can be highly contextual, sarcastic, humorous, or dependent on cultural expressions.

Examples of difficult cases included texts containing:

- Sarcasm
- Insults expressed humorously
- Praise within negative contexts
- Negative words used jokingly
- Positive expressions used sarcastically
- Very short statements
- Ambiguous slang

---

# 🏆 Final Model

The final selected model is a traditional NLP pipeline based on:

```text
Word TF-IDF
+
Character TF-IDF
+
Logistic Regression
```

Final configuration:

```text
Classifier:
Logistic Regression

Word n-grams:
(1, 2)

Character n-grams:
(3, 5)

Regularization parameter:
C = 2.0
```

The combined TF-IDF representation contained approximately:

```text
46,002 features
```

The final trained model and vectorizers were saved for use by the prediction application.

---

# 📊 Final Test Results

The final model was evaluated against the previously unseen test dataset containing:

```text
1,268 records
```

Final test performance:

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **68.22%** |
| Macro F1    | **44.80%** |
| Weighted F1 | **65.46%** |

### Class-level F1 scores

| Sentiment |   F1 Score |
| --------- | ---------: |
| Negative  | **77.73%** |
| Positive  | **54.17%** |
| Neutral   |  **2.50%** |

These results demonstrate that the model performs substantially better on Negative sentiment than on Neutral sentiment.

---

# ⚠️ Limitations

The most important limitation of this project is the performance of the Neutral class.

The training dataset contained only:

```text
347 Neutral examples
```

compared with:

```text
3,593 Negative
1,977 Positive
```

This imbalance made Neutral classification considerably more difficult.

The final test results reflect this:

```text
Negative F1: 77.73%
Positive F1: 54.17%
Neutral F1:   2.50%
```

Therefore, the model should **not** be considered a highly reliable production-grade sentiment classifier for all Nigerian Pidgin text.

Additional limitations include:

### 1. Sarcasm

The model may interpret sarcastic statements literally.

### 2. Context

Some Nigerian Pidgin expressions require cultural or conversational context to determine sentiment.

### 3. Code-switching

Nigerian Pidgin frequently occurs alongside Standard English and other Nigerian languages.

### 4. Slang

New slang and internet expressions may not be adequately represented in the training data.

### 5. Short messages

Very short texts provide limited information for classification.

### 6. Informal spelling

Different users may spell the same Nigerian Pidgin expression differently.

---

# 🖥️ Streamlit Demo

The trained model was integrated into a Streamlit web application.

The application allows users to enter Nigerian Pidgin text and receive a real-time sentiment prediction.

The application provides:

```text
Sentiment Prediction
        +
Confidence Score
        +
Probability Breakdown
```

Example workflow:

```text
User enters Nigerian Pidgin text
             ↓
Text preprocessing
             ↓
TF-IDF transformation
             ↓
Trained Logistic Regression model
             ↓
Prediction
             ↓
Confidence + probability breakdown
```

The application supports the following sentiment classes:

🟢 **Positive**

🔴 **Negative**

⚪ **Neutral**

---

## 🚀 Live Demo

The Streamlit application can be accessed here:

**[(http://10.112.8.154:8501)]**

Example:

```text
http://10.112.8.154:8501
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project:

```bash
cd nigerian-pidgin-sentiment
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

You can then enter Nigerian Pidgin text into the text box and click:

```text
Analyze Sentiment
```

The application returns the predicted sentiment and probability distribution.

---

# 📁 Project Structure

```text
nigerian-pidgin-sentiment/
│
├── app.py
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── final_logistic_regression.pkl
│   ├── final_word_tfidf_vectorizer.pkl
│   └── final_char_tfidf_vectorizer.pkl
│
├── reports/
│   ├── model_comparison.csv
│   ├── tuning_results.csv
│   ├── oversampling_comparison.csv
│   ├── class_weight_results.csv
│   └── final_test_predictions.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── error_analysis.py
│   ├── train_improved.py
│   ├── compare_models.py
│   ├── tune_model.py
│   ├── tune_class_weights.py
│   └── predict.py
│
└── tests/
```

> The exact contents of individual directories may change as the project evolves.

---

# 🔁 Reproducibility

The project was developed using a reproducible machine learning workflow.

The general pipeline is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Quality Checks
   ↓
Train / Validation / Test Split
   ↓
TF-IDF Feature Engineering
   ↓
Baseline Model
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Error Analysis
   ↓
Class Imbalance Experiments
   ↓
Final Model
   ↓
Test Evaluation
   ↓
Streamlit Application
```

The trained model and vectorizers are stored in the `models/` directory.

Experiment results are stored in the `reports/` directory.

---

# 📚 Technologies Used

The project was built primarily with Python and the following technologies:

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- TF-IDF
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Git
- GitHub

---

# 🔮 Future Improvements

Although the current system provides a functional baseline, several improvements could significantly increase performance.

### Better Neutral Classification

The Neutral class requires additional high-quality training examples.

Future work could include:

- More Neutral-labelled Nigerian Pidgin data
- Improved annotation guidelines
- Human review of ambiguous examples
- Better handling of conflicting labels

### Advanced NLP Models

Future versions could investigate transformer-based models such as:

- BERT
- multilingual transformer models
- African-language NLP models
- Nigerian Pidgin-specific language models

### Context-Aware Classification

Future systems could incorporate:

- Conversation context
- Previous messages
- User intent
- Emoji information
- Sarcasm detection

### Data Augmentation

Additional training examples could be generated or collected to improve representation of:

- Slang
- New expressions
- Regional variations
- Code-switching
- Neutral statements

### Production Improvements

Future versions could also include:

- Automated testing
- API deployment
- Model monitoring
- Prediction logging
- Continuous model evaluation
- Model versioning

---

# 🧠 Key Lessons From the Project

One of the major lessons from this project is that **accuracy alone does not tell the whole story**.

A model can achieve approximately 68% accuracy while still performing poorly on an important minority class.

This project therefore demonstrates the importance of examining:

- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- Confusion matrices
- Class distribution
- Error analysis

The Neutral class exposed a real-world machine learning challenge: **having a model that works reasonably well overall does not necessarily mean that it works equally well for every class.**

---

# 👨‍💻 Author

**Samuel O. Akinola**

B.Sc. Geology | Machine Learning & Data Science Enthusiast

Areas of interest include:

- Machine Learning
- Natural Language Processing
- Data Analysis
- Artificial Intelligence
- Geospatial Technology
- Technical Writing

---

# ⭐ Project Status

**Status: Completed — Initial Production Demo**

The project currently includes:

- ✅ Dataset preparation
- ✅ Data cleaning
- ✅ Train/validation/test split
- ✅ Data quality analysis
- ✅ TF-IDF feature engineering
- ✅ Baseline Logistic Regression
- ✅ Linear SVM comparison
- ✅ Multinomial Naive Bayes comparison
- ✅ Hyperparameter experiments
- ✅ Error analysis
- ✅ Class imbalance experiments
- ✅ Final model training
- ✅ Final test evaluation
- ✅ Model serialization
- ✅ Streamlit application
- ✅ GitHub repository
- ✅ Interactive sentiment prediction

The project is considered a completed end-to-end machine learning portfolio project, with further improvements possible in future versions.

````

### One thing I strongly recommend before you paste this

There are **two placeholders** you should replace:

```text
<YOUR-GITHUB-REPOSITORY-URL>
````

and:

```text
[Add your Streamlit deployment URL here]
```

Also, check the exact names of the three files inside your `models/` folder before keeping this section:

```text
models/
├── final_logistic_regression.pkl
├── final_word_tfidf_vectorizer.pkl
└── final_char_tfidf_vectorizer.pkl
```

The README should describe the **actual filenames in your repository**, not filenames we assume.

And one technical point: I deliberately documented the **final test result separately from the best validation result**. That's important. We don't want to accidentally advertise the 68.38% validation score as though it were the final unseen-test performance. Your final test result is **68.22% accuracy**, with **65.46% weighted F1 and 44.80% macro F1**. That's the honest number to put front and center for the completed project.

**Next after this README:** we should do the final **GitHub repository audit + deployment audit**, then deploy the Streamlit app publicly and put the live link in this README.
