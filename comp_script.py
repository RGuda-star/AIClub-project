import pandas as pd
import joblib
import sys
from pathlib import Path
import json
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import spacy
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import os
from webapp.utils.constants import (
    CATEGORY_MAPPING,
)  # TODO: change this to match the path to your constants

# TODO: change this path if required in order to load preprocess functions
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocess import preprocess_text

# TODO: update this to your model path
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "model", "review_classifier.pkl"
)

# TODO: adjust for your case
POS_WHITELIST = {"VERB", "NOUN", "ADV"}


# TODO: the next several functions are duplicate code with your Streamlit app main file
def load_nltk_and_spacy():
    """Load required NLP models"""
    try:
        nltk.data.find("vader_lexicon")
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download("vader_lexicon", quiet=True)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    analyzer = SentimentIntensityAnalyzer()
    return nlp, analyzer


def pos_counts(text, nlp):
    """Count part-of-speech tags in text"""
    doc = nlp(text)
    return Counter(token.pos_ for token in doc if token.pos_ in POS_WHITELIST)


def add_pos_features(df, nlp):
    """Add POS tag features to dataframe"""
    pos_data = df["cleaned_text"].apply(lambda text: pos_counts(text, nlp))
    pos_df = pd.DataFrame(list(pos_data)).fillna(0)
    pos_df.index = df.index

    for tag in POS_WHITELIST:
        if tag not in pos_df.columns:
            pos_df[tag] = 0

    return pd.concat([df, pos_df], axis=1)


def extract_features(text, rating, nlp, analyzer):
    """Extract features from a single review"""
    cleaned_text = preprocess_text(text)

    df = pd.DataFrame(
        [
            {
                "rating": rating,
                "char_length": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "punctuation_ct": sum(
                    1 for c in cleaned_text if c in string.punctuation
                ),
                "is_extreme_star": rating in [1.0, 5.0],
                "sentiment_score": analyzer.polarity_scores(cleaned_text)["compound"],
            }
        ]
    )

    df["cleaned_text"] = cleaned_text
    df = add_pos_features(df, nlp)

    return df


def prepare_features(df, feature_names, category_col):
    """Prepare features for prediction"""
    # Add category one-hot encoding
    for col in feature_names:
        if col.startswith(tuple(CATEGORY_MAPPING.values())) and col not in df.columns:
            df[col] = 0

    # Set the appropriate category column to 1
    if category_col in df.columns:
        df[category_col] = 1

    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0

    return df[feature_names]


def load_test_data():
    """Load the hidden test set"""
    # In actual competition, this would load from a file
    # Until the competition, you will just get one sample
    test_data = pd.DataFrame(
        [
            {
                "category": "Kindle_Store_5",
                "rating": 3.0,
                "label": "CG",  # AI-generated
                "text_": """Eva is on her to find a way to escape her abusive mother. When she meets a handsome stranger she doesn't know is the man she wants but the man she wants is strong and handsome. She is in love with him and will do anything to get her happily ever after.

I loved this book! I can't wait to see how the next book comes out!I received a free copy of this book from the author for an honest review.

This book was so good. I loved it. I loved the character development. I loved the relationship between the two main characters. The romance was real, and the story flowed at a great pace. It was a fun read. I would definitely recommend this book to anyone.I received this book for an honest review.  This is my first book by this author and I was very happy to see it.  I love the characters and the plot.  I will definitely be looking for more by this author.  This is a great series and I look forward to reading more from this author""",
            }
        ]
    )

    # Convert labels: CG (AI) = 0, OR (Human) = 1
    test_data["label_numeric"] = test_data["label"].map({"CG": 0, "OR": 1})

    return test_data


def main():
    print("=" * 60)
    print("Amazon Review Classification Competition")
    print("Model Evaluation Script")
    print("=" * 60)
    print()

    # Load student's model
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please save your trained model as 'model_name.pkl'")
        return

    print("Loading your model...")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    # TODO: load feature names. Change path if required
    feature_names_path = "model/feature_names.json"
    if not Path(feature_names_path).exists():
        print(f"❌ Error: Feature names file not found at {feature_names_path}")
        return

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    # Load NLP models
    print("\nLoading NLP models...")
    nlp, analyzer = load_nltk_and_spacy()
    print("✅ NLP models loaded!")

    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data()
    print(f"✅ Loaded {len(test_df)} test samples")

    # Extract features for all test samples
    print("\nExtracting features...")
    feature_dfs = []
    for idx, row in test_df.iterrows():
        features = extract_features(row["text_"], row["rating"], nlp, analyzer)
        features = prepare_features(features, feature_names, row["category"])
        feature_dfs.append(features)

    X_test = pd.concat(feature_dfs, ignore_index=True)
    y_test = test_df["label_numeric"].values

    # Make predictions
    print("Making predictions...")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"❌ Error making predictions: {str(e)}")
        return

    # Calculate metrics
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    print(f"\n🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall:    {recall:.4f}")
    print(f"🎯 F1 Score:  {f1:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0,1], target_names=["AI (CG)", "Human (OR)"]))

    print("\n" + "=" * 60)
    print(f"🏆 Your model achieved {accuracy*100:.2f}% accuracy!")
    print("=" * 60)


if __name__ == "__main__":
    main()
