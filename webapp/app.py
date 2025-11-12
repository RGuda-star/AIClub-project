import streamlit as st

import pandas as pd

import torch

import sys, os 


from pathlib import Path

import joblib

import string

from nltk.sentiment import SentimentIntensityAnalyzer

import nltk

import json

import spacy

from collections import Counter

from utils.constants import CATEGORY_MAPPING # You shouldn’t have to change this unless you placed constants elsewhere

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess_text
@st.cache_resource

def get_nlp_models():

    # Download VADER lexicon if not already present and only if you included sentiment analysis as a feature

    try:

        nltk.data.find("vader_lexicon")

    except LookupError:

        nltk.download("vader_lexicon", quiet=True)

    

    # Load spaCy model (disable unused components for speed). This will be used to tokenize our reviews

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    analyzer = SentimentIntensityAnalyzer()
    return nlp, analyzer
@st.cache_resource

def get_xgb_model():

# 1. Create a Path object to your saved model (i.e. model_name.pkl)
    
    model_path = Path(__file__).resolve().parent.parent / "model" / "review_classifier.pkl"
# 2. Load the model using joblib
    best_model = joblib.load(model_path)
    return {"best_model": best_model}

POS_WHITELIST = {"VERB", "NOUN", "ADV"}
def pos_counts(text,nlp):

    doc = nlp(text)  # tokenizes the text

    return Counter(

        token.pos_ for token in doc if token.pos_ in POS_WHITELIST

    )  
def add_pos_features(df,nlp):

    pos_data = df["cleaned_text"].apply(lambda text:pos_counts(text,nlp))

    pos_df = pd.DataFrame(list(pos_data)).fillna(0)  # fill null counts with 0

    pos_df.index = df.index  # align columns with original df (dataframe)

    return pd.concat([df, pos_df], axis=1)

def extract_features(text, rating=5.0, include_pos=False):

    cleaned_text = preprocess_text(text)

    nlp, analyzer = get_nlp_models()

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

    df["cleaned_text"] = cleaned_text  # Text should already be cleaned, but let’s just make sure

    if include_pos:

        df = add_pos_features(df, nlp)
    return df
def prepare_features_for_prediction(text, category="unknown", rating=5.0):

    # TODO: Decide if you used POS features in your final model

    include_pos = True  

    # TODO: call extract_features in order to create a df

    df = extract_features(text, rating=rating, include_pos=include_pos)

    # Load the feature names your model expects
    feature_path = Path(__file__).resolve().parent.parent / "model" / "feature_names.json"
    with open(feature_path, "r") as f:
        feature_data = json.load(f)

    expected_features = feature_data
    # TODO: Update path to your saved feature_names.json from Week 4

    

    # TODO: Initialize all category value columns to 0
    category_columns = [f for f in expected_features if f.startswith("category_")]

    for col in category_columns:
        df[col] = 0
    # The last 10 features in feature_data are likely your category columns

    # TODO: Set the appropriate category column to 1 based on the input category

    category_col_name = f"category_{category.lower()}"
    if category_col_name in df.columns:
        df[category_col_name] = 1

    # TODO: Ensure all expected features are present in the dataframe. Loop through each feature in feature_data and check if it is in df.columns

    # If a feature is missing, add it with a value of 0.0    
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0.0
    # Return features in the exact order the model expects
    df = df[expected_features]
    return df 
def xgb_predict(text, model, category="unknown", rating=5.0):

    # TODO: Prepare features using prepare_features_for_prediction
    features = prepare_features_for_prediction(text, category=category, rating=rating)
    

   # Make prediction

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = probabilities[prediction]



    # Convert prediction to label (0=AI/CG, 1=Human/OR)

    label = "Human" if prediction == 1 else "AI"

    

    return label, confidence, probabilities.tolist()

def main():
    st.set_page_config(

        page_title="Amazon Review Analyzer",

        page_icon="🤖"

    )
    # Configure the page
    header = st.container() 
    description = st.container()
    

    

    # TODO: Title and description
    with header:
        st.title("Welcome to the Amazon Review Analyzer!")
   


    
    with description:
        st.text("This is the place where you can figure out if any review on Amazon is AI generated or not!")

    # Load model with a loading spinner

    with st.spinner("Loading XGBoost model..."):

        model_dict = get_xgb_model()

    

    if model_dict is None:

        st.error("Failed to load model. Please check if model files exist.")

        return 
    st.success("XGBoost model loaded successfully!")

    

    # TODO: Create the input section (see 3.2 below)
    # TODO: Create two columns for layout
    col1, col2 = st.columns(2)


    with col1:

        st.write("**Enter Review Text:**")

        
        review = st.text_area("Enter your review:", placeholder="Place your review here...")

        

        # Optional inputs for better predictions

        col1_input, col2_input = st.columns(2)

        

        with col1_input:

            # TODO: Create a select box for product category

            # Hint: Use the keys from CATEGORY_MAPPING
            category = st.selectbox("Please select the category", options =list(CATEGORY_MAPPING.keys()))
                   

        with col2_input:

            # TODO: Create a number input for rating (1-5)
            rating = st.number_input("Enter the rating from 1 to 5", min_value=1, max_value=5, value=5, step=1)
        

        # Analyze button

            analyze_button = st.button("Analyze Review", type="primary")


    

    # TODO: Create the results section (see 3.3 below)
        with col2:

            st.write("**Analysis Results:**")

        

        # Only run analysis if button is clicked and there's input

            if analyze_button and review.strip():

                with st.spinner("Analyzing with XGBoost model..."):

                    try:

                    # Map user-friendly category to dataset category
                        dataset_category = CATEGORY_MAPPING[category]
                        features = prepare_features_for_prediction(review, category=category, rating=rating)
                        st.write("### Features used for prediction:")
                        st.dataframe(features)
                        prediction, confidence, probab = xgb_predict(
                        review, model_dict["best_model"], category=category, rating=rating
                        )
                        st.write("### Prediction Probabilities:", probab)
                        if prediction == "Human":
                            st.success("Human Review Predicted!")
                        else:
                            st.error("AI Review Predicted!")


                        st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                        with st.expander("Feature Analysis"):
                            include_pos=True 
                            features_exp = extract_features(review, rating, include_pos=include_pos)
                            st.write("**Extracted Features**")
                            features_used = {"Category": category,
                            "Rating": rating,
                            "Review Length": len(review)}
                            st.write("Features Used in Prediction")
                            st.table(pd.DataFrame(features_used.items(), columns=["Feature", "Value"]))
                            pass
                    except Exception as e:

                        st.error(f"Error during analysis: {str(e)}")

                        st.exception(e)

        

            elif analyze_button and not review.strip():

                st.warning("Please enter a review to analyze!")




if __name__ == "__main__":
    main()





