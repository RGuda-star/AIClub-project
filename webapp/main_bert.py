import streamlit as st

import pandas as pd

import torch

import sys, os 

from peft import PeftModel, PeftConfig

import torch.nn as nn

from pathlib import Path

import joblib

import string

from nltk.sentiment import SentimentIntensityAnalyzer

import nltk

from transformers import BertTokenizerFast, BertForSequenceClassification

import torch

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




@st.cache_resource

@st.cache_resource

def load_lora_model():

    """Load LoRA fine-tuned model"""

    model_path = Path(__file__).resolve().parent.parent / "model" / "bert_lora_model" # Adjust path as needed

    

    # Load base model first

    config = PeftConfig.from_pretrained(model_path)

    model = BertForSequenceClassification.from_pretrained(

        config.base_model_name_or_path,

        num_labels=2,

        return_dict=True,

        torch_dtype=torch.float32,

    )

    

    tokenizer = BertTokenizerFast.from_pretrained(config.base_model_name_or_path)

    

    # Apply the same modifications as during training

    for param in model.parameters():

        param.requires_grad = False

        if param.ndim == 1:

            param.data = param.data.to(torch.float32)

    

    model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    

    # Cast classifier head to float32 for stability

    class CastOutputToFloat(nn.Sequential):

        def forward(self, x):

            return super().forward(x).to(torch.float32)

    

    model.classifier = CastOutputToFloat(model.classifier)

    

    # Load LoRA adapters

    model = PeftModel.from_pretrained(model, model_path)

    

    # Move to CPU

    model = model.to("cpu")

    model.eval()
    return model, tokenizer 

    



def predict_with_bert(text, model, tokenizer):

    """Make prediction using BERT model"""

    # Preprocess the text (reuse your existing function)

    processed_text = preprocess_text(text)

    

    # Tokenize

    inputs = tokenizer(

        processed_text,

        padding="max_length",

        truncation=True,

        max_length=512,

        return_tensors="pt",

    )

    

    # Make prediction

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

    

    # Convert to probabilities

    probs = torch.nn.functional.softmax(logits, dim=1)

    prediction = torch.argmax(logits, dim=1).item()

    confidence = probs[0][prediction].item()

    

    # Map to labels (0=AI/CG, 1=Human/OR)

    label = "AI" if prediction == 0 else "Human"

    

    return label, confidence, probs[0].tolist()



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

def extract_features(text, include_pos=False):

    cleaned_text = preprocess_text(text)

    nlp, analyzer = get_nlp_models()

    df = pd.DataFrame(

        [

            {

                

                "char_length": len(cleaned_text),

                "word_count": len(cleaned_text.split()),

                "punctuation_ct": sum(

                    1 for c in cleaned_text if c in string.punctuation

                ),

                
                "sentiment_score": analyzer.polarity_scores(cleaned_text)["compound"],

            }

        ]

    )

    df["cleaned_text"] = cleaned_text  # Text should already be cleaned, but let’s just make sure

    if include_pos:

        df = add_pos_features(df, nlp)
    return df
def prepare_features_for_prediction(text):

    # TODO: Decide if you used POS features in your final model

    include_pos = True  

    # TODO: call extract_features in order to create a df

    df = extract_features(text, include_pos=include_pos)

    # Load the feature names your model expects
   
    # TODO: Update path to your saved feature_names.json from Week 4

    

    # TODO: Initialize all category value columns to 0
   
    # The last 10 features in feature_data are likely your category columns

    # TODO: Set the appropriate category column to 1 based on the input category

    

    # TODO: Ensure all expected features are present in the dataframe. Loop through each feature in feature_data and check if it is in df.columns

    # If a feature is missing, add it with a value of 0.0    
    
    # Return features in the exact order the model expects
    return df


    

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

    with st.spinner("Loading BERT model..."):

        bert_model, tokenizer = load_lora_model()

        st.success("✅ BERT model loaded successfully!")

    

    if bert_model is None:

        st.error("Failed to load model. Please check if model files exist.")

        return 
    st.success("BERT model loaded successfully!")

    

    # TODO: Create the input section (see 3.2 below)
    # TODO: Create two columns for layout
    col1, col2 = st.columns(2)


    with col1:

        st.write("**Enter Review Text:**")

        
        review = st.text_area("Enter your review:", placeholder="Place your review here...")

        

       
        

        # Analyze button

        analyze_button = st.button("Analyze Review", type="primary")


    

    # TODO: Create the results section (see 3.3 below)
        with col2:

            st.write("**Analysis Results:**")

        

        # Only run analysis if button is clicked and there's input

            if analyze_button and review.strip():

                with st.spinner("Analyzing with BERT model..."):

                    try:

                    # Map user-friendly category to dataset category
                        
                        label, confidence, probabilities = predict_with_bert(

                        review, bert_model, tokenizer

                        )

                        if label == "Human":
                            st.success("Human Review Predicted!")
                        else:
                            st.error("AI Review Predicted!")


                        st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                        with st.expander("Feature Analysis"):
                            include_pos=True 
                            features = extract_features(review, include_pos=include_pos)
                            st.write("**Extracted Features**")
                            features_used = {
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





