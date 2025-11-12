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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

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
    
    xgb_model_path = os.path.join("model", "review_classifier.pkl")
# 2. Load the model using joblib
    best_model = joblib.load(xgb_model_path)
    return {"best_model": best_model}




@st.cache_resource

@st.cache_resource

def load_lora_model():

    """Load LoRA fine-tuned model"""

    model_path = Path(__file__).resolve().parent / "model" / "bert_lora_model"
    model_path_str = str(model_path)
 # Adjust path as needed

    

    # Load base model first

    config = PeftConfig.from_pretrained(model_path_str)

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

    model = PeftModel.from_pretrained(
        model,                              # pass base model here
        pretrained_model_name_or_path=model_path_str,
        repo_type="local"                   # tell PEFT it’s local
    )


    

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

def extract_features_xlp(text, rating=5.0, include_pos=False):

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
def extract_features_bert(text, include_pos=False):

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
def prepare_features_for_prediction_bert(text):

    # TODO: Decide if you used POS features in your final model

    include_pos = True  

    # TODO: call extract_features in order to create a df

    df = extract_features_bert(text, include_pos=include_pos)

    # Load the feature names your model expects
   
    # TODO: Update path to your saved feature_names.json from Week 4

    

    # TODO: Initialize all category value columns to 0
   
    # The last 10 features in feature_data are likely your category columns

    # TODO: Set the appropriate category column to 1 based on the input category

    

    # TODO: Ensure all expected features are present in the dataframe. Loop through each feature in feature_data and check if it is in df.columns

    # If a feature is missing, add it with a value of 0.0    
    
    # Return features in the exact order the model expects

def prepare_features_for_prediction_xlp(text, category="unknown", rating=5.0):

    # TODO: Decide if you used POS features in your final model

    include_pos = True  

    # TODO: call extract_features in order to create a df

    df = extract_features_xlp(text, rating=rating, include_pos=include_pos)

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

def combine_predict(review, category="unknown", rating=5.0):
    model_dict = get_xgb_model()
    bert_model, tokenizer = load_lora_model()
    label_xgb, confidence_xgb, _ = xgb_predict(review, model_dict["best_model"], category=category, rating=rating)
    label_bert, confidence_bert, _ = predict_with_bert(review, bert_model, tokenizer)
    prob_xgb = confidence_xgb if label_xgb == "AI" else (1 - confidence_xgb)
    prob_bert = confidence_bert if label_bert == "AI" else (1 - confidence_bert)
    final_score = 0.3 * prob_xgb + 0.7 * prob_bert
    final_label = "AI" if final_score >= 0.5 else "Human"
    confidence = final_score if final_label == "AI" else 1 - final_score
    return {
        "final_label": final_label, "confidence": confidence, "label_xgb":label_xgb,
        "confidence_xgb": confidence_xgb, "confidence_bert": confidence_bert,
        "label_bert": label_bert
    }

    

def main_bert():
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
                            features = extract_features_bert(review, include_pos=include_pos)
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






def xgb_predict(text, model, category="unknown", rating=5.0):

    # TODO: Prepare features using prepare_features_for_prediction
    features = prepare_features_for_prediction_xlp(text, category=category, rating=rating)
    

   # Make prediction

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = probabilities[prediction]



    # Convert prediction to label (0=AI/CG, 1=Human/OR)

    label = "Human" if prediction == 1 else "AI"

    

    return label, confidence, probabilities.tolist()
def main_xlp():
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
                        features = prepare_features_for_prediction_xlp(review, category=category, rating=rating)
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
                            features_exp = extract_features_xlp(review, rating, include_pos=include_pos)
                            st.write("**Extracted Features**")
                            features_used = {"Category": category,
                            "Rating": rating,
                            "Review Length": len(review)}
                            st.write("Features Used in Prediction")
                            st.table(pd.DataFrame(features_used.items(), columns=["Feature", "Value"]))
                            
                    except Exception as e:

                        st.error(f"Error during analysis: {str(e)}")

                        st.exception(e)

        

            elif analyze_button and not review.strip():

                st.warning("Please enter a review to analyze!")


def combine_main():
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
    st.write("**Try these examples:**")

    example_human = "What a product!"

    example_ai = "This product is good. It works well. I recommend it to others."



    if st.button("Load Human Example"):

        st.session_state.review_text = example_human

    if st.button("Load AI Example"):

        st.session_state.review_text = example_ai
    with description:
        st.text("This is the place where you can figure out if any review on Amazon is AI generated or not!")
    
    col1, col2 = st.columns(2)


    with col1:

        st.write("**Enter Review Text:**")

        
        # Use session_state key to make it reactive
        review = st.text_area(
        "Enter your review:",
        key="review_text",
        placeholder="Place your review here..."
        )


        

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
    with col2:
            
            analyze_button = st.button("Analyze Review", type="primary")
            if analyze_button and review.strip():
                
                    with st.spinner("Running hybrid inference..."):
                        try:
                            result = combine_predict(review, category, rating)
                            
                            st.subheader(f"Final Verdict: {result['final_label']}")
                            st.metric(label="Confidence", value=f"{result['confidence']*100:.2f}%")
                            if result["confidence"] < 0.7:
                                st.warning("Model is uncertain; this may be human or AI.")
                            with st.expander("Feature Analysis"):
                                include_pos=True 
                                features_exp = extract_features_xlp(review, rating, include_pos=include_pos)
                                st.write("**Extracted Features**")
                                features_used = {"Category": category,
                                "Rating": rating,
                                "Review Length": len(review)}
                                st.write("Features Used in Prediction")
                                st.table(pd.DataFrame(features_used.items(), columns=["Feature", "Value"]))
                        except Exception as e:

                            st.error(f"Error during analysis: {str(e)}")

                            st.exception(e)

            else:
                st.error("Please enter a review.")
                

if __name__ == "__main__":
    combine_main()





