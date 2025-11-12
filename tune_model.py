from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from pathlib import Path
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk



import json


model_dir = Path("./model")

baseline_model_path = model_dir / "review_classifier.pkl"

model = joblib.load(baseline_model_path)

new_path = Path("processed-dataset.csv")
df = pd.read_csv(new_path, encoding='latin-1')
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

df['sentiment_score'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

numeric_features = df[["rating", "char_length", "word_count", "punctuation_ct", "is_extreme_star", "sentiment_score"]]

# Add POS features if present in the dataset
pos_features = df[[col for col in df.columns if col in ["NOUN", "VERB", "ADV"]]] if any(col in df.columns for col in ["NOUN", "VERB", "ADV"]) else pd.DataFrame()

# One-hot encode category columns
category_features = pd.get_dummies(df["category"], prefix="category")

# Combine all features
X = pd.concat([numeric_features, pos_features, category_features], axis=1)
label_map = {"OR": 1, "CG": 0}
y = df["label"].map(label_map)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({

    'feature': X.columns,

    'importance': importances

}).sort_values('importance', ascending=False)



print(feature_importance_df.head(10))

plot_importance(model)

pyplot.show()

parameters = {

    "n_estimators": [50, 100, 200, 500],

    "learning_rate": [0.1, 0.3, 0.6, 1.0],

    "max_depth": [3, 6, 10],

    "reg_alpha": [0.0, 0.1, 0.5, 1.0],

    "reg_lambda": [0.1, 0.5, 1.0, 1.5],

}

xgb_model = XGBClassifier(

    use_label_encoder=False,

    eval_metric="logloss",

    random_state=42,

)



grid_search = GridSearchCV(

    estimator=xgb_model,

    param_grid=parameters,

    scoring="roc_auc",

    cv=5,  # 5-fold cross-validation

    n_jobs=-1,  # Use all CPU cores

    verbose=1,

    return_train_score=True,

)



grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

best_model = grid_search.best_estimator_

best_score = grid_search.best_score_



print("\n" + "=" * 50)

print("GRID SEARCH RESULTS")

print("=" * 50)

print(f"Best parameters: {best_params}")

print(f"Best cross-validation AUC score: {best_score:.4f}")

print("=" * 50)



best_pred = best_model.predict(X_test)

best_prob = best_model.predict_proba(X_test)[:, 1]



print("\nBest Model Performance on Test Set:")

print("Classification Report:\n", classification_report(y_test, best_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, best_pred))

print(f"Test AUC Score: {roc_auc_score(y_test, best_prob):.4f}")
baseline_pred = model.predict(X_test)

baseline_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)

print("BASELINE vs BEST MODEL COMPARISON")

print("=" * 50)

print(f"Baseline AUC: {roc_auc_score(y_test, baseline_prob):.4f}")

print(f"Best Model AUC: {roc_auc_score(y_test, best_prob):.4f}")

thresh = 0.020  # Only keep features with importance >= 0.020

selection = SelectFromModel(model, threshold=thresh, prefit=True)

select_X_train = selection.transform(X_train)

xgb2model = XGBClassifier(**best_params,

    use_label_encoder=False,

    eval_metric="logloss",

    random_state=42,

)
xgb2model.fit(select_X_train, y_train)
select_X_test = selection.transform(X_test) 
newy_pred = xgb2model.predict(select_X_test)
newy_prob = xgb2model.predict_proba(select_X_test)[:,1]
print("Classification Report:\n", classification_report(y_test, newy_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, newy_pred))

print(f"Test AUC Score: {roc_auc_score(y_test, newy_prob):.4f}")

# Save the fine-tuned model

joblib.dump(xgb2model, model_dir / "selection_model.pkl")


feature_names = X.columns.tolist()



# Save model metadata

model_metadata = {

    "best_params": best_params,

    "best_cv_score": float(best_score), # may need to remove if grid search is commented out

    "test_auc_best": float(roc_auc_score(y_test, best_prob)), # may need to remove if grid search is commented out

    "num_original_features": len(feature_names),

}



with open(model_dir / "selection_metadata.json", "w") as f:

    json.dump(model_metadata, f, indent=2)

