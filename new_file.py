import pandas as pd

from pathlib import Path # To manipulate file paths easier as objects rather than just strings

import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from src.preprocess import preprocess_text 
from src.feature_extraction import extract_features


new_path = Path("fake-reviews.csv")
df = pd.read_csv(new_path)
df["cleaned_text"] = df["text_"].apply(preprocess_text)
include_position = True
df = (extract_features(df, include_pos=include_position))
newthis = Path(__file__).resolve().parent / "processed-dataset.csv"
df.to_csv(newthis, index=False)
print("Processed dataset saved to:", newthis)
print("Script finished running successfully!")



