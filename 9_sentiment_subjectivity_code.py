# =========================================
# Sentiment & Subjectivity → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge response columns -> one text per user
#    - Compute sentiment features:
#        * VADER: compound, pos, neu, neg
#        * TextBlob: polarity (-1..1), subjectivity (0..1)
#    - Map Rating -> 1..5
# 2) Stack (user, file) rows → 80/20 split
# 3) Train RandomForestRegressor → print MSE on test
# 4) Compute per-user averages: true vs predicted across 5 runs

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sentiment libs
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# =========================================
# ============== CONFIG ===================
# =========================================
files: List[str] = [
 "Q3_Submission1_cleaned.csv",
    "Q3_Submission2_cleaned.csv",
    "Q3_Submission3_cleaned.csv",
    "Q3_Submission4_cleaned.csv",
    "Q3_Submission5_cleaned.csv",
    ]

rating_col = "Rating"  # change if needed

# Map textual labels → 1..5 (edit if your labels differ)
rating_map: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

# Detect response columns (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_sentiment_rows.csv"
per_user_out_csv = "per_user_helpfulness_sentiment_averages.csv"

# =========================================
# ============ HELPERS / LOADING ==========
# =========================================
def merge_responses(df: pd.DataFrame) -> pd.Series:
    """
    Merge all columns whose name contains 'response' (case-insensitive) into one text per row.
    """
    resp_cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not resp_cols:
        raise ValueError(
            "No response columns found. "
            "Adjust RESPONSE_COL_PATTERN or rename columns to include 'response'. "
            f"Columns present: {df.columns.tolist()}"
        )
    merged = df[resp_cols].fillna("").agg(" ".join, axis=1).str.strip()
    return merged, resp_cols

# Initialize analyzers once
_vader = SentimentIntensityAnalyzer()

def sentiment_features(text: str) -> dict:
    """
    Compute VADER + TextBlob sentiment features for a single string.
    Returns a dict with:
      - vader_compound, vader_pos, vader_neu, vader_neg
      - tb_polarity, tb_subjectivity
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "vader_compound": 0.0, "vader_pos": 0.0, "vader_neu": 0.0, "vader_neg": 0.0,
            "tb_polarity": 0.0, "tb_subjectivity": 0.0
        }
    # VADER
    vs = _vader.polarity_scores(text)
    # TextBlob
    tb = TextBlob(text).sentiment
    return {
        "vader_compound": float(vs.get("compound", 0.0)),
        "vader_pos": float(vs.get("pos", 0.0)),
        "vader_neu": float(vs.get("neu", 0.0)),
        "vader_neg": float(vs.get("neg", 0.0)),
        "tb_polarity": float(tb.polarity),
        "tb_subjectivity": float(tb.subjectivity),
    }

def prepare_sentiment_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute sentiment features, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (sentiment features...)
    """
    df = pd.read_csv(path)
    merged, resp_cols = merge_responses(df)

    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}")

    # Map ratings to numeric 1..5
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0, 5.0)

    feats = [sentiment_features(txt) for txt in merged]
    feats_df = pd.DataFrame(feats)

    out = pd.DataFrame({
        "user_id": np.arange(len(df)),
        "file_id": file_id,
        "rating": y
    })
    out = pd.concat([out, feats_df], axis=1)
    out = out.loc[out["rating"].notna()].reset_index(drop=True)
    return out

# =========================================
# =============== PIPELINE ================
# =========================================
# 1) Load & stack five files
all_rows = []
for j, fpath in enumerate(files):
    p = Path(fpath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    all_rows.append(prepare_sentiment_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
feature_cols = ["vader_compound","vader_pos","vader_neu","vader_neg","tb_polarity","tb_subjectivity"]
X = data[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = data["rating"].astype(float)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

# 3) Random Forest
rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    random_state=random_state,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 4) Evaluate on held-out test set
y_pred_test = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"\nMSE (Random Forest on sentiment features): {mse:.4f}")

# 5) Predict for ALL rows (per-user averages across 5 runs)
data["pred_rating"] = np.clip(rf.predict(X), 1.0, 5.0)

per_user = (
    data
    .groupby("user_id", as_index=False)
    .agg(
        true_avg_rating=("rating", "mean"),
        pred_avg_rating=("pred_rating", "mean"),
        n_rows=("rating", "size")
    )
)
per_user["true_avg_rating"] = per_user["true_avg_rating"].round(2)
per_user["pred_avg_rating"] = per_user["pred_avg_rating"].round(2)

print("\nPer-user helpfulness averages (head):")
print(per_user.head())

# 6) Save
data_out = data.copy()
for col in feature_cols:
    data_out[col] = data_out[col].round(4)
data_out["pred_rating"] = data_out["pred_rating"].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)
print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
