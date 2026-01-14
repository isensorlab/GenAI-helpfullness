import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import Counter

import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
rating_col = "Rating"   # change if different

# Map textual labels -> 1..5 (edit if your labels differ)
rating_map: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful": 2,
    "Neutral": 3,
    "Helpful": 4,
    "Very Helpful": 5,
}

# Regex to detect response columns (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Where to save outputs
stacked_out_csv   = "stacked_pos_rows.csv"
per_user_out_csv  = "per_user_helpfulness_pos_averages.csv"

random_state = 42
test_size    = 0.2

# Universal POS tags to track (spaCy token.pos_)
# You can add/remove tags here
TRACK_POS = [
    "NOUN","VERB","ADJ","ADV","PRON","PROPN","AUX","ADP",
    "DET","CCONJ","SCONJ","PART","NUM"
]

# =========================================
# ============ spaCy PIPELINE =============
# =========================================
# Load spaCy English model. Disable NER for speed; POS tagging needs tagger + attribute_ruler + morph.
nlp = spacy.load("en_core_web_sm", disable=["ner"])
# nlp.max_length = 2_000_000  # uncomment if you have very long rows

# =========================================
# ============ HELPER FUNCTIONS ===========
# =========================================
def merge_response_columns(df: pd.DataFrame) -> pd.Series:
    """Merge all columns whose name matches RESPONSE_COL_PATTERN into one text per row."""
    response_cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not response_cols:
        raise ValueError(
            f"No response columns found. Looking for names containing 'response' (case-insensitive). "
            f"Columns present: {df.columns.tolist()}"
        )
    merged = df[response_cols].fillna("").agg(" ".join, axis=1).str.strip()
    return merged, response_cols

def pos_features_for_docs(texts, batch_size=200):
    """
    Compute POS features for an iterable of texts using spaCy in batches.
    Returns a DataFrame with:
      - token_count (non-space, non-punct tokens)
      - pos_count_* for TRACK_POS
      - pos_ratio_* for TRACK_POS
      - pos_entropy (Shannon entropy over TRACK_POS ratios)
    """
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        # Filter tokens: ignore spaces and pure punctuation
        toks = [t for t in doc if not (t.is_space or t.is_punct)]
        token_count = len(toks)

        # Count POS over selected set
        cnt = Counter(t.pos_ for t in toks) if token_count > 0 else Counter()

        # Build counts & ratios for TRACK_POS
        row = {"token_count": float(token_count)}
        for tag in TRACK_POS:
            c = float(cnt.get(tag, 0))
            row[f"pos_count_{tag}"] = c
            row[f"pos_ratio_{tag}"] = (c / token_count) if token_count > 0 else 0.0

        # POS entropy (over ratios of TRACK_POS only)
        ratios = [row[f"pos_ratio_{tag}"] for tag in TRACK_POS]
        # avoid log(0) by only using positive probs
        entropy = -sum(p * math.log(p, 2) for p in ratios if p > 0)
        row["pos_entropy"] = float(entropy)

        rows.append(row)

    return pd.DataFrame(rows)

def prepare_pos_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute POS features and map rating.
    Returns DataFrame with:
      user_id, file_id, rating, (pos features...)
    """
    df = pd.read_csv(path)

    merged, response_cols = merge_response_columns(df)

    # Map ratings to numeric 1..5
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}")
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0, 5.0)

    # Compute POS features via spaCy
    feats = pos_features_for_docs(merged)

    out = pd.DataFrame({
        "user_id": np.arange(len(df)),
        "file_id": file_id,
        "rating": y
    })
    out = pd.concat([out, feats], axis=1)
    # Drop rows with missing ratings
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
    all_rows.append(prepare_pos_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity check
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
#    Feature matrix = all numeric POS features
feature_cols = [c for c in data.columns if c not in ["user_id","file_id","rating"]]
X = data[feature_cols].astype(float)
y = data["rating"].astype(float)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

# 3) Fit Random Forest
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
print(f"\nMSE (Random Forest on POS features): {mse:.4f}")

# 5) Predict for ALL rows (so we can compute per-user averages across 5 runs)
data["pred_rating"] = np.clip(rf.predict(X), 1.0, 5.0)

# 6) Per-user helpfulness averages
per_user = (
    data
    .groupby("user_id", as_index=False)
    .agg(
        true_avg_rating=("rating", "mean"),
        pred_avg_rating=("pred_rating", "mean"),
        n_rows=("rating", "size")
    )
)

# Optional rounding for nicer display
per_user["true_avg_rating"] = per_user["true_avg_rating"].round(2)
per_user["pred_avg_rating"] = per_user["pred_avg_rating"].round(2)

print("\nPer-user helpfulness averages (head):")
print(per_user.head())

# 7) Save outputs (stacked rows + per-user table)
data_out = data.copy()
# round some columns for readability
for tag in TRACK_POS:
    for kind in (f"pos_ratio_{tag}", f"pos_count_{tag}"):
        if kind in data_out.columns:
            data_out[kind] = data_out[kind].round(4)
data_out["pos_entropy"] = data_out["pos_entropy"].round(4)
data_out["pred_rating"] = data_out["pred_rating"].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)

print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")

# ---------------------------------------------------------------
# If you prefer per-user averages based ONLY on test rows
# (to avoid any leakage), uncomment below:
#
# test_only = data.loc[idx_test].copy()
# per_user_test_only = (
#     test_only
#     .groupby("user_id", as_index=False)
#     .agg(
#         true_avg_rating=("rating","mean"),
#         pred_avg_rating=("pred_rating","mean"),
#         n_test_rows=("rating","size")
#     )
# )
# per_user_test_only["true_avg_rating"] = per_user_test_only["true_avg_rating"].round(2)
# per_user_test_only["pred_avg_rating"] = per_user_test_only["pred_avg_rating"].round(2)
# per_user_test_only.to_csv("per_user_helpfulness_pos_TEST_ONLY.csv", index=False)
# print("\nSaved TEST-ONLY per-user averages to: per_user_helpfulness_pos_TEST_ONLY.csv")
# ---------------------------------------------------------------
