# =========================================
# Dependency Tree Depth → Helpfulness (1–5)
# =========================================
# What this script does:
# 1) For each of 5 CSVs:
#    - Merge response columns -> 1 text per user
#    - Run spaCy to parse dependencies
#    - Compute dependency-tree features per user:
#        * avg_tree_depth, max_tree_depth, std_tree_depth
#        * avg_dep_distance (mean |i - head_i| per token)
#        * root_children_mean (avg #children of sentence roots)
#    - Map Rating -> 1..5
# 2) Stack all (user, file) rows → 80/20 train/test split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted (across the 5 runs)
#
# Notes:
# - Response columns are auto-detected by name containing "response" (case-insensitive).
# - If your columns use different names, tweak RESPONSE_COL_PATTERN.

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

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

rating_col = "Rating"   # change if your label column is different

# Map textual labels -> 1..5 (edit if your labels differ)
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
stacked_out_csv  = "stacked_dependency_rows.csv"
per_user_out_csv = "per_user_helpfulness_dependency_averages.csv"

# =========================================
# ============ spaCy PIPELINE =============
# =========================================
# Load English model; disable NER for speed (POS+parser are needed)
nlp = spacy.load("en_core_web_sm", disable=["ner"])
# If you have very long responses, you may increase this:
# nlp.max_length = 2_000_000

# =========================================
# ============ HELPER FUNCTIONS ===========
# =========================================
def merge_responses(df: pd.DataFrame) -> pd.Series:
    """Merge all columns whose name contains 'response' into one text per row."""
    resp_cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not resp_cols:
        raise ValueError(
            "No response columns found. Rename or adjust RESPONSE_COL_PATTERN. "
            f"Columns present: {df.columns.tolist()}"
        )
    merged = df[resp_cols].fillna("").agg(" ".join, axis=1).str.strip()
    return merged, resp_cols

def sentence_tree_depth(sent) -> int:
    """
    Depth of a dependency tree for one sentence:
    length of the longest path from the root to any token.
    """
    # cache depths per token to avoid recomputation
    depth_cache = {}

    def token_depth(tok):
        if tok in depth_cache:
            return depth_cache[tok]
        d = 0
        cur = tok
        # climb heads until root (tok.head == tok for root)
        while cur.head != cur:
            d += 1
            cur = cur.head
        depth_cache[tok] = d
        return d

    if len(sent) == 0:
        return 0
    return max(token_depth(tok) for tok in sent)

def sentence_dep_distance(sent) -> float:
    """
    Mean absolute dependency distance: average |token.i - token.head.i| over tokens (excluding roots).
    """
    dists = []
    for tok in sent:
        if tok.head != tok:  # exclude root
            dists.append(abs(tok.i - tok.head.i))
    return float(np.mean(dists)) if dists else 0.0

def sentence_root_children(sent) -> int:
    """Number of children of the sentence root."""
    for tok in sent:
        if tok.head == tok:  # root
            return len(list(tok.children))
    return 0

def dependency_features_for_docs(texts, batch_size=200) -> pd.DataFrame:
    """
    Compute dependency-based features for an iterable of texts using spaCy in batches.
    Returns DataFrame with one row per text:
      - avg_tree_depth, max_tree_depth, std_tree_depth
      - avg_dep_distance
      - root_children_mean
      - n_sentences, n_tokens
    """
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        depths = []
        dep_dists = []
        root_children = []
        sent_count = 0
        token_count = 0

        # ensure we have sentence boundaries; if not, set sentencizer
        if not doc.has_annotation("SENT_START"):
            # optional: add sentencizer if needed
            pass

        for sent in doc.sents:
            sent_count += 1
            token_count += len([t for t in sent if not (t.is_space)])
            d = sentence_tree_depth(sent)
            depths.append(d)
            dep_dists.append(sentence_dep_distance(sent))
            root_children.append(sentence_root_children(sent))

        if sent_count == 0:
            # Fallback: treat full doc as one span if no sentence splits
            span = doc[:]
            sent_count = 1
            token_count = len([t for t in span if not t.is_space])
            depths = [sentence_tree_depth(span)]
            dep_dists = [sentence_dep_distance(span)]
            root_children = [sentence_root_children(span)]

        row = {
            "avg_tree_depth": float(np.mean(depths)),
            "max_tree_depth": float(np.max(depths)),
            "std_tree_depth": float(np.std(depths)),
            "avg_dep_distance": float(np.mean(dep_dists)),
            "root_children_mean": float(np.mean(root_children)),
            "n_sentences": int(sent_count),
            "n_tokens": int(token_count),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def prepare_dependency_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute dependency features, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (dep features...)
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

    feats = dependency_features_for_docs(merged)

    out = pd.DataFrame({
        "user_id": np.arange(len(df)),
        "file_id": file_id,
        "rating": y
    })
    out = pd.concat([out, feats], axis=1)
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
    all_rows.append(prepare_dependency_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity output
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
feature_cols = [
    "avg_tree_depth", "max_tree_depth", "std_tree_depth",
    "avg_dep_distance", "root_children_mean",
    "n_sentences", "n_tokens"
]
X = data[feature_cols].astype(float)
y = data["rating"].astype(float)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

# 3) Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=random_state,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 4) Evaluate on held-out test set
y_pred_test = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"\nMSE (Random Forest on dependency-depth features): {mse:.4f}")

# 5) Predict for ALL rows (for per-user averages across 5 runs)
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

# 6) Save outputs
data_out = data.copy()
for col in feature_cols:
    data_out[col] = data_out[col].round(3)
data_out["pred_rating"] = data_out["pred_rating"].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)

print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
