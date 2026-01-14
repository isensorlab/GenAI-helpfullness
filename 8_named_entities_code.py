# =========================================
# Named Entities (NER) → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge response columns → one text per user
#    - Run spaCy NER; compute features:
#        * token_count, entity_count, entity_density
#        * unique_entity_types, entity_label_entropy
#        * avg_entity_length_tokens
#        * per-label counts for common labels (PERSON, ORG, GPE, ...)
#    - Map Rating → 1..5
# 2) Stack (user, file) rows → 80/20 split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across 5 runs

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict

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

rating_col = "Rating"  # change if needed

# Map textual labels → 1..5
rating_map: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

# Detect response columns (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Outputs
stacked_out_csv  = "stacked_ner_rows.csv"
per_user_out_csv = "per_user_helpfulness_ner_averages.csv"

# Train/test split
random_state = 42
test_size    = 0.20

# Labels to track explicitly (spaCy en_core_web_sm)
NER_LABELS = [
    "PERSON","ORG","GPE","LOC","NORP","FAC","EVENT","WORK_OF_ART","LAW","LANGUAGE","PRODUCT",
    "DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
]

# =========================================
# ============ spaCy PIPELINE =============
# =========================================
# NER must be enabled (keep default components)
nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 2_000_000  # uncomment if you have very long rows

# =========================================
# ============ HELPER FUNCTIONS ===========
# =========================================
def merge_responses(df: pd.DataFrame) -> pd.Series:
    """
    Merge all columns whose name contains 'response' (case-insensitive) into one text per row.
    """
    resp_cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not resp_cols:
        raise ValueError(
            "No response columns found. Adjust RESPONSE_COL_PATTERN or rename columns to include 'response'. "
            f"Columns present: {df.columns.tolist()}"
        )
    merged = df[resp_cols].fillna("").agg(" ".join, axis=1).str.strip()
    return merged, resp_cols

def ner_features_for_docs(texts, batch_size=200) -> pd.DataFrame:
    """
    Compute NER-based features for an iterable of texts using spaCy in batches.
    Returns DataFrame with one row per text containing:
      - token_count, entity_count, entity_density
      - unique_entity_types, entity_label_entropy
      - avg_entity_length_tokens
      - per-label counts for labels in NER_LABELS (prefixed with ent_count_)
    """
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        # Tokens excluding pure spaces (punct is fine since tokenizer includes it)
        tokens = [t for t in doc if not t.is_space]
        token_count = len(tokens)

        ents = list(doc.ents)
        entity_count = len(ents)
        entity_density = (entity_count / token_count) if token_count > 0 else 0.0

        # Per-label counts
        label_counts = Counter(ent.label_ for ent in ents)
        unique_entity_types = len(label_counts)

        # Entropy over label distribution
        if entity_count > 0:
            probs = np.array([c / entity_count for c in label_counts.values()], dtype=float)
            entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
        else:
            entropy = 0.0

        # Avg entity length in tokens
        if entity_count > 0:
            avg_ent_len = float(np.mean([len(ent) for ent in ents]))
        else:
            avg_ent_len = 0.0

        row = {
            "token_count": float(token_count),
            "entity_count": float(entity_count),
            "entity_density": float(entity_density),
            "unique_entity_types": float(unique_entity_types),
            "entity_label_entropy": float(entropy),
            "avg_entity_length_tokens": float(avg_ent_len),
        }

        # Add per-label counts (0 if absent)
        for lbl in NER_LABELS:
            row[f"ent_count_{lbl}"] = float(label_counts.get(lbl, 0))

        rows.append(row)

    return pd.DataFrame(rows)

def prepare_ner_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute NER features, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (ner features...)
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

    feats_df = ner_features_for_docs(merged)

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
    all_rows.append(prepare_ner_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
exclude_cols = {"user_id","file_id","rating"}
feature_cols = [c for c in data.columns if c not in exclude_cols]
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
print(f"\nMSE (Random Forest on NER features): {mse:.4f}")

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
# round some readable columns
round_cols = ["token_count","entity_count","entity_density","unique_entity_types",
              "entity_label_entropy","avg_entity_length_tokens"]
for c in round_cols:
    if c in data_out.columns:
        data_out[c] = data_out[c].round(4)
data_out["pred_rating"] = data_out["pred_rating"].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)
print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
