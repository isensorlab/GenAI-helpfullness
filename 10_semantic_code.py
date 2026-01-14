# =========================================
# Semantic Embeddings (Q↔R similarity) → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge QUESTION columns -> one text per user (e.g., columns containing 'question' or 'prompt')
#    - Merge RESPONSE columns -> one text per user (columns containing 'response')
#    - Embed with Sentence-Transformers (all-MiniLM-L6-v2)
#    - Compute features: cosine_similarity, ||q||, ||r||, | ||q|| - ||r|| |
#    - Map Rating -> 1..5
# 2) Stack (user, file) rows → 80/20 split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across 5 runs

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
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

# Map textual labels → 1..5 (edit if your labels differ)
rating_map: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

# Column detectors (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)
QUESTION_COL_PATTERN = re.compile(r"(question|prompt|query)", re.IGNORECASE)

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_semantic_rows.csv"
per_user_out_csv = "per_user_helpfulness_semantic_averages.csv"

# =========================================
# ========= SENTENCE TRANSFORMER ==========
# =========================================
# all-MiniLM-L6-v2: 384-dim, fast & solid for semantic similarity
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================================
# ============ HELPERS / LOADING ==========
# =========================================
def merge_cols(df: pd.DataFrame, pattern: re.Pattern) -> pd.Series:
    """
    Merge all columns whose name matches `pattern` into one text per row.
    Returns a single Series of merged text; raises if none found (for responses).
    """
    cols = [c for c in df.columns if pattern.search(str(c))]
    if not cols:
        if pattern is RESPONSE_COL_PATTERN:
            raise ValueError(
                "No response columns found. Rename or adjust RESPONSE_COL_PATTERN "
                f"(columns present: {df.columns.tolist()})"
            )
        # For questions: allow empty string fallback
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between A and B (same shape: n×d).
    Returns (n,) vector.
    """
    # Normalize to unit vectors
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.sum(A_norm * B_norm, axis=1)

def prepare_semantic_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge questions and responses, embed, compute similarity features, map rating → 1..5.
    Returns DataFrame with: user_id, file_id, rating, cosine_sim, q_norm, r_norm, norm_gap
    """
    df = pd.read_csv(path)

    responses = merge_cols(df, RESPONSE_COL_PATTERN)
    questions = merge_cols(df, QUESTION_COL_PATTERN)

    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}")

    # Ratings → numeric 1..5
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0, 5.0)

    # Encode in batches (SentenceTransformer handles batching internally)
    q_emb = model.encode(questions.tolist(), convert_to_numpy=True, show_progress_bar=False)
    r_emb = model.encode(responses.tolist(), convert_to_numpy=True, show_progress_bar=False)

    cos_sim = cosine_similarity_matrix(q_emb, r_emb)
    q_norm  = np.linalg.norm(q_emb, axis=1)
    r_norm  = np.linalg.norm(r_emb, axis=1)
    norm_gap = np.abs(q_norm - r_norm)

    out = pd.DataFrame({
        "user_id": np.arange(len(df)),
        "file_id": file_id,
        "rating": y.values,
        "cosine_sim": cos_sim.astype(float),
        "q_norm": q_norm.astype(float),
        "r_norm": r_norm.astype(float),
        "norm_gap": norm_gap.astype(float),
    })
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
    all_rows.append(prepare_semantic_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity output
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
feature_cols = ["cosine_sim", "q_norm", "r_norm", "norm_gap"]
X = data[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
print(f"\nMSE (Random Forest on semantic similarity features): {mse:.4f}")

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
for col in feature_cols + ["pred_rating"]:
    data_out[col] = data_out[col].round(4 if col != "pred_rating" else 2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)
print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
