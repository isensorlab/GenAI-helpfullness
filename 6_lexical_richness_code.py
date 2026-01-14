# =========================================
# Lexical Richness → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge response columns -> one text per user
#    - Compute lexical richness features (TTR, RTTR, CTTR, Herdan C, Maas a², Yule K, MTLD, etc.)
#    - Map Rating -> 1..5
# 2) Stack (user, file) rows → 80/20 train/test split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across 5 runs
#
# Notes:
# - Response columns are detected by name containing "response" (case-insensitive).
# - MTLD uses default threshold 0.72 (classic).
# - All metrics are guarded for short texts; features become 0 or NaN-safe values when tokens are too few.

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import Counter

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

rating_col = "Rating"  # change to your label column

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
stacked_out_csv  = "stacked_lexical_richness_rows.csv"
per_user_out_csv = "per_user_helpfulness_lexical_richness_averages.csv"

# =========================================
# ============ TOKENIZATION ===============
# =========================================
# Basic, robust word tokenizer:
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

def tokenize(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return [w.lower() for w in WORD_RE.findall(text)]

# =========================================
# ======= LEXICAL RICHNESS METRICS ========
# =========================================
def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0

def ttr(types: int, tokens: int) -> float:
    return types / tokens if tokens > 0 else 0.0

def rttr(types: int, tokens: int) -> float:
    return types / math.sqrt(tokens) if tokens > 0 else 0.0  # Guiraud's R

def cttr(types: int, tokens: int) -> float:
    return types / math.sqrt(2.0 * tokens) if tokens > 0 else 0.0

def herdan_c(types: int, tokens: int) -> float:
    # log(types) / log(tokens)
    return (safe_log(types) / safe_log(tokens)) if tokens > 1 and types > 1 else 0.0

def maas_a2(types: int, tokens: int) -> float:
    # (log(tokens) - log(types)) / (log(tokens)^2)
    if tokens <= 1 or types <= 1:
        return 0.0
    lnN = safe_log(tokens)
    return (lnN - safe_log(types)) / (lnN * lnN) if lnN > 0 else 0.0

def yules_k(freqs: Counter) -> float:
    # Yule's K = 10^4 * ( (sum i^2 * V_i) - N ) / N^2
    # where V_i = number of types with frequency i, N = total tokens
    N = sum(freqs.values())
    if N == 0:
        return 0.0
    V_i = Counter(freqs.values())
    sum_i2Vi = sum((i * i) * Vi for i, Vi in V_i.items())
    K = (sum_i2Vi - N) / (N * N)
    return 1e4 * K

def mtld(tokens: List[str], ttr_threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (MTLD).
    Classic implementation (forward and backward; average both).
    Returns 0 if insufficient tokens.
    """
    def mtld_seq(seq):
        if len(seq) < 10:
            return 0.0
        factors = 0
        types_set = set()
        token_count = 0
        for w in seq:
            token_count += 1
            types_set.add(w)
            current_ttr = len(types_set) / token_count
            if current_ttr <= ttr_threshold:
                factors += 1
                types_set = set()
                token_count = 0
        # partial factor
        if token_count > 0:
            factors += (1 - (len(types_set) / max(token_count, 1))) / (1 - ttr_threshold)
        return len(seq) / factors if factors > 0 else 0.0

    if not tokens:
        return 0.0
    forward = mtld_seq(tokens)
    backward = mtld_seq(list(reversed(tokens)))
    if forward == 0.0 and backward == 0.0:
        return 0.0
    return (forward + backward) / (2 if (forward > 0 and backward > 0) else 1)

def lexical_richness_features(text: str) -> dict:
    toks = tokenize(text)
    N = len(toks)
    if N == 0:
        return {
            "tokens": 0, "types": 0,
            "ttr": 0.0, "rttr": 0.0, "cttr": 0.0, "herdan_c": 0.0,
            "maas_a2": 0.0, "yules_k": 0.0, "mtld": 0.0,
            "hapax_ratio": 0.0, "dis_legomena_ratio": 0.0,
            "avg_word_len": 0.0
        }

    freqs = Counter(toks)
    V = len(freqs)
    # hapax, dis legomena
    hapax = sum(1 for w, f in freqs.items() if f == 1)
    disleg = sum(1 for w, f in freqs.items() if f == 2)

    avg_len = np.mean([len(w) for w in toks]) if N > 0 else 0.0

    return {
        "tokens": N,
        "types": V,
        "ttr": ttr(V, N),
        "rttr": rttr(V, N),
        "cttr": cttr(V, N),
        "herdan_c": herdan_c(V, N),
        "maas_a2": maas_a2(V, N),
        "yules_k": yules_k(freqs),
        "mtld": mtld(toks),
        "hapax_ratio": (hapax / V) if V > 0 else 0.0,
        "dis_legomena_ratio": (disleg / V) if V > 0 else 0.0,
        "avg_word_len": float(avg_len),
    }

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

def prepare_lexical_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute lexical richness features, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (lexical features...)
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

    feats = [lexical_richness_features(txt) for txt in merged]

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
    all_rows.append(prepare_lexical_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
feature_cols = [
    "tokens","types","ttr","rttr","cttr","herdan_c","maas_a2","yules_k","mtld",
    "hapax_ratio","dis_legomena_ratio","avg_word_len"
]
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
print(f"\nMSE (Random Forest on lexical richness features): {mse:.4f}")

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
