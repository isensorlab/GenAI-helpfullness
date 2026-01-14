# =========================================
# Discourse Markers (PDTB-style) → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge response columns → one text per user
#    - Count discourse markers (connectives) by category:
#         Temporal, Comparison, Contingency/Cause, Expansion/Additive
#    - Features: totals, densities, diversity (entropy), per-category counts,
#      enumeration_count (first/second/third...), avg_connective_position
#    - Map Rating → 1..5
# 2) Stack (user, file) rows → 80/20 split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across 5 runs

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

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

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_discourse_rows.csv"
per_user_out_csv = "per_user_helpfulness_discourse_averages.csv"

# =========================================
# ======= Discourse Marker Inventories =====
# =========================================
# Compact PDTB-style sets (edit/extend as desired).
# All matched case-insensitively with word boundaries.
TEMPORAL = [
    "after", "afterward", "afterwards", "before", "earlier", "later", "meantime",
    "meanwhile", "subsequently", "then", "next", "finally", "first", "second",
    "third", "fourth", "ultimately", "previously", "simultaneously", "soon",
    "eventually"
]
COMPARISON = [
    "however", "but", "nevertheless", "nonetheless", "yet", "though", "although",
    "whereas", "while", "instead", "in contrast", "on the other hand", "rather"
]
CONTINGENCY = [
    "because", "since", "as", "so", "therefore", "thus", "hence", "consequently",
    "accordingly", "as a result", "resultantly", "due to", "owing to"
]
EXPANSION = [
    "and", "also", "moreover", "furthermore", "in addition", "besides",
    "additionally", "namely", "for example", "for instance", "that is",
    "in other words"
]

# Build regex patterns with word boundaries; handle multiword by joining spaces
def pattern_list(words: List[str]) -> List[re.Pattern]:
    pats = []
    for w in words:
        w_escaped = re.escape(w)
        # allow internal spaces for multi-word connectives
        w_escaped = w_escaped.replace(r"\ ", r"\s+")
        pats.append(re.compile(rf"\b{w_escaped}\b", re.IGNORECASE))
    return pats

PAT_TEMPORAL   = pattern_list(TEMPORAL)
PAT_COMPARISON = pattern_list(COMPARISON)
PAT_CAUSE      = pattern_list(CONTINGENCY)
PAT_EXPANSION  = pattern_list(EXPANSION)

# Simple tokenization for counts/positions
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# Enumeration indicator list (ordinal-style structuring)
ENUM_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth"]

# =========================================
# ============ HELPERS / LOADING ==========
# =========================================
def merge_responses(df: pd.DataFrame) -> pd.Series:
    resp_cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not resp_cols:
        raise ValueError(
            "No response columns found. Adjust RESPONSE_COL_PATTERN or rename columns to include 'response'. "
            f"Columns present: {df.columns.tolist()}"
        )
    merged = df[resp_cols].fillna("").agg(" ".join, axis=1).str.strip()
    return merged, resp_cols

def find_all_matches(text: str, patterns: List[re.Pattern]) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx) for all pattern matches in text.
    """
    spans = []
    for pat in patterns:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return spans

def discourse_features(text: str) -> dict:
    """
    Compute discourse marker features for a single response string.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "token_count": 0,
            "total_connectives": 0,
            "connective_density_per_100w": 0.0,
            "unique_connectives": 0,
            "connective_entropy": 0.0,
            "count_temporal": 0, "count_comparison": 0, "count_cause": 0, "count_expansion": 0,
            "density_temporal": 0.0, "density_comparison": 0.0, "density_cause": 0.0, "density_expansion": 0.0,
            "enumeration_count": 0,
            "avg_connective_position": 0.0
        }

    tokens = WORD_RE.findall(text)
    n_tokens = len(tokens)

    # category matches (character spans)
    spans_temporal   = find_all_matches(text, PAT_TEMPORAL)
    spans_comparison = find_all_matches(text, PAT_COMPARISON)
    spans_cause      = find_all_matches(text, PAT_CAUSE)
    spans_expansion  = find_all_matches(text, PAT_EXPANSION)

    # Count totals
    ct_temporal   = len(spans_temporal)
    ct_comparison = len(spans_comparison)
    ct_cause      = len(spans_cause)
    ct_expansion  = len(spans_expansion)

    total = ct_temporal + ct_comparison + ct_cause + ct_expansion

    # Density per 100 words
    per100 = (100.0 / n_tokens) if n_tokens > 0 else 0.0
    d_temporal   = ct_temporal   * per100
    d_comparison = ct_comparison * per100
    d_cause      = ct_cause      * per100
    d_expansion  = ct_expansion  * per100

    # Unique connective surface forms found (by literal, case-insensitive)
    found_set = set()
    for group, vocab in [("T", TEMPORAL), ("C", COMPARISON), ("K", CONTINGENCY), ("E", EXPANSION)]:
        for w in vocab:
            # quick, case-insensitive whole-word match
            w_escaped = re.escape(w).replace(r"\ ", r"\s+")
            if re.search(rf"\b{w_escaped}\b", text, flags=re.IGNORECASE):
                found_set.add(w.lower())
    unique_connectives = len(found_set)

    # Entropy over category distribution (how balanced the use is)
    if total > 0:
        dist = np.array([ct_temporal, ct_comparison, ct_cause, ct_expansion], dtype=float) / total
        entropy = -float(np.sum(dist * np.log2(dist + 1e-12)))
    else:
        entropy = 0.0

    # Enumeration words (structuring lists/steps)
    enumeration_count = 0
    if n_tokens > 0:
        lower_toks = [t.lower() for t in tokens]
        enumeration_count = sum(lower_toks.count(w) for w in ENUM_WORDS)

    # Approximate average connective position as mean token index / n_tokens
    # (rough: use first token index at or after each char-start)
    def token_index_for_charpos(charpos: int) -> int:
        # scan tokens to find the one whose start char is just <= charpos
        # build token spans once
        return 0

    # Build simple token character offsets
    tok_spans = []
    cursor = 0
    for t in tokens:
        # find next occurrence of t from cursor (case-insensitive search on original text is harder;
        # instead, approximate via incremental scan on a lowercase version)
        # Simplify: we won't compute exact token positions; we fallback to using proportion of char index.
        tok_spans.append(None)

    # Fallback: use character positions normalized by text length as proxy
    text_len = len(text)
    char_positions = [s for s, e in (spans_temporal + spans_comparison + spans_cause + spans_expansion)]
    avg_pos_norm = (np.mean([p / text_len for p in char_positions]) if (text_len > 0 and char_positions) else 0.0)

    return {
        "token_count": int(n_tokens),
        "total_connectives": int(total),
        "connective_density_per_100w": float(total * per100 if n_tokens > 0 else 0.0),
        "unique_connectives": int(unique_connectives),
        "connective_entropy": float(entropy),
        "count_temporal": int(ct_temporal),
        "count_comparison": int(ct_comparison),
        "count_cause": int(ct_cause),
        "count_expansion": int(ct_expansion),
        "density_temporal": float(d_temporal),
        "density_comparison": float(d_comparison),
        "density_cause": float(d_cause),
        "density_expansion": float(d_expansion),
        "enumeration_count": int(enumeration_count),
        "avg_connective_position": float(round(avg_pos_norm, 4)),  # 0..1
    }

def prepare_discourse_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute discourse features, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (discourse features...)
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

    feats = [discourse_features(txt) for txt in merged]
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
    all_rows.append(prepare_discourse_rows(str(p), j))

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
print(f"\nMSE (Random Forest on discourse marker features): {mse:.4f}")

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
