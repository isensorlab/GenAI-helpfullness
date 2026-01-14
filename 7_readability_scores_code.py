# =========================================
# Readability → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge response columns → one text per user
#    - Compute readability features (Flesch, FKGL, Fog, SMOG, ARI, Coleman–Liau)
#    - Map Rating → 1..5 (if labels are strings)
# 2) Stack (user, file) rows → 80/20 train/test split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across 5 runs
#
# Notes:
# - Response columns are detected by names containing "response" (case-insensitive).
# - Syllable counter is heuristic but robust for large-scale analysis.
# - SMOG is adjusted for short texts (common in chat); uses standard extrapolation.

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

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

# Detect response columns (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_readability_rows.csv"
per_user_out_csv = "per_user_helpfulness_readability_averages.csv"

# =========================================
# ============ TOKENIZATION ===============
# =========================================
SENT_SPLIT_RE = re.compile(r"[.!?]+")
WORD_RE       = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

def sentence_split(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

def tokenize(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return WORD_RE.findall(text)

# =========================================
# =========== SYLLABLE HEURISTIC ==========
# =========================================
VOWELS = "aeiouy"

def count_syllables(word: str) -> int:
    """
    Rough English syllable heuristic:
    - Count vowel groups.
    - Adjust for silent 'e' at end, 'le' endings, etc.
    - Ensure at least 1 syllable for any alphabetic word.
    """
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    # Remove trailing 'e' if not 'le'
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    if not w:
        return 1

    groups = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_is_vowel:
            groups += 1
        prev_is_vowel = is_vowel

    # 'le' endings (bottle, little) add a syllable
    if w.endswith("le") and len(w) > 2 and w[-3] not in VOWELS:
        groups += 1

    return max(groups, 1)

def count_syllables_in_tokens(tokens: List[str]) -> int:
    return sum(count_syllables(tok) for tok in tokens)

# =========================================
# ======= READABILITY METRIC HELPERS ======
# =========================================
def safe_div(a, b, default=0.0):
    return (a / b) if (b and b != 0) else default

def readability_features(text: str) -> dict:
    """
    Compute a set of classic readability metrics and supporting stats.
    Returns a dict of numeric features (0-safe).
    """
    sents = sentence_split(text)
    tokens = tokenize(text)
    words = [w for w in tokens if re.search(r"[A-Za-z]", w)]
    n_sent = max(len(sents), 1)   # guard to avoid div-by-zero
    n_words = len(words)
    n_chars = sum(len(w) for w in words)

    # Syllables & polysyllables
    syllables = count_syllables_in_tokens(words) if n_words > 0 else 0
    polysyllables = sum(1 for w in words if count_syllables(w) >= 3)

    # Basic rates
    words_per_sentence   = safe_div(n_words, n_sent, default=0.0)
    syllables_per_word   = safe_div(syllables, n_words, default=0.0)
    chars_per_word       = safe_div(n_chars, n_words, default=0.0)
    polysyllable_rate    = safe_div(polysyllables, n_words, default=0.0)

    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    # Flesch-Kincaid Grade
    fkgl   = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    # Gunning Fog
    fog    = 0.4 * (words_per_sentence + 100.0 * polysyllable_rate)
    # Automated Readability Index (ARI)
    ari    = 4.71 * chars_per_word + 0.5 * words_per_sentence - 21.43
    # Coleman–Liau: L = letters per 100 words, S = sentences per 100 words
    L = safe_div(n_chars, n_words, 0.0) * 100.0
    S = safe_div(n_sent, n_words, 0.0) * 100.0
    coleman_liau = 0.0588 * L - 0.296 * S - 15.8

    # SMOG (adjust for short texts via extrapolation)
    # Standard: 1.0430 * sqrt(30 * (polysyllables / sentences)) + 3.1291
    smog = 0.0
    if polysyllables > 0 and n_sent > 0:
        smog = 1.0430 * math.sqrt(30.0 * (polysyllables / n_sent)) + 3.1291

    return {
        "n_sentences": n_sent,
        "n_words": n_words,
        "n_chars": n_chars,
        "syllables": syllables,
        "polysyllables": polysyllables,

        "words_per_sentence": round(words_per_sentence, 4),
        "syllables_per_word": round(syllables_per_word, 4),
        "chars_per_word": round(chars_per_word, 4),
        "polysyllable_rate": round(polysyllable_rate, 4),

        "flesch_reading_ease": round(flesch, 4),
        "fk_grade_level": round(fkgl, 4),
        "gunning_fog": round(fog, 4),
        "smog_index": round(smog, 4),
        "ari": round(ari, 4),
        "coleman_liau": round(coleman_liau, 4),
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

def prepare_readability_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute readability features, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (readability features...)
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

    feats = [readability_features(txt) for txt in merged]
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
    all_rows.append(prepare_readability_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
feature_cols = [
    "n_sentences","n_words","n_chars","syllables","polysyllables",
    "words_per_sentence","syllables_per_word","chars_per_word","polysyllable_rate",
    "flesch_reading_ease","fk_grade_level","gunning_fog","smog_index","ari","coleman_liau"
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
print(f"\nMSE (Random Forest on readability features): {mse:.4f}")

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
