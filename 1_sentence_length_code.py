# import pandas as pd
# import numpy as np
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # -----------------------------
# # Config
# # -----------------------------
# input_file = "Q4_Submission1_cleaned.xlsx"   # change if needed
# target_col = "Rating"                        # label column
# output_csv = "train_test_predictions.csv"
# random_state = 42
# test_size = 0.2

# # Map categorical ratings to 1–5 (adjust if your labels differ)
# rating_map = {
#     "Not Helpful at All": 1,
#     "Slightly Helpful": 2,
#     "Neutral": 3,
#     "Helpful": 4,
#     "Very Helpful": 5
# }

# # -----------------------------
# # Load
# # -----------------------------
# df = pd.read_excel(input_file)

# # -----------------------------
# # Pick ONLY response columns and merge them
# # -----------------------------
# # Heuristic: any column name containing "response" (case-insensitive)
# response_cols = [c for c in df.columns if re.search(r"response", str(c), flags=re.IGNORECASE)]

# if not response_cols:
#     raise ValueError("No response columns found.")

# def join_responses(row) -> str:
#     parts = []
#     for c in response_cols:
#         val = row.get(c, None)
#         if pd.notna(val):
#             parts.append(str(val))
#     return " ".join(parts).strip()

# df["_merged_responses"] = df.apply(join_responses, axis=1)

# # -----------------------------
# # Build sentence_length (avg words per sentence)
# # -----------------------------
# sent_splitter  = re.compile(r"[.!?]+")
# word_tokenizer = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# def sentence_lengths_in_words(doc: str):
#     if not isinstance(doc, str) or not doc.strip():
#         return []
#     sentences = [s.strip() for s in sent_splitter.split(doc) if s.strip()]
#     return [len(word_tokenizer.findall(s)) for s in sentences]

# def avg_sentence_length(doc: str) -> float:
#     lens = sentence_lengths_in_words(doc)
#     return float(np.mean(lens)) if lens else 0.0

# df["sentence_length"] = df["_merged_responses"].apply(avg_sentence_length)

# # -----------------------------
# # Prepare target (1–5 scale)
# # -----------------------------
# if target_col not in df.columns:
#     raise ValueError(f"Target column '{target_col}' not found. Columns present: {df.columns.tolist()}")

# if df[target_col].dtype == "O":
#     y = df[target_col].map(rating_map)
# else:
#     y = df[target_col].astype(float).clip(1.0, 5.0)

# # Keep only valid rows (non-empty merged responses and valid ratings)
# mask = df["_merged_responses"].str.strip().astype(bool) & y.notna()
# df = df.loc[mask].reset_index(drop=True)
# y  = y.loc[mask].reset_index(drop=True)

# X = df[["sentence_length"]].copy()

# # -----------------------------
# # Train/test split (80/20)
# # -----------------------------
# X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
#     X, y, df.index, test_size=test_size, random_state=random_state
# )

# # -----------------------------
# # Model: Random Forest Regressor
# # -----------------------------
# rf = RandomForestRegressor(
#     n_estimators=400,
#     max_depth=None,
#     min_samples_leaf=2,
#     random_state=random_state,
#     n_jobs=-1
# )
# rf.fit(X_train, y_train)

# # -----------------------------
# # Predict & evaluate on test set
# # -----------------------------
# y_pred = rf.predict(X_test)
# mse = float(mean_squared_error(y_test, y_pred))


# print("Test set evaluation (sentence_length → helpfulness, 1–5):")
# print(f"  MSE: {mse:.4f}")
# # print(f"  MAE : {mae:.4f}")
# # print(f"  R^2 : {r2:.4f}")

# # -----------------------------
# # Save predictions (rounded)
# # -----------------------------
# out = df.copy()
# out["split"] = "train"
# out.loc[test_idx, "split"] = "test"

# # Predict for all rows and clip/round
# out["predicted_helpfulness"] = np.clip(rf.predict(X), 1.0, 5.0).round(2)
# out["sentence_length"] = out["sentence_length"].round(2)

# # Keep only requested columns + outputs
# keep_cols = response_cols + [target_col, "sentence_length", "predicted_helpfulness", "split"]
# out_final = out[keep_cols]
# out_final.to_csv(output_csv, index=False)
# print(f"Saved predictions to: {output_csv}")


import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
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
rating_col = "Rating"   # change if your rating column is named differently

# If ratings are text labels, map to 1..5:
rating_map = {
    "Not Helpful at All": 1,
    "Slightly Helpful": 2,
    "Neutral": 3,
    "Helpful": 4,
    "Very Helpful": 5,
}

# Where to save outputs
stacked_out_csv   = "stacked_sentence_length_rows.csv"
per_user_out_csv  = "per_user_helpfulness_averages.csv"

random_state = 42
test_size    = 0.2

# Response column detector (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Choose how to compute sentence length:
#   avg words per sentence (robust to #sentences)
#   or total words across all sentences (set USE_AVG=False)
USE_AVG_SENT_LEN = True

# =========================================
# =========== TOKENIZERS/HELPERS ==========
# =========================================
sent_splitter  = re.compile(r"[.!?]+")
word_tokenizer = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

def compute_sentence_length(text: str) -> float:
    """
    Compute sentence length on merged responses.
    If USE_AVG_SENT_LEN=True: average words per sentence
    Else: total words across all sentences
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    sentences = [s.strip() for s in sent_splitter.split(text) if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(word_tokenizer.findall(s)) for s in sentences]
    return float(np.mean(word_counts) if USE_AVG_SENT_LEN else np.sum(word_counts))

def load_file(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns per row, compute sentence length and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, sentence_length, rating
    """
    df = pd.read_csv(path)

    # Identify response columns
    response_cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not response_cols:
        raise ValueError(
            f"No response columns found in {path}. "
            f"Looking for columns containing 'response' (case-insensitive). "
            f"Columns: {df.columns.tolist()}"
        )

    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}. Columns: {df.columns.tolist()}")

    # Merge all response columns into one text field
    merged = df[response_cols].fillna("").agg(" ".join, axis=1)

    # Compute sentence length feature
    sl = merged.apply(compute_sentence_length)

    # Map ratings
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float)
        # If numeric but not in 1..5, clip:
        y = y.clip(1.0, 5.0)

    out = pd.DataFrame({
        "user_id": np.arange(len(df)),      # assumes same row index == same user across files
        "file_id": file_id,
        "sentence_length": sl.astype(float),
        "rating": y
    })
    # Keep only valid ratings
    out = out.loc[out["rating"].notna()].reset_index(drop=True)
    return out

# =========================================
# =============== PIPELINE ================
# =========================================
# 1) Load & stack all files
all_rows = []
for j, fpath in enumerate(files):
    p = Path(fpath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    all_rows.append(load_file(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Optional sanity checks
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
X = data[["sentence_length"]].copy()
y = data["rating"].astype(float)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

# 3) Fit Random Forest
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=random_state,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 4) Evaluate MSE on held-out test set
y_pred_test = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"\nMSE (Random Forest on stacked rows, feature = sentence_length): {mse:.4f}")

# 5) Predict for ALL rows (so we can compute per-user averages across their 5 runs)
#    NOTE: This uses a model trained on the 80% train split. Predictions for train users'
#    train rows are in-sample; comment below if you want "test-only" aggregates.
data["pred_rating"] = rf.predict(X).clip(1.0, 5.0)

# 6) Per-user helpfulness (average across the 5 runs)
per_user = (
    data
    .groupby("user_id", as_index=False)
    .agg(
        true_avg_rating = ("rating", "mean"),
        pred_avg_rating = ("pred_rating", "mean"),
        n_rows         = ("rating", "size")
    )
)

# Optional: round for display
per_user["true_avg_rating"] = per_user["true_avg_rating"].round(2)
per_user["pred_avg_rating"] = per_user["pred_avg_rating"].round(2)

print("\nPer-user helpfulness averages (head):")
print(per_user.head())

# 7) Save outputs for auditing
data_out = data.copy()
data_out["sentence_length"] = data_out["sentence_length"].round(2)
data_out["pred_rating"]     = data_out["pred_rating"].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)

print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")


# test_users = data.loc[idx_test, ["user_id", "rating"]].groupby("user_id").size().reset_index(name="test_rows")
# per_user_test_only = (
#     data.loc[idx_test]
#         .groupby("user_id", as_index=False)
#         .agg(true_avg_rating=("rating","mean"),
#              pred_avg_rating=("pred_rating","mean"),
#              n_test_rows=("rating","size"))
# )
# print("\nPer-user (TEST-ONLY) averages — head:")
# print(per_user_test_only.head())
# per_user_test_only.to_csv("per_user_helpfulness_averages_TEST_ONLY.csv", index=False)
