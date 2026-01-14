# =========================================
# Perplexity-based helpfulness prediction (1–5)
# =========================================
# What this script does:
# 1) For each of 5 CSVs:
#    - Merge response columns -> 1 text per user
#    - Compute perplexity with a Hugging Face causal LM (GPT-2 family)
#    - Map Rating -> 1..5 (if text labels)
# 2) Stack (user, file) rows -> 80/20 split
# 3) Train RandomForestRegressor -> print MSE on test
# 4) Compute per-user averages: true vs predicted
#
# Notes:
# - Uses a sliding window so long texts never exceed the model's context window.
# - Set MODEL_NAME to "distilgpt2" for speed, "gpt2" (default) for stronger baseline,
#   or "gpt2-medium" for even stronger (slower).
# - If your response columns don't contain the word "response",
#   adapt RESPONSE_COL_PATTERN accordingly.

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# ---------- ML / HF ----------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

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

rating_col = "Rating"   # change if your label column name differs

# Map textual labels -> 1..5 (edit if your labels differ)
rating_map: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

# Hugging Face model for perplexity
MODEL_NAME = "gpt2"     # "distilgpt2" (fast), "gpt2" (default), "gpt2-medium" (slower/stronger)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Perplexity windowing settings (GPT-2 context is 1024 tokens)
MAX_CONTEXT = 1024   # model max context window (we'll cap at 1024 for GPT-2 family)
STRIDE      = 512    # overlap to reduce boundary artifacts

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_perplexity_rows.csv"
per_user_out_csv = "per_user_helpfulness_perplexity_averages.csv"

# =========================================
# ======= Perplexity with Hugging Face =====
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 has no pad token by default — set pad_token to EOS to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Set actual max length from model config, but cap to 1024 (GPT-2 behavior)
model_max_len = getattr(model.config, "n_positions", MAX_CONTEXT)
MAX_LENGTH = min(int(model_max_len), MAX_CONTEXT)

@torch.no_grad()
def text_perplexity(text: str) -> float:
    """
    Sliding-window perplexity per Hugging Face recipe.
    Returns exp(mean negative log-likelihood per token).
    """
    if not isinstance(text, str) or not text.strip():
        return float("inf")

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    n_tokens = input_ids.size(1)

    if n_tokens == 0:
        return float("inf")

    # Short text: single pass
    if n_tokens <= MAX_LENGTH:
        outputs = model(input_ids, labels=input_ids)
        return float(math.exp(outputs.loss.item()))

    # Long text: stride windows with overlap
    nlls = []
    prev_end = 0
    for begin_loc in range(0, n_tokens, STRIDE):
        end_loc = min(begin_loc + MAX_LENGTH, n_tokens)
        trg_len = end_loc - prev_end  # tokens we compute loss on (avoid double-counting overlap)

        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        # mask out everything except the last trg_len tokens
        if target_ids.size(1) > trg_len:
            target_ids[:, :-trg_len] = -100

        outputs = model(input_ids_slice, labels=target_ids)
        nlls.append(outputs.loss.item() * trg_len)
        prev_end = end_loc
        if end_loc == n_tokens:
            break

    total_nll = float(sum(nlls))
    # exact count of predicted tokens over all slices
    total_pred_tokens = float(sum(
        min(MAX_LENGTH, n_tokens - i) if i == 0 else min(STRIDE, n_tokens - i)
        for i in range(0, n_tokens, STRIDE)
    ))
    if total_pred_tokens <= 0:
        return float("inf")

    ppl = math.exp(total_nll / total_pred_tokens)
    return float(ppl)

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

def prepare_perplexity_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute perplexity, and map rating to 1..5.
    Returns DataFrame with: user_id, file_id, perplexity, rating
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

    # Compute perplexity per row
    it = merged if not TQDM else tqdm(merged, desc=f"PPL {Path(path).name}", unit="row")
    ppls = []
    for text in it:
        try:
            ppls.append(text_perplexity(text))
        except Exception:
            # Any unexpected HF/tokenization issue: mark as very high perplexity
            ppls.append(float("inf"))

    out = pd.DataFrame({
        "user_id": np.arange(len(df)),  # assumes same row index == same user across files
        "file_id": file_id,
        "perplexity": ppls,
        "rating": y
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
    all_rows.append(prepare_perplexity_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Basic sanity output
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Clean feature and split
X = data[["perplexity"]].astype(float)
y = data["rating"].astype(float)

# Replace inf/NaN perplexities with a high sentinel (or the column max if finite)
finite_max = X["perplexity"].replace([np.inf, -np.inf], np.nan).max()
fill_value = finite_max if np.isfinite(finite_max) else 1e6
X["perplexity"] = X["perplexity"].replace([np.inf, -np.inf], np.nan).fillna(fill_value)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

# 3) Train Random Forest
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
print(f"\nMSE (Random Forest on perplexity): {mse:.4f}")

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
data_out["perplexity"]  = data_out["perplexity"].round(3)
data_out["pred_rating"] = data_out["pred_rating"].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)

print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
