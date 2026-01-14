# =========================================
# GLTR-style Token-Rank features → Helpfulness (1–5)
# =========================================
# What this script does:
# 1) For each of 5 CSVs:
#    - Merge response columns -> 1 text/user
#    - Compute token-rank stats with a causal LM (GPT-2 family)
#      Bins: top-10, top-100, top-1000, >1000
#      Aggregates: proportions, mean/median rank, entropy over bins
#    - Map Rating -> 1..5
# 2) Stack (user,file) rows → 80/20 train/test split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across the 5 runs
#
# Notes:
# - Uses sliding windows so long texts never exceed the model's context.
# - Set MODEL_NAME to "distilgpt2" for speed, "gpt2" (default), or "gpt2-medium" (slower/stronger).
# - If your response columns don’t contain “response”, change RESPONSE_COL_PATTERN.

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

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

rating_col = "Rating"   # change if different
rating_map: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

# Hugging Face model
MODEL_NAME = "gpt2"    # "distilgpt2" for speed, "gpt2" default, "gpt2-medium" stronger/slower

# Token windowing (GPT-2 context)
MAX_CONTEXT = 1024
STRIDE      = 512

# Detect response columns (case-insensitive)
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_tokenrank_rows.csv"
per_user_out_csv = "per_user_helpfulness_tokenrank_averages.csv"

# =========================================
# ============ HF Model Setup =============
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

model_max_len = getattr(model.config, "n_positions", MAX_CONTEXT)
MAX_LENGTH = min(int(model_max_len), MAX_CONTEXT)

@torch.no_grad()
def token_rank_bins_for_text(text: str):
    """
    Compute GLTR-style token-rank features for a single text.

    Returns dict with:
      - proportion_top10 / top100 / top1000 / over1000
      - mean_rank, median_rank
      - rank_entropy (over the 4-bin distribution)
      - n_pred_tokens (how many tokens were evaluated)

    Method:
      For each token position t>0, we take logits for t from the model
      given tokens up to t-1, and compute the RANK of the actual token at t
      among the vocabulary (1 = highest probability). We then bin those ranks.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "proportion_top10": 0.0, "proportion_top100": 0.0,
            "proportion_top1000": 0.0, "proportion_over1000": 0.0,
            "mean_rank": float("nan"), "median_rank": float("nan"),
            "rank_entropy": 0.0, "n_pred_tokens": 0
        }

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    n_tokens  = input_ids.size(1)

    if n_tokens <= 1:
        return {
            "proportion_top10": 0.0, "proportion_top100": 0.0,
            "proportion_top1000": 0.0, "proportion_over1000": 0.0,
            "mean_rank": float("nan"), "median_rank": float("nan"),
            "rank_entropy": 0.0, "n_pred_tokens": 0
        }

    # We'll accumulate ranks across windows
    ranks = []

    # Sliding window so we don't exceed MAX_LENGTH
    prev_end = 0
    for begin_loc in range(0, n_tokens, STRIDE):
        end_loc = min(begin_loc + MAX_LENGTH, n_tokens)
        # We only compute predictions for the tokens after prev_end (to avoid overlap double-count)
        trg_len = end_loc - prev_end
        tok_slice = input_ids[:, begin_loc:end_loc]  # shape (1, L)
        L = tok_slice.size(1)

        # Get logits for all positions in slice
        outputs = model(tok_slice)
        # logits: (1, L, vocab_size), we predict token at pos i using context up to i-1
        logits = outputs.logits[:, :-1, :]  # positions we can predict
        targets = tok_slice[:, 1:]          # the next-token ids

        # If this is not the first window, mask out the overlap so we only evaluate last trg_len tokens
        # Number of positions to evaluate in this window:
        eval_len = min(trg_len, logits.size(1))
        if eval_len <= 0:
            prev_end = end_loc
            if end_loc == n_tokens:
                break
            continue

        # We'll evaluate the last eval_len positions (to avoid counting overlap)
        logits_eval  = logits[:, -eval_len:, :]   # (1, eval_len, V)
        targets_eval = targets[:, -eval_len:]     # (1, eval_len)

        # Compute rank for each position efficiently:
        # rank = 1 + number of vocab logits greater than the logit of the true token.
        # We'll do this by gathering the true-token logits and comparing.
        # true logits: (eval_len,)
        true_logits = logits_eval.gather(2, targets_eval.unsqueeze(-1)).squeeze(-1)  # (1, eval_len) -> (eval_len,)
        true_logits = true_logits.squeeze(0)  # shape (eval_len,)

        # Compare across vocab: for each position, count how many logits > true_logit
        # This is memory heavy if we do full compare; we can do a trick:
        # Sort descending and find insertion index of true logits. That's the rank-1.
        # Using torch.argsort on (eval_len, V) can be big; but GPT-2 vocab ~50k is manageable for short texts.
        # We'll vectorize by sorting once per position (loop over eval_len).
        vocab_size = logits_eval.size(-1)
        for i in range(eval_len):
            # sort descending for this position: (V,)
            pos_logits = logits_eval[0, i, :]
            # number of tokens with strictly higher logit than true token
            # Using (pos_logits > true_logits[i]).sum() avoids full sort:
            rank = int((pos_logits > true_logits[i]).sum().item()) + 1
            ranks.append(rank)

        prev_end = end_loc
        if end_loc == n_tokens:
            break

    n = len(ranks)
    if n == 0:
        return {
            "proportion_top10": 0.0, "proportion_top100": 0.0,
            "proportion_top1000": 0.0, "proportion_over1000": 0.0,
            "mean_rank": float("nan"), "median_rank": float("nan"),
            "rank_entropy": 0.0, "n_pred_tokens": 0
        }

    ranks_arr = np.array(ranks, dtype=np.int32)

    # Bin counts
    c_top10    = np.sum(ranks_arr <= 10)
    c_top100   = np.sum((ranks_arr > 10) & (ranks_arr <= 100))
    c_top1000  = np.sum((ranks_arr > 100) & (ranks_arr <= 1000))
    c_over1000 = np.sum(ranks_arr > 1000)

    # Proportions
    p_top10    = c_top10 / n
    p_top100   = c_top100 / n
    p_top1000  = c_top1000 / n
    p_over1000 = c_over1000 / n

    # Entropy over 4-bin distribution
    dist = np.array([p_top10, p_top100, p_top1000, p_over1000], dtype=float)
    eps = 1e-12
    entropy = -np.sum(dist * np.log2(dist + eps))

    return {
        "proportion_top10": float(p_top10),
        "proportion_top100": float(p_top100),
        "proportion_top1000": float(p_top1000),
        "proportion_over1000": float(p_over1000),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "rank_entropy": float(entropy),
        "n_pred_tokens": int(n),
    }

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

def prepare_tokenrank_rows(path: str, file_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    merged, resp_cols = merge_responses(df)

    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}")

    # Map ratings to numeric 1..5
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0, 5.0)

    # Compute token-rank features per row
    iterator = merged if not TQDM else tqdm(merged, desc=f"TokenRank {Path(path).name}", unit="row")
    feats = []
    for text in iterator:
        try:
            feats.append(token_rank_bins_for_text(text))
        except Exception:
            # If anything goes wrong, fill with safe defaults
            feats.append({
                "proportion_top10": 0.0, "proportion_top100": 0.0,
                "proportion_top1000": 0.0, "proportion_over1000": 1.0,
                "mean_rank": float("nan"), "median_rank": float("nan"),
                "rank_entropy": 0.0, "n_pred_tokens": 0
            })

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
# 1) Load & stack
all_rows = []
for j, fpath in enumerate(files):
    p = Path(fpath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    all_rows.append(prepare_tokenrank_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
feature_cols = [
    "proportion_top10", "proportion_top100", "proportion_top1000", "proportion_over1000",
    "mean_rank", "median_rank", "rank_entropy", "n_pred_tokens"
]
X = data[feature_cols].astype(float).fillna(0.0)
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

# 4) Evaluate
y_pred_test = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"\nMSE (Random Forest on token-rank features): {mse:.4f}")

# 5) Per-user averages across 5 runs
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
for col in ["proportion_top10","proportion_top100","proportion_top1000","proportion_over1000","rank_entropy"]:
    data_out[col] = data_out[col].round(4)
for col in ["mean_rank","median_rank","pred_rating"]:
    data_out[col] = data_out[col].round(2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)
print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
