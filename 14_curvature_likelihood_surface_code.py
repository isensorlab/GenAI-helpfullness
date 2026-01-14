# =========================================
# Curvature of Likelihood Surface (DetectGPT-style) → Helpfulness (1–5)
# =========================================
# Pipeline:
# 1) Merge response columns per row (per file)
# 2) Score original text with GPT-2 (avg log-likelihood per token)
# 3) Generate K perturbations via span-masked fill with RoBERTa (fill-mask)
# 4) Re-score perturbations with GPT-2
# 5) Features: ll_orig, mean_ll_pert, std_ll_pert, curvature_z
# 6) Train RandomForestRegressor → MSE; per-user averages

import re
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForMaskedLM, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---------- CONFIG ----------
files: List[str] = [
     "Q3_Submission1_cleaned.csv",
    "Q3_Submission2_cleaned.csv",
    "Q3_Submission3_cleaned.csv",
    "Q3_Submission4_cleaned.csv",
    "Q3_Submission5_cleaned.csv",
]
rating_col = "Rating"
rating_map: Dict[str,int] = {
    "Not Helpful at All":1, "Slightly Helpful":2, "Neutral":3, "Helpful":4, "Very Helpful":5
}
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)

# Causal LM for scoring (perplexity/logprob)
LM_NAME = "gpt2"  # "distilgpt2" for speed; "gpt2-medium" stronger/slower

# Masked LM for perturbations
MLM_NAME = "roberta-base"  # used with fill-mask pipeline
MASK_TOKEN = "<mask>"      # RoBERTa's mask token

# Curvature settings
NUM_PERTURB = 8            # number of perturbations per response (increase for stability)
MASK_PROB   = 0.15         # fraction of word tokens to mask
AVG_SPAN    = 3            # average span length (in words)
TOPK_FILL   = 5            # sample from top-k for each mask
SEED        = 42

# Train/test split
random_state = 42
test_size    = 0.20

# Outputs
stacked_out_csv  = "stacked_curvature_rows.csv"
per_user_out_csv = "per_user_helpfulness_curvature_averages.csv"

# ---------- SEED ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- DEVICES ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- MODELS ----------
# Scoring model (causal)
score_tok = AutoTokenizer.from_pretrained(LM_NAME)
if score_tok.pad_token is None:
    score_tok.pad_token = score_tok.eos_token
score_model = AutoModelForCausalLM.from_pretrained(LM_NAME).to(device)
score_model.eval()
MAX_CONTEXT = min(getattr(score_model.config, "n_positions", 1024), 1024)
STRIDE = 512

# Masked LM (perturbation generator)
mlm_tok = AutoTokenizer.from_pretrained(MLM_NAME)
mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_NAME).to(device)
mlm_model.eval()
fill_mask = pipeline("fill-mask", model=mlm_model, tokenizer=mlm_tok, device=0 if device=="cuda" else -1, top_k=TOPK_FILL)

# ---------- HELPERS ----------
def merge_responses(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not cols:
        raise ValueError(
            "No response columns found. Adjust RESPONSE_COL_PATTERN "
            f"(columns present: {df.columns.tolist()})"
        )
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

@torch.no_grad()
def avg_loglikelihood(text: str) -> float:
    """
    Average token log-likelihood under the causal LM with sliding window.
    Returns negative values (log probs).
    """
    if not isinstance(text, str) or not text.strip():
        return float("-inf")

    enc = score_tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    n = input_ids.size(1)
    if n == 0:
        return float("-inf")

    if n <= MAX_CONTEXT:
        out = score_model(input_ids, labels=input_ids)
        # loss = mean NLL per token
        return -float(out.loss.item())

    nlls = []
    prev_end = 0
    for begin in range(0, n, STRIDE):
        end = min(begin + MAX_CONTEXT, n)
        trg_len = end - prev_end
        ids_slice = input_ids[:, begin:end]
        target_ids = ids_slice.clone()
        if target_ids.size(1) > trg_len:
            target_ids[:, :-trg_len] = -100
        out = score_model(ids_slice, labels=target_ids)
        nlls.append(out.loss.item() * trg_len)
        prev_end = end
        if end == n:
            break
    total_nll = float(sum(nlls))
    # exact count of predicted tokens
    total_pred_tokens = float(sum(
        min(MAX_CONTEXT, n - i) if i == 0 else min(STRIDE, n - i)
        for i in range(0, n, STRIDE)
    ))
    return -(total_nll / max(total_pred_tokens, 1.0))

def split_words(text: str) -> List[str]:
    # simple whitespace tokenizer (word-level span masking)
    return [w for w in re.findall(r"\S+", text)]

def sample_spans(n_tokens: int, mask_prob: float, avg_span: int) -> List[Tuple[int, int]]:
    """
    Sample non-overlapping spans over token indices [0..n-1].
    Each span length ~ geometric with mean = avg_span.
    Returns list of (start, end) with end exclusive.
    """
    if n_tokens == 0 or mask_prob <= 0:
        return []
    target = max(1, int(round(n_tokens * mask_prob)))
    spans = []
    covered = set()
    i = 0
    while len(covered) < target and i < n_tokens:
        # geometric length with mean avg_span => p = 1/avg_span
        span_len = max(1, np.random.geometric(1.0 / max(avg_span, 1)))
        start = np.random.randint(0, n_tokens)
        end = min(start + span_len, n_tokens)
        new_idxs = set(range(start, end))
        if not (covered & new_idxs):
            spans.append((start, end))
            covered |= new_idxs
        i += 1
    # sort by start
    spans.sort()
    return spans

def build_masked_text(words: List[str], spans: List[Tuple[int,int]]) -> str:
    """
    Replace each span with one or more <mask> tokens (one per word in the span).
    """
    if not spans:
        return " ".join(words)
    out = []
    span_iter = iter(spans)
    cur_span = next(span_iter, None)
    i = 0
    while i < len(words):
        if cur_span and i == cur_span[0]:
            length = cur_span[1] - cur_span[0]
            out.extend([MASK_TOKEN] * length)
            i = cur_span[1]
            cur_span = next(span_iter, None)
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)

def fill_masks_once(text_with_masks: str) -> str:
    """
    Fill all <mask> tokens in the text. RoBERTa can handle multiple masks,
    but the pipeline returns candidates per mask independently.
    We fill masks left-to-right, sampling from top-k for diversity.
    """
    if MASK_TOKEN not in text_with_masks:
        return text_with_masks
    # Iteratively replace each mask token
    result = text_with_masks
    # Safety: cap passes
    for _ in range(256):
        if MASK_TOKEN not in result:
            break
        # pipeline needs exactly one mask at a time; replace first mask
        first = result.replace(MASK_TOKEN, fill_mask.tokenizer.mask_token, 1)
        preds = fill_mask(first)
        # If only a single mask present, preds is list[dict]; if multiple, pipeline returns list[list[dict]]
        # We ensured one mask in 'first', so preds is list[dict].
        if isinstance(preds, list) and preds and isinstance(preds[0], dict):
            choice = random.choice(preds[:TOPK_FILL]) if len(preds) >= TOPK_FILL else preds[0]
            token_str = choice["token_str"].strip()
            # Replace the *first* mask with the sampled token_str
            result = first.replace(fill_mask.tokenizer.mask_token, token_str, 1)
        else:
            # Fallback: remove mask
            result = first.replace(fill_mask.tokenizer.mask_token, "", 1)
    return result

def make_perturbations(text: str, k: int = NUM_PERTURB) -> List[str]:
    """
    Create K perturbed paraphrases by span-masking + MLM fill.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    words = split_words(text)
    perturbed = []
    for _ in range(k):
        spans = sample_spans(len(words), MASK_PROB, AVG_SPAN)
        masked = build_masked_text(words, spans)
        filled = fill_masks_once(masked)
        perturbed.append(filled)
    return perturbed

def curvature_features(text: str) -> dict:
    """
    Compute DetectGPT-style curvature proxy using GPT-2 scoring and RoBERTa perturbations.
    Returns:
      - ll_orig: avg log-likelihood of original text
      - mean_ll_pert, std_ll_pert
      - curvature_z
      - n_perturb_used
    """
    ll_orig = avg_loglikelihood(text)
    perts = make_perturbations(text, NUM_PERTURB)
    ll_perts = []
    for ptxt in perts:
        try:
            ll_perts.append(avg_loglikelihood(ptxt))
        except Exception:
            continue
    if not ll_perts:
        return {
            "ll_orig": ll_orig, "mean_ll_pert": ll_orig, "std_ll_pert": 0.0,
            "curvature_z": 0.0, "n_perturb_used": 0
        }
    mean_p = float(np.mean(ll_perts))
    std_p  = float(np.std(ll_perts))
    z = (ll_orig - mean_p) / (std_p + 1e-8)
    return {
        "ll_orig": ll_orig,
        "mean_ll_pert": mean_p,
        "std_ll_pert": std_p,
        "curvature_z": float(z),
        "n_perturb_used": int(len(ll_perts))
    }

def prepare_curvature_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge response columns, compute curvature features, map rating → 1..5.
    """
    df = pd.read_csv(path)
    responses = merge_responses(df)

    # Map rating
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}")
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0, 5.0)

    feats = []
    # optional progress
    try:
        from tqdm import tqdm
        iterator = tqdm(responses, desc=f"Curvature {Path(path).name}", unit="row")
    except Exception:
        iterator = responses
    for txt in iterator:
        try:
            feats.append(curvature_features(txt))
        except Exception:
            feats.append({
                "ll_orig": float("-inf"),
                "mean_ll_pert": float("-inf"),
                "std_ll_pert": 0.0,
                "curvature_z": 0.0,
                "n_perturb_used": 0
            })

    feats_df = pd.DataFrame(feats)
    out = pd.DataFrame({
        "user_id": np.arange(len(df)),
        "file_id": file_id,
        "rating": y
    })
    out = pd.concat([out, feats_df], axis=1)
    return out.loc[out["rating"].notna()].reset_index(drop=True)

# ---------- PIPELINE ----------
all_rows = []
for j, fpath in enumerate(files):
    p = Path(fpath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    all_rows.append(prepare_curvature_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity
print(f"Loaded rows: {data.shape[0]} | Unique users: {data['user_id'].nunique()}")

# Train/test
exclude = {"user_id","file_id","rating"}
feature_cols = [c for c in data.columns if c not in exclude]
# Clean infinities
X = (data[feature_cols]
     .replace([np.inf, -np.inf], np.nan)
     .fillna(0.0)
     .astype(float))
y = data["rating"].astype(float)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

rf = RandomForestRegressor(
    n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=random_state, n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE (Random Forest on curvature features): {mse:.4f}")

# Per-user averages
data["pred_rating"] = np.clip(rf.predict(X), 1.0, 5.0)
per_user = (
    data.groupby("user_id", as_index=False)
        .agg(true_avg_rating=("rating","mean"),
             pred_avg_rating=("pred_rating","mean"),
             n_rows=("rating","size"))
)
per_user["true_avg_rating"] = per_user["true_avg_rating"].round(2)
per_user["pred_avg_rating"] = per_user["pred_avg_rating"].round(2)

print("\nPer-user helpfulness averages (head):")
print(per_user.head())

# Save
data_out = data.copy()
for c in ["ll_orig","mean_ll_pert","std_ll_pert","curvature_z","pred_rating"]:
    data_out[c] = data_out[c].round(4 if c != "pred_rating" else 2)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)
print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
