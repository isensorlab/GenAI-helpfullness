# =========================================
# NLI Consistency / Factuality Proxy → Helpfulness (1–5)
# =========================================
# 1) For each of 5 CSVs:
#    - Merge QUESTION columns → one text per row (fallback "" if absent)
#    - Merge RESPONSE columns → one text per row
#    - Split responses into sentences
#    - Use an NLI model (roberta-base-mnli) on sentence pairs (within-response):
#        * mean_entail_prob, mean_contra_prob, max_contra_prob
#        * frac_entail (>0.5), frac_contra (>0.5), pairs_evaluated
#    - NLI between question and response (both directions):
#        * q_to_r_entail_prob, r_to_q_entail_prob
#    - Map Rating → 1..5
# 2) Stack (user, file) rows → 80/20 split
# 3) Train RandomForestRegressor → print MSE (test)
# 4) Compute per-user averages: true vs predicted across 5 runs

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

# NLI model (MNLI-trained)
NLI_MODEL_NAME = "cross-encoder/nli-roberta-base"  # good quality & size; can switch to "facebook/bart-large-mnli" for stronger (heavier)

# Sentence handling
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # simple, fast split
MAX_SENTENCES = 10      # cap sentences per response for speed (evaluate pairs among first K sentences)
PAIR_STRATEGY = "adjacent"  # "adjacent" or "all_pairs"
# - "adjacent": (s1→s2, s2→s3, ...) faster and usually enough
# - "all_pairs": all i<j pairs (O(K^2)), more thorough but slower

# Outputs
stacked_out_csv  = "stacked_nli_consistency_rows.csv"
per_user_out_csv = "per_user_helpfulness_nli_consistency_averages.csv"

# =========================================
# ============ NLI MODEL SETUP ============
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
nli_model.eval()

# roberta-base-mnli label order: ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']
# We'll softmax logits to probs and read indices 0,1,2 respectively.
def nli_probs(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    """Return (p_contra, p_neutral, p_entail) for a single pair."""
    if not premise.strip() or not hypothesis.strip():
        return 0.0, 1.0, 0.0  # treat empty as neutral
    with torch.no_grad():
        enc = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
        out = nli_model(**enc)
        probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()[0]
    # Order: 0=CONTRADICTION,1=NEUTRAL,2=ENTAILMENT
    return float(probs[0]), float(probs[1]), float(probs[2])

# Batch utility (optional; used internally)
def nli_probs_batch(premises: List[str], hypotheses: List[str], batch_size: int = 16) -> np.ndarray:
    """Return array of shape (N, 3): probs over [CONTRADICTION, NEUTRAL, ENTAILMENT]."""
    out_list = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hypotheses[i:i+batch_size]
        with torch.no_grad():
            enc = nli_tokenizer(batch_p, batch_h, return_tensors="pt", truncation=True, padding=True).to(device)
            logits = nli_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            out_list.append(probs)
    if not out_list:
        return np.zeros((0,3), dtype=float)
    return np.vstack(out_list)

# =========================================
# ============ HELPERS / LOADING ==========
# =========================================
def merge_cols(df: pd.DataFrame, pattern: re.Pattern, required: bool = False) -> pd.Series:
    cols = [c for c in df.columns if pattern.search(str(c))]
    if not cols:
        if required:
            raise ValueError(f"No columns matching pattern {pattern.pattern} in: {df.columns.tolist()}")
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # fast split; you can swap in a better seg if you already use spaCy in your pipeline
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if MAX_SENTENCES and len(sents) > MAX_SENTENCES:
        sents = sents[:MAX_SENTENCES]
    return sents

def nli_consistency_features(response_text: str, question_text: str) -> dict:
    """
    Build NLI features:
      - Within-response sentence-pair NLI stats (entail/contra)
      - Question↔Response entailment (both directions)
    """
    sents = split_sentences(response_text)

    # Build pairs to evaluate inside the response
    pairs = []
    if len(sents) >= 2:
        if PAIR_STRATEGY == "all_pairs":
            for i in range(len(sents)):
                for j in range(i+1, len(sents)):
                    # both directions (si->sj and sj->si) to catch contradict/entail
                    pairs.append((sents[i], sents[j]))
                    pairs.append((sents[j], sents[i]))
        else:  # adjacent (default)
            for i in range(len(sents)-1):
                pairs.append((sents[i], sents[i+1]))
                pairs.append((sents[i+1], sents[i]))

    # Run NLI on pairs
    if pairs:
        p_list = nli_probs_batch([p for p, h in pairs], [h for p, h in pairs], batch_size=16)
        p_contra = p_list[:, 0]
        p_entail = p_list[:, 2]

        mean_entail = float(np.mean(p_entail))
        mean_contra = float(np.mean(p_contra))
        max_contra  = float(np.max(p_contra))
        frac_entail = float(np.mean(p_entail > 0.5))
        frac_contra = float(np.mean(p_contra > 0.5))
        pairs_evaluated = int(len(pairs))
    else:
        mean_entail = mean_contra = max_contra = frac_entail = frac_contra = 0.0
        pairs_evaluated = 0

    # Question <-> Response NLI
    # r_to_q: does the response support/entail the question statement?
    # q_to_r: does the question imply the response (often low; still a helpful signal)
    if question_text and response_text:
        c_rq, n_rq, e_rq = nli_probs(response_text, question_text)
        c_qr, n_qr, e_qr = nli_probs(question_text, response_text)
        r_to_q_entail = float(e_rq)
        q_to_r_entail = float(e_qr)
        r_to_q_contra = float(c_rq)
        q_to_r_contra = float(c_qr)
    else:
        r_to_q_entail = q_to_r_entail = r_to_q_contra = q_to_r_contra = 0.0

    return {
        "nli_mean_entail": mean_entail,
        "nli_mean_contra": mean_contra,
        "nli_max_contra": max_contra,
        "nli_frac_entail_gt50": frac_entail,
        "nli_frac_contra_gt50": frac_contra,
        "nli_pairs_evaluated": float(pairs_evaluated),
        "nli_r_to_q_entail": r_to_q_entail,
        "nli_q_to_r_entail": q_to_r_entail,
        "nli_r_to_q_contra": r_to_q_contra,
        "nli_q_to_r_contra": q_to_r_contra,
        "n_sentences_resp": float(len(sents)),
    }

def prepare_nli_rows(path: str, file_id: int) -> pd.DataFrame:
    """
    Load one CSV, merge question/response, compute NLI features, map rating to 1..5.
    Returns DataFrame with: user_id, file_id, rating, (NLI features...)
    """
    df = pd.read_csv(path)

    responses = merge_cols(df, RESPONSE_COL_PATTERN, required=True)
    questions = merge_cols(df, QUESTION_COL_PATTERN, required=False)

    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found in {path}")

    # Ratings → numeric 1..5
    if df[rating_col].dtype == "O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0, 5.0)

    feats = []
    for resp, ques in zip(responses, questions):
        feats.append(nli_consistency_features(resp, ques))

    feats_df = pd.DataFrame(feats)
    out = pd.DataFrame({
        "user_id": np.arange(len(df)),
        "file_id": file_id,
        "rating": y.values
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
    all_rows.append(prepare_nli_rows(str(p), j))

data = pd.concat(all_rows, ignore_index=True)

# Sanity output
n_users = data["user_id"].nunique()
rows_per_file = data.groupby("file_id")["user_id"].nunique().to_dict()
print(f"Loaded rows: {data.shape[0]} | Unique users: {n_users} | Users per file: {rows_per_file}")

# 2) Train/test split on stacked rows
exclude_cols = {"user_id","file_id","rating"}
feature_cols = [c for c in data.columns if c not in exclude_cols]
X = (data[feature_cols]
     .astype(float)
     .replace([np.inf, -np.inf], np.nan)
     .fillna(0.0))
y = data["rating"].astype(float)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index, test_size=test_size, random_state=random_state
)

# 3) Train Random Forest
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
print(f"\nMSE (Random Forest on NLI consistency features): {mse:.4f}")

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
    if col == "pred_rating":
        data_out[col] = data_out[col].round(2)
    else:
        data_out[col] = data_out[col].round(4)

data_out.to_csv(stacked_out_csv, index=False)
per_user.to_csv(per_user_out_csv, index=False)
print(f"\nSaved stacked rows to: {stacked_out_csv}")
print(f"Saved per-user averages to: {per_user_out_csv}")
