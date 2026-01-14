# linear_mixed_effects_model.py
# -------------------------------------------------------------
# Linear Mixed-Effects model (statsmodels) for Helpfulness (1–5)
# - Fixed effects: standardized text features (edit FIXED_FEATURES)
# - Random effects: random intercept by user_id
# - Variance component: crossed effect for file_id
# - Reports FE-only and Conditional predictions (adds user RE if seen)
# - Saves predictions to from_scratch_models_out/lme_test_predictions.csv
# -------------------------------------------------------------

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# -----------------------------
# CONFIG
# -----------------------------
MASTER_CSV = Path("from_scratch_models_out/master_features_all.csv")  # change if needed
OUT_DIR    = Path("from_scratch_models_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Choose a solid, not-too-collinear set of fixed-effect features (standardized)
FIXED_FEATURES = [
    "cosine_sim",
    "nli_mean_entail",
    "content_vs_function_ratio",
    "pos_entropy",
    "verbs_per_sentence",
    "flesch_reading_ease",
    "ttr",
    "mtld",
    "avg_tree_depth",
    "entity_continuity",
    "vader_compound",
    "tb_subjectivity",
    "avg_logprob",
    "sentence_length",
]

# Random structure
GROUP_COL = "user_id"  # random intercept by user
VC_COL    = "file_id"  # variance component across files
VC_NAME   = "file"     # label used in the vc_formula


# -----------------------------
# LOAD DATA
# -----------------------------
if not MASTER_CSV.exists():
    raise FileNotFoundError(f"Master feature table not found: {MASTER_CSV.resolve()} "
                            f"(run your from-scratch pipeline first)")

df = pd.read_csv(MASTER_CSV)

needed = ["rating", GROUP_COL, VC_COL] + FIXED_FEATURES
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in {MASTER_CSV.name}: {missing}")

df = df[needed].dropna(subset=["rating"]).copy()

# Coerce numeric & drop remaining NAs in features
for c in FIXED_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=FIXED_FEATURES).reset_index(drop=True)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train = df.loc[train_idx].copy()
test  = df.loc[test_idx].copy()


# -----------------------------
# STANDARDIZE FIXED EFFECTS
# -----------------------------
scaler = StandardScaler()
train_z = pd.DataFrame(
    scaler.fit_transform(train[FIXED_FEATURES]),
    columns=[f"{c}_z" for c in FIXED_FEATURES],
    index=train.index,
)
test_z = pd.DataFrame(
    scaler.transform(test[FIXED_FEATURES]),
    columns=[f"{c}_z" for c in FIXED_FEATURES],
    index=test.index,
)

train = pd.concat([train[["rating", GROUP_COL, VC_COL]], train_z], axis=1)
test  = pd.concat([test[["rating", GROUP_COL, VC_COL]],  test_z],  axis=1)


# -----------------------------
# FORMULA & FIT
# -----------------------------
fixed_terms = " + ".join([f"{c}_z" for c in FIXED_FEATURES])
formula = f"rating ~ 1 + {fixed_terms}"

# Variance component for file_id: one indicator per file level
vc_formula = {VC_NAME: f"0 + C({VC_COL})"}

print("Fitting MixedLM with:")
print("  Fixed-effects formula:", formula)
print("  Random intercept by:", GROUP_COL)
print("  Variance component:", vc_formula)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    md = MixedLM.from_formula(
        formula=formula,
        groups=GROUP_COL,      # random intercept by user
        re_formula="1",        # random intercept only
        vc_formula=vc_formula, # variance component for files
        data=train,
    )
    mdf = md.fit(method="lbfgs", reml=True, maxiter=1000, disp=False)

print(mdf.summary())


# -----------------------------
# PREDICTION (FE-only & Conditional)
# -----------------------------
# FE-only (population-level): uses only the fixed part
yhat_fe = mdf.predict(test)

# Conditional: add the learned random intercept for users seen in training
def user_rand_intercept(u):
    re = mdf.random_effects.get(u)
    if re is None:
        return 0.0  # unseen user → no random intercept available
    # intercept is the first/only element since re_formula="1"
    return float(re[0])

u_adj = test[GROUP_COL].map(user_rand_intercept).values
yhat_cond = yhat_fe + u_adj

# Clamp to rating scale [1,5]
yhat_fe   = np.clip(yhat_fe,   1.0, 5.0)
yhat_cond = np.clip(yhat_cond, 1.0, 5.0)


# -----------------------------
# METRICS
# -----------------------------
y_true = test["rating"].astype(float).values

mse_fe  = mean_squared_error(y_true, yhat_fe)
mae_fe  = mean_absolute_error(y_true, yhat_fe)
mse_ci  = mean_squared_error(y_true, yhat_cond)
mae_ci  = mean_absolute_error(y_true, yhat_cond)

print("\n=== Test metrics (lower is better) ===")
print(f"Fixed-effects only:     MSE={mse_fe:.4f}  MAE={mae_fe:.4f}")
print(f"Conditional (if known): MSE={mse_ci:.4f}  MAE={mae_ci:.4f}")


# -----------------------------
# SAVE PREDICTIONS
# -----------------------------
out = test[[GROUP_COL, VC_COL, "rating"]].copy()
out["pred_fe_only"]     = np.round(yhat_fe, 2)
out["pred_conditional"] = np.round(yhat_cond, 2)

out_path = OUT_DIR / "lme_test_predictions.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved predictions → {out_path.resolve()}")
