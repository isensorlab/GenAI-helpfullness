# =========================================
# Entity-Grid Coherence → Helpfulness (1–5)
# =========================================
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -------- CONFIG --------
files: List[str] = [
    "Q3_Submission1_cleaned.csv",
    "Q3_Submission2_cleaned.csv",
    "Q3_Submission3_cleaned.csv",
    "Q3_Submission4_cleaned.csv",
    "Q3_Submission5_cleaned.csv",
]
rating_col = "Rating"
rating_map: Dict[str,int] = {
    "Not Helpful at All":1,"Slightly Helpful":2,"Neutral":3,"Helpful":4,"Very Helpful":5,
}
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)
random_state, test_size = 42, 0.20
stacked_out_csv  = "stacked_entitygrid_rows.csv"
per_user_out_csv = "per_user_helpfulness_entitygrid_averages.csv"

# -------- spaCy --------
nlp = spacy.load("en_core_web_sm", disable=["ner"])
# We only need POS, deps, syntax.

# -------- HELPERS --------
def merge_responses(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if RESPONSE_COL_PATTERN.search(str(c))]
    if not cols:
        raise ValueError(f"No response columns found in: {df.columns.tolist()}")
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

def entity_grid_features(text: str) -> dict:
    if not isinstance(text,str) or not text.strip():
        return {
            "unique_entities":0,"avg_entities_per_sent":0.0,
            "entity_continuity":0.0,"entity_introductions":0.0,
            "role_transitions":0.0,"grid_density":0.0
        }
    doc = nlp(text)
    sents = list(doc.sents)
    if len(sents)<2:  # too short for transitions
        ents = [t.lemma_.lower() for t in doc if t.pos_ in {"NOUN","PROPN","PRON"}]
        return {
            "unique_entities": len(set(ents)),
            "avg_entities_per_sent": float(len(ents))/max(len(sents),1),
            "entity_continuity": 0.0,"entity_introductions":1.0,
            "role_transitions":0.0,"grid_density":0.0
        }
    # Build entity grid
    entity_sents = defaultdict(list)  # ent -> list of roles per sent
    for si,sent in enumerate(sents):
        for tok in sent:
            if tok.pos_ in {"NOUN","PROPN","PRON"}:
                ent = tok.lemma_.lower()
                role = "O"  # default
                if tok.dep_ in {"nsubj","nsubjpass"}: role="S"
                elif tok.dep_ in {"dobj","pobj","iobj"}: role="O"
                else: role="X"
                entity_sents[ent].append((si,role))
    unique_entities = len(entity_sents)
    n_sents = len(sents)
    # Stats
    total_mentions = sum(len(v) for v in entity_sents.values())
    avg_entities_per_sent = total_mentions/max(n_sents,1)
    continuity = sum(1 for v in entity_sents.values() if len(v)>1)/max(unique_entities,1)
    introductions = sum(1 for v in entity_sents.values() if len(v)==1)/max(unique_entities,1)
    # Role transitions: count how often entity role changes
    transitions = []
    for ent, occs in entity_sents.items():
        roles = [r for _,r in sorted(occs)]
        transitions.extend([roles[i]!=roles[i-1] for i in range(1,len(roles))])
    role_transitions = np.mean(transitions) if transitions else 0.0
    grid_density = total_mentions/(n_sents*max(unique_entities,1))
    return {
        "unique_entities":unique_entities,
        "avg_entities_per_sent":avg_entities_per_sent,
        "entity_continuity":continuity,
        "entity_introductions":introductions,
        "role_transitions":role_transitions,
        "grid_density":grid_density
    }

def prepare_entitygrid_rows(path:str, file_id:int)->pd.DataFrame:
    df = pd.read_csv(path)
    responses = merge_responses(df)
    if rating_col not in df.columns:
        raise ValueError(f"No rating col in {path}")
    if df[rating_col].dtype=="O":
        y = df[rating_col].map(rating_map)
    else:
        y = df[rating_col].astype(float).clip(1.0,5.0)
    feats = [entity_grid_features(txt) for txt in responses]
    feats_df = pd.DataFrame(feats)
    out = pd.DataFrame({
        "user_id": np.arange(len(df)),"file_id": file_id,"rating": y
    })
    out = pd.concat([out,feats_df],axis=1)
    return out.loc[out["rating"].notna()].reset_index(drop=True)

# -------- PIPELINE --------
all_rows=[]
for j,fpath in enumerate(files):
    p=Path(fpath)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    all_rows.append(prepare_entitygrid_rows(str(p),j))
data=pd.concat(all_rows,ignore_index=True)
print(f"Loaded rows: {data.shape[0]} | Unique users: {data['user_id'].nunique()}")

# Train/test
exclude={"user_id","file_id","rating"}
feature_cols=[c for c in data.columns if c not in exclude]
X=data[feature_cols].astype(float).fillna(0.0)
y=data["rating"].astype(float)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size,random_state=random_state)
rf=RandomForestRegressor(n_estimators=500,random_state=random_state,n_jobs=-1)
rf.fit(Xtr,ytr)
pred=rf.predict(Xte)
print("MSE:", mean_squared_error(yte,pred))

# Per-user avg
data["pred_rating"]=np.clip(rf.predict(X),1.0,5.0)
per_user=(data.groupby("user_id",as_index=False)
          .agg(true_avg=("rating","mean"),pred_avg=("pred_rating","mean"),n_rows=("rating","size")))
per_user["true_avg"]=per_user["true_avg"].round(2)
per_user["pred_avg"]=per_user["pred_avg"].round(2)
print(per_user.head())

# Save
data.to_csv(stacked_out_csv,index=False)
per_user.to_csv(per_user_out_csv,index=False)
print("Saved:",stacked_out_csv,per_user_out_csv)
