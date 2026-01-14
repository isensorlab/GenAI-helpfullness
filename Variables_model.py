
import os
import re
import math
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# NLP core
import spacy

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Transformers & sentence embeddings
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForMaskedLM, AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer

# Progress bar
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# -------------------------
# CONFIG
# -------------------------
FILES: List[str] = [
"Q4_Submission1_cleaned.csv","Q4_Submission2_cleaned.csv","Q4_Submission3_cleaned.csv","Q4_Submission4_cleaned.csv","Q4_Submission5_cleaned.csv",
]

RATING_COL = "Rating"
RATING_MAP: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

# Column detectors
RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)
QUESTION_COL_PATTERN = re.compile(r"(question|prompt|query)", re.IGNORECASE)

# Train/test
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Heavy feature switches (set False to speed up)
USE_EMBEDDINGS = True
USE_NLI       = True
USE_PERPLEX   = True
USE_TOKENRANK = True
USE_CURVATURE = True

# You can globally disable heavy models quickly:
USE_HEAVY_FEATURES = True
if not USE_HEAVY_FEATURES:
    USE_EMBEDDINGS = USE_NLI = USE_PERPLEX = USE_TOKENRANK = USE_CURVATURE = False

# Model choices / limits
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME   = "cross-encoder/nli-roberta-base"   # compact and public
LM_NAME          = "distilgpt2"                       # faster than gpt2
MLM_NAME         = "roberta-base"                     # for perturbations
MASK_TOKEN       = "<mask>"
MAX_CONTEXT      = 1024
STRIDE           = 512

# Curvature params
NUM_PERTURB   = 6       # increase to 12–24 for more stability
MASK_PROB     = 0.15
AVG_SPAN      = 3
TOPK_FILL     = 5

# NLI sentence pairing
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MAX_SENTENCES_FOR_NLI = 8
PAIR_STRATEGY = "adjacent"   # or "all_pairs"

# Output directory
OUT_DIR = Path("from_scratch_models_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Silence HF warnings if noisy
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# -------------------------
# UTILITIES
# -------------------------
def safe_div(a, b, default=0.0):
    return (a / b) if (b and b != 0) else default

def merge_cols(df: pd.DataFrame, pattern: re.Pattern, fallback_empty: bool = False) -> pd.Series:
    cols = [c for c in df.columns if pattern.search(str(c))]
    if not cols:
        if fallback_empty:
            return pd.Series([""] * len(df), index=df.index)
        raise ValueError(f"No columns matching /{pattern.pattern}/ in: {df.columns.tolist()}")
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip()

def map_rating(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        return series.map(RATING_MAP)
    return series.astype(float).clip(1.0, 5.0)

def sentence_split(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
def tokenize_words(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip(): return []
    return WORD_RE.findall(text)

# -------------------------
# spaCy pipeline (POS/DEP/NER)
# -------------------------
print("Loading spaCy en_core_web_sm ...")
nlp = spacy.load("en_core_web_sm")  # keep NER enabled (we need it)
# nlp.max_length = 2_000_000  # uncomment for very long rows

# -------------------------
# Sentiment analyzers
# -------------------------
try:
    _vader = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon")
    _vader = SentimentIntensityAnalyzer()

# -------------------------
# SentenceTransformer (Embeddings)
# -------------------------
if USE_EMBEDDINGS:
    print("Loading SentenceTransformer embeddings model ...")
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)

# -------------------------
# Causal LM for perplexity / token ranks
# -------------------------
if USE_PERPLEX or USE_TOKENRANK or USE_CURVATURE:
    print("Loading causal LM for scoring ...")
    score_tok = AutoTokenizer.from_pretrained(LM_NAME)
    if score_tok.pad_token is None:
        score_tok.pad_token = score_tok.eos_token
    score_model = AutoModelForCausalLM.from_pretrained(LM_NAME)
    score_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    score_model.to(device)
    try:
        MAX_CONTEXT = min(getattr(score_model.config, "n_positions", MAX_CONTEXT), MAX_CONTEXT)
    except Exception:
        pass

# -------------------------
# Masked LM for curvature perturbations
# -------------------------
if USE_CURVATURE:
    print("Loading masked LM for perturbations ...")
    mlm_tok   = AutoTokenizer.from_pretrained(MLM_NAME)
    mlm_model = AutoModelForMaskedLM.from_pretrained(MLM_NAME).to(device if 'device' in locals() else "cpu")
    mlm_model.eval()
    fill_mask = pipeline("fill-mask", model=mlm_model, tokenizer=mlm_tok,
                         device=0 if torch.cuda.is_available() else -1, top_k=TOPK_FILL)

# -------------------------
# NLI model
# -------------------------
if USE_NLI:
    print("Loading NLI model ...")
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    nli_model.eval()
    nli_model.to(device if 'device' in locals() else "cpu")

    # dynamic label indices (to avoid hardcoding)
    id2label = {i: lab.upper() for i, lab in nli_model.config.id2label.items()}
    def _find_label_index(substrs: List[str]) -> int:
        for i, lab in id2label.items():
            if any(s in lab for s in substrs): return i
        raise ValueError(f"NLI label not found among {id2label}")
    IDX_ENTAIL  = _find_label_index(["ENTAIL"])
    IDX_CONTRA  = _find_label_index(["CONTRAD"])
    IDX_NEUTRAL = _find_label_index(["NEUTRAL","NEU"])

# ============================================================
# FEATURE EXTRACTORS
# ============================================================

# --- 1) Sentence length & counts
def feat_sentence_length(text: str) -> dict:
    sents = sentence_split(text)
    toks = tokenize_words(text)
    return {
        "n_sentences": len(sents),
        "n_tokens": len(toks),
        "sentence_length": len(toks)  # alias
    }

# --- 2) Readability
VOWELS = "aeiouy"
def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 0
    if w.endswith("e") and not w.endswith("le"): w = w[:-1]
    if not w: return 1
    groups, prev = 0, False
    for ch in w:
        isv = ch in VOWELS
        if isv and not prev: groups += 1
        prev = isv
    if w.endswith("le") and len(w) > 2 and w[-3] not in VOWELS:
        groups += 1
    return max(groups, 1)

def feat_readability(text: str) -> dict:
    sents = sentence_split(text)
    words = [w for w in tokenize_words(text) if re.search(r"[A-Za-z]", w)]
    n_sent = max(len(sents), 1)
    n_words = len(words)
    n_chars = sum(len(w) for w in words)
    syll   = sum(count_syllables(w) for w in words) if n_words>0 else 0
    polys  = sum(1 for w in words if count_syllables(w) >= 3)
    wps = safe_div(n_words, n_sent)
    spw = safe_div(syll, n_words)
    cpw = safe_div(n_chars, n_words)
    pr  = safe_div(polys, n_words)
    flesch = 206.835 - 1.015*wps - 84.6*spw
    fkgl   = 0.39*wps + 11.8*spw - 15.59
    fog    = 0.4*(wps + 100*pr)
    ari    = 4.71*cpw + 0.5*wps - 21.43
    L = cpw*100.0
    S = safe_div(n_sent, n_words)*100.0
    coleman = 0.0588*L - 0.296*S - 15.8
    smog = 0.0
    if polys>0 and n_sent>0:
        smog = 1.0430*math.sqrt(30.0*(polys/n_sent)) + 3.1291
    return {
        "n_sentences_r": len(sents),
        "n_words_r": n_words,
        "n_chars_r": n_chars,
        "syllables": syll,
        "polysyllables": polys,
        "words_per_sentence": round(wps,4),
        "syllables_per_word": round(spw,4),
        "chars_per_word": round(cpw,4),
        "polysyllable_rate": round(pr,4),
        "flesch_reading_ease": round(flesch,4),
        "fk_grade_level": round(fkgl,4),
        "gunning_fog": round(fog,4),
        "smog_index": round(smog,4),
        "ari": round(ari,4),
        "coleman_liau": round(coleman,4),
    }

# --- 3) Lexical richness
def safe_log(x: float) -> float: return math.log(x) if x>0 else 0.0
def mtld(tokens: List[str], ttr_threshold: float = 0.72) -> float:
    def mtld_seq(seq):
        if len(seq)<10: return 0.0
        factors, types, count = 0, set(), 0
        for w in seq:
            count += 1; types.add(w)
            if len(types)/count <= ttr_threshold:
                factors += 1; types=set(); count=0
        if count>0:
            factors += (1 - (len(types)/max(count,1))) / (1 - ttr_threshold)
        return len(seq)/factors if factors>0 else 0.0
    if not tokens: return 0.0
    f = mtld_seq(tokens); b = mtld_seq(list(reversed(tokens)))
    return (f+b)/2 if (f>0 and b>0) else max(f,b)

def feat_lexical_richness(text: str) -> dict:
    toks = [w.lower() for w in tokenize_words(text)]
    N = len(toks)
    if N==0:
        return {k:0.0 for k in [
            "tokens","types","ttr","rttr","cttr","herdan_c","maas_a2",
            "yules_k","mtld","hapax_ratio","dis_legomena_ratio","avg_word_len"
        ]}
    freqs = Counter(toks); V=len(freqs)
    hapax = sum(1 for f in freqs.values() if f==1)
    dis2  = sum(1 for f in freqs.values() if f==2)
    avg_len = np.mean([len(w) for w in toks])
    # yule's K
    Vi = Counter(freqs.values()); sum_i2Vi = sum((i*i)*Vi for i,Vi in Vi.items())
    K = 1e4 * ((sum_i2Vi - N)/(N*N))
    lnN = safe_log(N)
    Maas = (lnN - safe_log(V))/(lnN*lnN) if (lnN>0 and V>1) else 0.0
    return {
        "tokens": N, "types": V,
        "ttr": V/N,
        "rttr": V/math.sqrt(N),
        "cttr": V/math.sqrt(2*N),
        "herdan_c": (safe_log(V)/lnN) if (V>1 and lnN>0) else 0.0,
        "maas_a2": Maas,
        "yules_k": K,
        "mtld": mtld(toks),
        "hapax_ratio": hapax/max(V,1),
        "dis_legomena_ratio": dis2/max(V,1),
        "avg_word_len": float(avg_len),
    }

# --- 4) POS distribution + entropy
POS_TAGS = [
    "NOUN","PROPN","PRON","VERB","AUX","ADJ","ADV","ADP","DET",
    "CCONJ","SCONJ","PART","NUM","INTJ","SYM","X","PUNCT"
]
CONTENT_POS = {"NOUN","PROPN","VERB","ADJ","ADV"}
FUNCTION_POS = {"DET","ADP","AUX","PRON","CCONJ","SCONJ","PART"}

def feat_pos(text: str) -> dict:
    doc = nlp(text)
    toks = [t for t in doc if not t.is_space]
    n_tok = len(toks)
    pos_counts = Counter(t.pos_ for t in toks)
    props = {f"pos_prop_{p}": (pos_counts.get(p,0)/n_tok if n_tok>0 else 0.0) for p in POS_TAGS}
    content = sum(pos_counts.get(p,0) for p in CONTENT_POS)
    function = sum(pos_counts.get(p,0) for p in FUNCTION_POS)
    pron_ratio = pos_counts.get("PRON",0)/max(n_tok,1)
    verbs_per_sent = (pos_counts.get("VERB",0)+pos_counts.get("AUX",0))/max(len(list(doc.sents)) or 1, 1)
    # entropy
    if n_tok>0 and pos_counts:
        dist = np.array([c/n_tok for c in pos_counts.values()], dtype=float)
        pos_entropy = -float(np.sum(dist*np.log2(dist+1e-12)))
    else:
        pos_entropy = 0.0
    return {
        "token_count": n_tok,
        "sentence_count": len(list(doc.sents)),
        **props,
        "content_vs_function_ratio": (content/max(function,1)) if function>0 else float(content>0),
        "pronoun_ratio": pron_ratio,
        "verbs_per_sentence": verbs_per_sent,
        "pos_entropy": pos_entropy,
    }

# --- 5) Dependency tree depth
def sentence_tree_depth(sent) -> int:
    cache = {}
    def depth(tok):
        if tok in cache: return cache[tok]
        d, cur = 0, tok
        while cur.head != cur:
            d += 1; cur = cur.head
        cache[tok] = d; return d
    return max((depth(t) for t in sent), default=0)

def sentence_dep_distance(sent) -> float:
    d = [abs(t.i - t.head.i) for t in sent if t.head != t]
    return float(np.mean(d)) if d else 0.0

def sentence_root_children(sent) -> int:
    for t in sent:
        if t.head == t:
            return len(list(t.children))
    return 0

def feat_dependency(text: str) -> dict:
    doc = nlp(text)
    depths, dists, roots, n_sent, n_tok = [], [], [], 0, 0
    for sent in doc.sents:
        n_sent += 1
        n_tok += sum(1 for t in sent if not t.is_space)
        depths.append(sentence_tree_depth(sent))
        dists.append(sentence_dep_distance(sent))
        roots.append(sentence_root_children(sent))
    if n_sent==0:
        return {"avg_tree_depth":0.0,"max_tree_depth":0.0,"std_tree_depth":0.0,
                "avg_dep_distance":0.0,"root_children_mean":0.0,"n_sentences":0,"n_tokens":0}
    return {
        "avg_tree_depth": float(np.mean(depths)),
        "max_tree_depth": float(np.max(depths)),
        "std_tree_depth": float(np.std(depths)),
        "avg_dep_distance": float(np.mean(dists)),
        "root_children_mean": float(np.mean(roots)),
        "n_sentences": n_sent,
        "n_tokens": n_tok
    }

# --- 6) NER features
NER_LABELS = [
    "PERSON","ORG","GPE","LOC","NORP","FAC","EVENT","WORK_OF_ART","LAW","LANGUAGE","PRODUCT",
    "DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
]
def feat_ner(text: str) -> dict:
    doc = nlp(text)
    toks = [t for t in doc if not t.is_space]
    ents = list(doc.ents)
    token_count = len(toks)
    entity_count = len(ents)
    density = entity_count/max(token_count,1)
    from collections import Counter as C
    lc = C(e.label_ for e in ents)
    unique_types = len(lc)
    if entity_count>0:
        probs = np.array([c/entity_count for c in lc.values()], dtype=float)
        entropy = -float(np.sum(probs*np.log2(probs+1e-12)))
        avg_len = float(np.mean([len(e) for e in ents]))
    else:
        entropy = 0.0; avg_len = 0.0
    row = {
        "token_count_ner": token_count,
        "entity_count": entity_count,
        "entity_density": density,
        "unique_entity_types": unique_types,
        "entity_label_entropy": entropy,
        "avg_entity_length_tokens": avg_len
    }
    for lbl in NER_LABELS:
        row[f"ent_count_{lbl}"] = float(lc.get(lbl,0))
    return row

# --- 7) Sentiment & subjectivity
def feat_sentiment(text: str) -> dict:
    if not isinstance(text,str) or not text.strip():
        return {"vader_compound":0.0,"vader_pos":0.0,"vader_neu":0.0,"vader_neg":0.0,
                "tb_polarity":0.0,"tb_subjectivity":0.0}
    vs = _vader.polarity_scores(text)
    tb = TextBlob(text).sentiment
    return {
        "vader_compound": float(vs.get("compound",0.0)),
        "vader_pos": float(vs.get("pos",0.0)),
        "vader_neu": float(vs.get("neu",0.0)),
        "vader_neg": float(vs.get("neg",0.0)),
        "tb_polarity": float(tb.polarity),
        "tb_subjectivity": float(tb.subjectivity),
    }

# --- 8) Discourse markers (PDTB-ish)
TEMPORAL = ["after","afterward","afterwards","before","earlier","later","meantime","meanwhile","subsequently",
            "then","next","finally","first","second","third","fourth","ultimately","previously","simultaneously","soon",
            "eventually"]
COMPARISON = ["however","but","nevertheless","nonetheless","yet","though","although","whereas","while","instead",
              "in contrast","on the other hand","rather"]
CONTINGENCY = ["because","since","as","so","therefore","thus","hence","consequently","accordingly","as a result",
               "resultantly","due to","owing to"]
EXPANSION = ["and","also","moreover","furthermore","in addition","besides","additionally","namely","for example",
             "for instance","that is","in other words"]

def _patlist(words: List[str]) -> List[re.Pattern]:
    pats=[]
    for w in words:
        w_esc = re.escape(w).replace(r"\ ", r"\s+")
        pats.append(re.compile(rf"\b{w_esc}\b", re.IGNORECASE))
    return pats

PAT_TEMPORAL   = _patlist(TEMPORAL)
PAT_COMPARISON = _patlist(COMPARISON)
PAT_CAUSE      = _patlist(CONTINGENCY)
PAT_EXPANSION  = _patlist(EXPANSION)

ENUM_WORDS = ["first","second","third","fourth","fifth","sixth"]

def _find_spans(text: str, pats: List[re.Pattern]) -> List[Tuple[int,int]]:
    spans=[]
    for p in pats:
        spans += [m.span() for m in p.finditer(text)]
    return spans

def feat_discourse(text: str) -> dict:
    if not isinstance(text,str) or not text.strip():
        return {k:0.0 for k in [
            "disc_token_count","total_connectives","connective_density_per_100w","unique_connectives",
            "connective_entropy","count_temporal","count_comparison","count_cause","count_expansion",
            "density_temporal","density_comparison","density_cause","density_expansion",
            "enumeration_count","avg_connective_position"
        ]}
    toks = tokenize_words(text); n = len(toks)
    st = _find_spans(text, PAT_TEMPORAL)
    sc = _find_spans(text, PAT_COMPARISON)
    sk = _find_spans(text, PAT_CAUSE)
    se = _find_spans(text, PAT_EXPANSION)
    ct_t, ct_c, ct_k, ct_e = len(st), len(sc), len(sk), len(se)
    total = ct_t+ct_c+ct_k+ct_e
    per100 = 100.0/max(n,1)
    uniq = set()
    for vocab in [TEMPORAL,COMPARISON,CONTINGENCY,EXPANSION]:
        for w in vocab:
            if re.search(rf"\b{re.escape(w).replace(r'\ ', r'\s+')}\b", text, flags=re.IGNORECASE):
                uniq.add(w.lower())
    unique_connectives = len(uniq)
    if total>0:
        dist = np.array([ct_t, ct_c, ct_k, ct_e], dtype=float)/total
        entropy = -float(np.sum(dist*np.log2(dist+1e-12)))
        char_positions = [s for s,_ in (st+sc+sk+se)]
        avg_pos_norm = float(np.mean([p/max(len(text),1) for p in char_positions]))
    else:
        entropy = 0.0; avg_pos_norm=0.0
    enum_count = sum(t.lower() in ENUM_WORDS for t in toks)
    return {
        "disc_token_count": n,
        "total_connectives": total,
        "connective_density_per_100w": total*per100,
        "unique_connectives": unique_connectives,
        "connective_entropy": entropy,
        "count_temporal": ct_t, "count_comparison": ct_c, "count_cause": ct_k, "count_expansion": ct_e,
        "density_temporal": ct_t*per100, "density_comparison": ct_c*per100, "density_cause": ct_k*per100, "density_expansion": ct_e*per100,
        "enumeration_count": enum_count,
        "avg_connective_position": round(avg_pos_norm,4)
    }

# --- 9) Entity-grid coherence
def feat_entity_grid(text: str) -> dict:
    doc = nlp(text)
    sents = list(doc.sents)
    if len(sents)<2:
        ents = [t.lemma_.lower() for t in doc if t.pos_ in {"NOUN","PROPN","PRON"}]
        return {
            "unique_entities": len(set(ents)),
            "avg_entities_per_sent": float(len(ents))/max(len(sents),1),
            "entity_continuity": 0.0,
            "entity_introductions": 1.0 if len(ents)>0 else 0.0,
            "role_transitions": 0.0,
            "grid_density": 0.0
        }
    entity_sents = defaultdict(list)
    for si, sent in enumerate(sents):
        for tok in sent:
            if tok.pos_ in {"NOUN","PROPN","PRON"}:
                ent = tok.lemma_.lower()
                if tok.dep_ in {"nsubj","nsubjpass"}: role="S"
                elif tok.dep_ in {"dobj","pobj","iobj"}: role="O"
                else: role="X"
                entity_sents[ent].append((si, role))
    unique = len(entity_sents)
    total_mentions = sum(len(v) for v in entity_sents.values())
    continuity = sum(1 for v in entity_sents.values() if len(v)>1)/max(unique,1)
    introductions = sum(1 for v in entity_sents.values() if len(v)==1)/max(unique,1)
    transitions=[]
    for _, occs in entity_sents.items():
        roles = [r for _,r in sorted(occs)]
        transitions += [roles[i]!=roles[i-1] for i in range(1,len(roles))]
    role_trans = float(np.mean(transitions)) if transitions else 0.0
    grid_density = total_mentions/(len(sents)*max(unique,1))
    return {
        "unique_entities": unique,
        "avg_entities_per_sent": total_mentions/max(len(sents),1),
        "entity_continuity": continuity,
        "entity_introductions": introductions,
        "role_transitions": role_trans,
        "grid_density": grid_density
    }

# --- 10) Semantic embeddings (Q↔R)
def feat_embeddings(question: str, response: str) -> dict:
    if not USE_EMBEDDINGS:
        return {"cosine_sim":0.0,"q_norm":0.0,"r_norm":0.0,"norm_gap":0.0}
    q = [question]; r=[response]
    q_emb = emb_model.encode(q, convert_to_numpy=True, show_progress_bar=False)
    r_emb = emb_model.encode(r, convert_to_numpy=True, show_progress_bar=False)
    qn = np.linalg.norm(q_emb, axis=1); rn = np.linalg.norm(r_emb, axis=1)
    cos = float(np.sum((q_emb/np.clip(qn[:,None],1e-12,None)) * (r_emb/np.clip(rn[:,None],1e-12,None))))
    return {
        "cosine_sim": cos,
        "q_norm": float(qn[0]),
        "r_norm": float(rn[0]),
        "norm_gap": float(abs(qn[0]-rn[0]))
    }

# --- 11) Perplexity / avg log-prob (causal LM) + 12) Token-rank (GLTR-like)
@torch.no_grad()
def _score_chunks(input_ids: torch.Tensor, model, stride: int, max_ctx: int) -> Tuple[float,int, List[np.ndarray]]:
    n = input_ids.size(1)
    total_nll = 0.0
    total_pred = 0
    all_logits = []

    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + max_ctx, n)
        trg_len = end - prev_end
        inp = input_ids[:, begin:end].to(model.device)
        target = inp.clone()
        if target.size(1) > trg_len:
            target[:, :-trg_len] = -100
        out = model(inp, labels=target)
        loss = out.loss.item()
        total_nll += loss * trg_len
        total_pred += trg_len
        logits = out.logits[0, -trg_len:, :].detach().cpu().numpy()
        all_logits.append(logits)
        prev_end = end
        if end==n:
            break
    return total_nll, total_pred, all_logits

def feat_perplex_and_tokenrank(text: str) -> dict:
    if not USE_PERPLEX and not USE_TOKENRANK:
        return {"avg_logprob":0.0,"ppl":0.0,
                "proportion_top10":0.0,"proportion_top100":0.0,"proportion_top1000":0.0,
                "proportion_over1000":0.0,"mean_rank":0.0,"median_rank":0.0,
                "rank_entropy":0.0,"n_pred_tokens":0.0}

    if not isinstance(text,str) or not text.strip():
        return {"avg_logprob":0.0,"ppl":0.0,
                "proportion_top10":0.0,"proportion_top100":0.0,"proportion_top1000":0.0,
                "proportion_over1000":0.0,"mean_rank":0.0,"median_rank":0.0,
                "rank_entropy":0.0,"n_pred_tokens":0.0}

    enc = score_tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    total_nll, total_pred, logits_list = _score_chunks(
        input_ids, score_model, STRIDE, MAX_CONTEXT
    )
    if total_pred==0:
        return {"avg_logprob":0.0,"ppl":0.0,
                "proportion_top10":0.0,"proportion_top100":0.0,"proportion_top1000":0.0,
                "proportion_over1000":0.0,"mean_rank":0.0,"median_rank":0.0,
                "rank_entropy":0.0,"n_pred_tokens":0.0}

    avg_nll = total_nll / total_pred
    avg_logprob = -avg_nll
    ppl = float(math.exp(avg_nll))

    # Token rank stats (GLTR-like)
    ranks = []
    entropies = []
    pred_tokens = 0
    n = input_ids.size(1)
    prev_end = 0; idx = 0
    for begin in range(0, n, STRIDE):
        end = min(begin + MAX_CONTEXT, n)
        trg_len = end - prev_end
        gold = input_ids[:, begin:end]
        logits = logits_list[idx]
        idx += 1
        for t in range(trg_len):
            gold_id = int(gold[0, -trg_len + t])
            probs = torch.softmax(torch.tensor(logits[t]), dim=-1).numpy()
            order = np.argsort(-probs)
            rank = int(np.where(order == gold_id)[0][0]) + 1
            ranks.append(rank)
            ent = -float(np.sum(probs * np.log2(probs + 1e-12)))
            entropies.append(ent)
        prev_end = end
        pred_tokens += trg_len
        if end==n: break

    ranks = np.array(ranks, dtype=int) if ranks else np.array([], dtype=int)
    entropies = np.array(entropies, dtype=float) if entropies else np.array([], dtype=float)
    if ranks.size==0:
        return {"avg_logprob":avg_logprob, "ppl":ppl,
                "proportion_top10":0.0,"proportion_top100":0.0,"proportion_top1000":0.0,
                "proportion_over1000":0.0,"mean_rank":0.0,"median_rank":0.0,
                "rank_entropy":0.0,"n_pred_tokens":0.0}

    prop10 = float(np.mean(ranks<=10))
    prop100 = float(np.mean(ranks<=100))
    prop1000 = float(np.mean(ranks<=1000))
    over1000 = float(np.mean(ranks>1000))
    return {
        "avg_logprob": float(avg_logprob),
        "ppl": float(ppl),
        "proportion_top10": prop10,
        "proportion_top100": prop100,
        "proportion_top1000": prop1000,
        "proportion_over1000": over1000,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "rank_entropy": float(np.mean(entropies)) if entropies.size>0 else 0.0,
        "n_pred_tokens": float(pred_tokens)
    }

# --- 13) NLI consistency
@torch.no_grad()
def nli_probs_batch(prem: List[str], hyp: List[str]) -> np.ndarray:
    enc = nli_tok(prem, hyp, return_tensors="pt", truncation=True, padding=True).to(nli_model.device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    probs = probs[:, [IDX_CONTRA, IDX_NEUTRAL, IDX_ENTAIL]]
    return probs

@torch.no_grad()
def nli_probs_single(prem: str, hyp: str) -> Tuple[float,float,float]:
    if not prem.strip() or not hyp.strip(): return 0.0, 1.0, 0.0
    enc = nli_tok(prem, hyp, return_tensors="pt", truncation=True, padding=True).to(nli_model.device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    return float(probs[IDX_CONTRA]), float(probs[IDX_NEUTRAL]), float(probs[IDX_ENTAIL])

def feat_nli(response: str, question: str) -> dict:
    if not USE_NLI:
        return {k:0.0 for k in [
            "nli_mean_entail","nli_mean_contra","nli_max_contra",
            "nli_frac_entail_gt50","nli_frac_contra_gt50","nli_pairs_evaluated",
            "nli_r_to_q_entail","nli_q_to_r_entail","nli_r_to_q_contra","nli_q_to_r_contra",
            "n_sentences_resp"
        ]}
    sents = sentence_split(response)
    sents = sents[:MAX_SENTENCES_FOR_NLI] if len(sents)>MAX_SENTENCES_FOR_NLI else sents
    pairs=[]
    if len(sents)>=2:
        if PAIR_STRATEGY=="all_pairs":
            for i in range(len(sents)):
                for j in range(i+1,len(sents)):
                    pairs.append((sents[i], sents[j]))
                    pairs.append((sents[j], sents[i]))
        else:
            for i in range(len(sents)-1):
                pairs.append((sents[i], sents[i+1]))
                pairs.append((sents[i+1], sents[i]))
    if pairs:
        P = nli_probs_batch([p for p,_ in pairs], [h for _,h in pairs])
        p_contra, p_neu, p_ent = P[:,0], P[:,1], P[:,2]
        mean_ent = float(np.mean(p_ent))
        mean_con = float(np.mean(p_contra))
        max_con  = float(np.max(p_contra))
        frac_ent = float(np.mean(p_ent>0.5))
        frac_con = float(np.mean(p_contra>0.5))
        n_pairs = int(len(pairs))
    else:
        mean_ent=mean_con=max_con=frac_ent=frac_con=0.0; n_pairs=0

    if question and response:
        c_rq, n_rq, e_rq = nli_probs_single(response, question)
        c_qr, n_qr, e_qr = nli_probs_single(question, response)
        r2q_e, q2r_e = float(e_rq), float(e_qr)
        r2q_c, q2r_c = float(c_rq), float(c_qr)
    else:
        r2q_e=q2r_e=r2q_c=q2r_c=0.0

    return {
        "nli_mean_entail": mean_ent,
        "nli_mean_contra": mean_con,
        "nli_max_contra": max_con,
        "nli_frac_entail_gt50": frac_ent,
        "nli_frac_contra_gt50": frac_con,
        "nli_pairs_evaluated": float(n_pairs),
        "nli_r_to_q_entail": r2q_e,
        "nli_q_to_r_entail": q2r_e,
        "nli_r_to_q_contra": r2q_c,
        "nli_q_to_r_contra": q2r_c,
        "n_sentences_resp": float(len(sents)),
    }

# --- 14) Curvature (DetectGPT-style) — PATCHED HELPERS
# Hard cap for words sent to masked-LM so we stay under ~512 subword tokens
WORD_LIMIT_FOR_MLM = 380  # conservative for roberta-base

def _truncate_by_words(text: str, limit: int = WORD_LIMIT_FOR_MLM) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])

def _truncate_by_tokenizer(text: str, tokenizer, max_len: int = None) -> str:
    if max_len is None:
        max_len = getattr(tokenizer, "model_max_length", 512) or 512
        if max_len > 1024: max_len = 512
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_len)
    if enc["input_ids"].size(1) <= max_len:
        return text
    ids = enc["input_ids"][0]
    return tokenizer.decode(ids, skip_special_tokens=True)

def split_words(text: str) -> List[str]:
    return [w for w in re.findall(r"\S+", text)]

def sample_spans(n_tokens: int, mask_prob: float, avg_span: int) -> List[Tuple[int,int]]:
    if n_tokens==0 or mask_prob<=0: return []
    target = max(1, int(round(n_tokens * mask_prob)))
    spans=[]; covered=set(); attempts=0
    while len(covered)<target and attempts<max(5*target,50):
        span_len = max(1, np.random.geometric(1.0/max(avg_span,1)))
        start = np.random.randint(0, n_tokens)
        end = min(start+span_len, n_tokens)
        idxs=set(range(start,end))
        if not (covered & idxs):
            spans.append((start,end)); covered |= idxs
        attempts+=1
    spans.sort(); return spans

def build_masked_text(words: List[str], spans: List[Tuple[int,int]]) -> str:
    if not spans: return " ".join(words)
    out=[]; it=iter(spans); cur=next(it, None); i=0; N=len(words)
    while i<N:
        if cur and i==cur[0]:
            length = cur[1]-cur[0]
            out.extend([MASK_TOKEN]*length)
            i = cur[1]; cur=next(it, None)
        else:
            out.append(words[i]); i+=1
    return _truncate_by_words(" ".join(out), WORD_LIMIT_FOR_MLM)

@torch.no_grad()
def avg_loglikelihood(text: str) -> float:
    if not isinstance(text,str) or not text.strip(): return float("-inf")
    enc = score_tok(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(score_model.device)
    n = input_ids.size(1)
    if n==0: return float("-inf")
    if n<=MAX_CONTEXT:
        out = score_model(input_ids, labels=input_ids)
        return -float(out.loss.item())
    total_nll=0.0; prev_end=0; total_pred=0
    for begin in range(0, n, STRIDE):
        end = min(begin+MAX_CONTEXT, n)
        trg_len = end - prev_end
        ids_slice = input_ids[:, begin:end]
        target = ids_slice.clone()
        if target.size(1) > trg_len:
            target[:, :-trg_len] = -100
        out = score_model(ids_slice, labels=target)
        total_nll += out.loss.item() * trg_len
        total_pred += trg_len
        prev_end = end
        if end==n: break
    return -(total_nll/max(total_pred,1))

def fill_masks_once(text_with_masks: str) -> str:
    if MASK_TOKEN not in text_with_masks: return text_with_masks
    result = text_with_masks
    for _ in range(512):
        if MASK_TOKEN not in result: break
        tmp = result.replace(MASK_TOKEN, fill_mask.tokenizer.mask_token, 1)
        tmp = _truncate_by_tokenizer(tmp, fill_mask.tokenizer)
        preds = fill_mask(tmp)
        if isinstance(preds, list) and preds and isinstance(preds[0], dict):
            choice = random.choice(preds[:TOPK_FILL]) if len(preds)>=TOPK_FILL else preds[0]
            token_str = choice["token_str"].strip()
            result = tmp.replace(fill_mask.tokenizer.mask_token, token_str, 1)
        else:
            result = tmp.replace(fill_mask.tokenizer.mask_token, "", 1)
    return result

def feat_curvature(text: str) -> dict:
    if not USE_CURVATURE:
        return {"ll_orig":0.0,"mean_ll_pert":0.0,"std_ll_pert":0.0,"curvature_z":0.0,"n_perturb_used":0}

    base_text = _truncate_by_words(text, WORD_LIMIT_FOR_MLM)
    ll_orig = avg_loglikelihood(base_text)
    words = split_words(base_text)

    perts=[]
    for _ in range(NUM_PERTURB):
        spans = sample_spans(len(words), MASK_PROB, AVG_SPAN)
        masked = build_masked_text(words, spans)
        filled = fill_masks_once(masked)
        perts.append(filled)

    ll_perts=[]
    for ptxt in perts:
        try: ll_perts.append(avg_loglikelihood(ptxt))
        except Exception: pass

    if not ll_perts:
        return {"ll_orig": ll_orig, "mean_ll_pert": ll_orig, "std_ll_pert": 0.0, "curvature_z": 0.0, "n_perturb_used":0}
    mean_p = float(np.mean(ll_perts)); std_p=float(np.std(ll_perts))
    z = (ll_orig - mean_p)/(std_p+1e-8)
    return {"ll_orig": ll_orig, "mean_ll_pert": mean_p, "std_ll_pert": std_p, "curvature_z": float(z), "n_perturb_used": int(len(ll_perts))}

# ============================================================
# END FEATURE EXTRACTORS
# ============================================================

def compute_all_features_for_row(response: str, question: str) -> dict:
    row = {}
    row.update(feat_sentence_length(response))
    row.update(feat_readability(response))
    row.update(feat_lexical_richness(response))
    row.update(feat_pos(response))
    row.update(feat_dependency(response))
    row.update(feat_ner(response))
    row.update(feat_sentiment(response))
    row.update(feat_discourse(response))
    row.update(feat_entity_grid(response))
    row.update(feat_embeddings(question, response) if USE_EMBEDDINGS else {
        "cosine_sim":0.0,"q_norm":0.0,"r_norm":0.0,"norm_gap":0.0
    })
    row.update(feat_perplex_and_tokenrank(response) if (USE_PERPLEX or USE_TOKENRANK) else {
        "avg_logprob":0.0,"ppl":0.0,
        "proportion_top10":0.0,"proportion_top100":0.0,"proportion_top1000":0.0,"proportion_over1000":0.0,
        "mean_rank":0.0,"median_rank":0.0,"rank_entropy":0.0,"n_pred_tokens":0.0
    })
    row.update(feat_nli(response, question) if USE_NLI else {
        "nli_mean_entail":0.0,"nli_mean_contra":0.0,"nli_max_contra":0.0,
        "nli_frac_entail_gt50":0.0,"nli_frac_contra_gt50":0.0,"nli_pairs_evaluated":0.0,
        "nli_r_to_q_entail":0.0,"nli_q_to_r_entail":0.0,"nli_r_to_q_contra":0.0,"nli_q_to_r_contra":0.0,
        "n_sentences_resp":0.0
    })
    row.update(feat_curvature(response) if USE_CURVATURE else {
        "ll_orig":0.0,"mean_ll_pert":0.0,"std_ll_pert":0.0,"curvature_z":0.0,"n_perturb_used":0
    })
    return row

def build_master_table(files: List[str]) -> pd.DataFrame:
    all_rows=[]
    for j, fpath in enumerate(files):
        p = Path(fpath)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")
        df = pd.read_csv(p)
        responses = merge_cols(df, RESPONSE_COL_PATTERN, fallback_empty=False)
        questions = merge_cols(df, QUESTION_COL_PATTERN, fallback_empty=True)  # optional
        if RATING_COL not in df.columns:
            raise ValueError(f"Rating column '{RATING_COL}' not in {p.name}")
        ratings = map_rating(df[RATING_COL])
        iterator = zip(responses, questions, ratings)
        iterator = tqdm(list(iterator), desc=f"Features {p.name}", unit="row") if TQDM else iterator

        for i, (resp, ques, y) in enumerate(iterator):
            try:
                feats = compute_all_features_for_row(resp, ques)
            except Exception as e:
                # Fail-safe: in case any extractor blows up, keep row with zeros for heavy features.
                feats = {}
                feats.update(feat_sentence_length(resp))
                feats.update(feat_readability(resp))
                feats.update(feat_lexical_richness(resp))
                feats.update(feat_pos(resp))
                feats.update(feat_dependency(resp))
                feats.update(feat_ner(resp))
                feats.update(feat_sentiment(resp))
                feats.update(feat_discourse(resp))
                feats.update(feat_entity_grid(resp))
                feats.update({"cosine_sim":0.0,"q_norm":0.0,"r_norm":0.0,"norm_gap":0.0})
                feats.update({"avg_logprob":0.0,"ppl":0.0,"proportion_top10":0.0,"proportion_top100":0.0,
                              "proportion_top1000":0.0,"proportion_over1000":0.0,"mean_rank":0.0,"median_rank":0.0,
                              "rank_entropy":0.0,"n_pred_tokens":0.0})
                feats.update({"nli_mean_entail":0.0,"nli_mean_contra":0.0,"nli_max_contra":0.0,
                              "nli_frac_entail_gt50":0.0,"nli_frac_contra_gt50":0.0,"nli_pairs_evaluated":0.0,
                              "nli_r_to_q_entail":0.0,"nli_q_to_r_entail":0.0,"nli_r_to_q_contra":0.0,"nli_q_to_r_contra":0.0,
                              "n_sentences_resp":0.0})
                feats.update({"ll_orig":0.0,"mean_ll_pert":0.0,"std_ll_pert":0.0,"curvature_z":0.0,"n_perturb_used":0})
            feats.update({"user_id": i, "file_id": j, "rating": float(y)})
            all_rows.append(feats)
    data = pd.DataFrame(all_rows)
    # Clean numeric
    for c in data.columns:
        if c in {"user_id","file_id"}: continue
        if pd.api.types.is_numeric_dtype(data[c]):
            data[c] = data[c].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return data

# -------------------------
# MODEL COMBOS
# -------------------------
COMBOS: Dict[str, List[str]] = {
    # 4–5 vars
    "combo_4_A__embed_nli_pos_read":     ["cosine_sim","q_norm","r_norm","norm_gap",
                                          "nli_mean_entail","nli_mean_contra","nli_max_contra",
                                          "nli_frac_entail_gt50","nli_frac_contra_gt50",
                                          "content_vs_function_ratio","pronoun_ratio","verbs_per_sentence","pos_entropy",
                                          "flesch_reading_ease","fk_grade_level","gunning_fog","smog_index","ari","coleman_liau",
                                         ],
    "combo_5_B__perp_token_curv_lex_disc": [
        "avg_logprob","ppl","proportion_top10","proportion_top100","proportion_top1000","mean_rank","rank_entropy",
        "curvature_z","ll_orig","mean_ll_pert","std_ll_pert",
        "ttr","rttr","cttr","herdan_c","maas_a2","yules_k","mtld","hapax_ratio","dis_legomena_ratio","avg_word_len",
        "total_connectives","connective_entropy","connective_density_per_100w","enumeration_count"
    ],
    "combo_5_C__dep_grid_ner_sent_len":  [
        "avg_tree_depth","max_tree_depth","std_tree_depth","avg_dep_distance","root_children_mean",
        "unique_entities","avg_entities_per_sent","entity_continuity","entity_introductions","role_transitions","grid_density",
        "entity_count","entity_density","unique_entity_types","entity_label_entropy","avg_entity_length_tokens",
        "vader_compound","tb_subjectivity","sentence_length"
    ],

    # 7–8 vars
    "combo_8_D__embed_nli_pos_read_lex_dep_grid_sent": [
        "cosine_sim","norm_gap",
        "nli_mean_entail","nli_mean_contra",
        "content_vs_function_ratio","pos_entropy","verbs_per_sentence",
        "flesch_reading_ease","fk_grade_level",
        "ttr","mtld","yules_k",
        "avg_tree_depth","avg_dep_distance",
        "entity_continuity","role_transitions",
        "vader_compound","tb_subjectivity",
        "sentence_length"
    ],

    # 10–12 vars
    "combo_11_E__add_perp_token_ner_disc": [
        "cosine_sim","norm_gap","nli_mean_entail","nli_mean_contra",
        "content_vs_function_ratio","pos_entropy","verbs_per_sentence",
        "flesch_reading_ease","fk_grade_level",
        "ttr","mtld","yules_k",
        "avg_tree_depth","avg_dep_distance",
        "entity_continuity","role_transitions",
        "vader_compound","tb_subjectivity",
        "sentence_length",
        "avg_logprob","ppl","proportion_top100",
        "entity_density","entity_label_entropy",
        "connective_entropy","connective_density_per_100w"
    ],

    # ALL variables: automatically pick all numeric non-key columns
    "combo_ALL": ["__ALL__"]
}

def columns_for_combo(data: pd.DataFrame, combo_name: str, spec: List[str]) -> List[str]:
    key_cols = {"user_id","file_id","rating"}
    if spec == ["__ALL__"]:
        return [c for c in data.columns if c not in key_cols and pd.api.types.is_numeric_dtype(data[c])]
    cols = [c for c in spec if (c in data.columns and pd.api.types.is_numeric_dtype(data[c]))]
    return cols

def train_and_save_combo(data: pd.DataFrame, combo_name: str, feat_cols: List[str]):
    X = data[feat_cols].astype(float)
    y = data["rating"].astype(float)
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, data.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    rf = RandomForestRegressor(
        n_estimators=800, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(Xtr, ytr)
    yhat = rf.predict(Xte)
    mse = mean_squared_error(yte, yhat)
    print(f"[{combo_name}] features={len(feat_cols)}  MSE={mse:.4f}")

    # row-level predictions for all rows
    data_out = data[["user_id","file_id","rating"]].copy()
    data_out[f"{combo_name}__pred"] = np.clip(rf.predict(X), 1.0, 5.0).round(2)
    data_out = pd.concat([data_out, X.round(4)], axis=1)
    rows_path = OUT_DIR / f"{combo_name}__rows.csv"
    data_out.to_csv(rows_path, index=False)

    # per-user
    per_user = (
        data_out.groupby("user_id", as_index=False)
                .agg(true_avg_rating=("rating","mean"),
                     pred_avg_rating=(f"{combo_name}__pred","mean"),
                     n_rows=("rating","size"))
    )
    per_user["true_avg_rating"] = per_user["true_avg_rating"].round(2)
    per_user["pred_avg_rating"] = per_user["pred_avg_rating"].round(2)
    per_user_path = OUT_DIR / f"{combo_name}__per_user.csv"
    per_user.to_csv(per_user_path, index=False)

    return mse, str(rows_path), str(per_user_path)

def main():
    print("Building master feature table from raw CSVs ...")
    data = build_master_table(FILES)
    full_path = OUT_DIR / "master_features_all.csv"
    data.to_csv(full_path, index=False)
    print(f"Master table saved to: {full_path}  (rows={len(data)})")

    results=[]
    for name, spec in COMBOS.items():
        feat_cols = columns_for_combo(data, name, spec)
        if not feat_cols:
            print(f"!! Skipping {name}: no feature columns found in data for this combo.")
            continue
        mse, rows_path, per_user_path = train_and_save_combo(data, name, feat_cols)
        results.append((name, mse, rows_path, per_user_path))

    if results:
        print("\n=== Summary (lower MSE is better) ===")
        for name, mse, rows_path, per_user_path in results:
            print(f"{name:28s}  MSE={mse:.4f}  rows={rows_path}  per_user={per_user_path}")
    else:
        print("No models were trained. Check your files/config.")

if __name__ == "__main__":
    main()
