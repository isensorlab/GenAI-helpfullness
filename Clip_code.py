# ============================================================
# All Variables (no curvature/NLI) + CLIP: Text → Features → Helpfulness (1–5)
# ============================================================

import re
import os
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NLP core
import spacy

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Transformers (CLIP + causal LM for perplexity)
import torch
from transformers import (
    CLIPTokenizer, CLIPModel,
    AutoTokenizer, AutoModelForCausalLM
)

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# -------------------------
# CONFIG
# -------------------------
FILES: List[str] = [
     "Q4_Submission1_cleaned.csv",
    "Q4_Submission2_cleaned.csv",
    "Q4_Submission3_cleaned.csv",
    "Q4_Submission4_cleaned.csv",
    "Q4_Submission5_cleaned.csv",
]

RATING_COL = "Rating"
RATING_MAP: Dict[str, int] = {
    "Not Helpful at All": 1,
    "Slightly Helpful":   2,
    "Neutral":            3,
    "Helpful":            4,
    "Very Helpful":       5,
}

RESPONSE_COL_PATTERN = re.compile(r"response", re.IGNORECASE)
QUESTION_COL_PATTERN = re.compile(r"(question|prompt|query)", re.IGNORECASE)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# CLIP model
CLIP_NAME   = "openai/clip-vit-base-patch32"
CLIP_BATCH  = 32
# If True, CLIP encodes "question + response" instead of just response (optional)
USE_Q_PLUS_R_FOR_CLIP = False

# Perplexity model
LM_NAME     = "distilgpt2"
MAX_CONTEXT = 1024
STRIDE      = 512

OUT_DIR = Path("allvars_clip_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
        mapped = series.map(RATING_MAP)
        if mapped.isna().any():
            unknown = series[mapped.isna()].unique()
            raise ValueError(f"Unmapped ratings detected: {unknown}")
        return mapped.astype(float)
    return series.astype(float).clip(1.0, 5.0)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
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
nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 2_000_000

# -------------------------
# Sentiment analyzers
# -------------------------
try:
    _vader = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon")
    _vader = SentimentIntensityAnalyzer()

# -------------------------
# CLIP encoder
# -------------------------
class ClipTextEncoder:
    def __init__(self, model_name: str = CLIP_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = CLIP_BATCH) -> np.ndarray:
        vecs_all = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**enc)  # [B, 512], L2-normalized
            vecs_all.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(vecs_all)

    @torch.no_grad()
    def cosine_sim(self, a_texts: List[str], b_texts: List[str]) -> np.ndarray:
        a = self.encode(a_texts)
        b = self.encode(b_texts)
        return np.sum(a * b, axis=1)  # dot == cosine (already normalized)

# -------------------------
# Causal LM for perplexity
# -------------------------
print("Loading causal LM for perplexity ...")
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

@torch.no_grad()
def avg_logprob_and_ppl(text: str) -> Tuple[float, float]:
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
    enc = score_tok(text, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"].to(score_model.device)
    n = input_ids.size(1)
    if n == 0:
        return 0.0, 0.0
    if n <= MAX_CONTEXT:
        out = score_model(input_ids, labels=input_ids)
        avg_nll = out.loss.item()
        return -avg_nll, float(math.exp(avg_nll))
    total_nll, total_pred = 0.0, 0
    prev_end = 0
    for begin in range(0, n, STRIDE):
        end = min(begin + MAX_CONTEXT, n)
        trg_len = end - prev_end
        ids_slice = input_ids[:, begin:end]
        target = ids_slice.clone()
        if target.size(1) > trg_len:
            target[:, :-trg_len] = -100
        out = score_model(ids_slice, labels=target)
        total_nll += out.loss.item() * trg_len
        total_pred += trg_len
        prev_end = end
        if end == n:
            break
    if total_pred == 0:
        return 0.0, 0.0
    avg_nll = total_nll / total_pred
    return -avg_nll, float(math.exp(avg_nll))

# ============================================================
# FEATURE EXTRACTORS (no curvature/NLI)
# ============================================================
def feat_sentence_length(text: str) -> dict:
    sents = sentence_split(text)
    toks = tokenize_words(text)
    return {"n_sentences": len(sents), "n_tokens": len(toks), "sentence_length": len(toks)}

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
    Vi = Counter(freqs.values()); sum_i2Vi = sum((i*i)*Vi for i,Vi in Vi.items())
    K = 1e4 * ((sum_i2Vi - N)/(N*N))
    lnN = safe_log(N)
    Maas = (lnN - safe_log(V))/(lnN*lnN) if (lnN>0 and V>1) else 0.0
    return {
        "tokens": N, "types": V,
        "ttr": V/N, "rttr": V/math.sqrt(N), "cttr": V/math.sqrt(2*N),
        "herdan_c": (safe_log(V)/lnN) if (V>1 and lnN>0) else 0.0,
        "maas_a2": Maas, "yules_k": K, "mtld": mtld(toks),
        "hapax_ratio": hapax/max(V,1), "dis_legomena_ratio": dis2/max(V,1),
        "avg_word_len": float(avg_len),
    }

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
    n_sents = max(len(list(doc.sents)), 1)
    verbs_per_sent = (pos_counts.get("VERB",0)+pos_counts.get("AUX",0))/n_sents
    if n_tok>0 and pos_counts:
        dist = np.array([c/n_tok for c in pos_counts.values()], dtype=float)
        pos_entropy = -float(np.sum(dist*np.log2(dist+1e-12)))
    else:
        pos_entropy = 0.0
    return {
        "token_count": n_tok, "sentence_count": n_sents,
        **props,
        "content_vs_function_ratio": (content/max(function,1)) if function>0 else float(content>0),
        "pronoun_ratio": pron_ratio, "verbs_per_sentence": verbs_per_sent,
        "pos_entropy": pos_entropy,
    }

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
                "avg_dep_distance":0.0,"root_children_mean":0.0,"n_sentences_dep":0,"n_tokens_dep":0}
    return {
        "avg_tree_depth": float(np.mean(depths)),
        "max_tree_depth": float(np.max(depths)),
        "std_tree_depth": float(np.std(depths)),
        "avg_dep_distance": float(np.mean(dists)),
        "root_children_mean": float(np.mean(roots)),
        "n_sentences_dep": n_sent,
        "n_tokens_dep": n_tok
    }

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

TEMPORAL = ["after","afterward","afterwards","before","earlier","later","meantime","meanwhile","subsequently",
            "then","next","finally","first","second","third","fourth","ultimately","previously","simultaneously","soon",
            "eventually"]
COMPARISON = ["however","but","nevertheless","nonetheless","yet","though","although","whereas","while","instead",
              "in contrast","on the other hand","rather"]
CONTINGENCY = ["because","since","as","so","therefore","thus","hence","consequently","accordingly","as a result",
               "due to","owing to"]
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

def feat_ppl(text: str) -> dict:
    avg_logprob, ppl = avg_logprob_and_ppl(text)
    return {"avg_logprob": float(avg_logprob), "ppl": float(ppl)}

def compute_intrinsic_features(response: str) -> dict:
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
    row.update(feat_ppl(response))
    return row

# -------------------------
# MASTER TABLE (with CLIP)
# -------------------------
def build_master_table(files: List[str], clip) -> pd.DataFrame:
    all_rows=[]
    for j, fpath in enumerate(files):
        p = Path(fpath)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p.resolve()}")
        df = pd.read_csv(p)

        responses = merge_cols(df, RESPONSE_COL_PATTERN, fallback_empty=False)
        questions = merge_cols(df, QUESTION_COL_PATTERN, fallback_empty=True)
        if RATING_COL not in df.columns:
            raise ValueError(f"Rating column '{RATING_COL}' not in {p.name}")
        ratings = map_rating(df[RATING_COL])

        rows = list(zip(responses, questions, ratings))
        iterator = tqdm(rows, desc=f"Features {p.name}", unit="row") if TQDM else rows

        # Prepare CLIP batches
        resp_texts = []
        q_texts = []
        for resp, ques, _ in iterator:
            resp_texts.append(resp if not USE_Q_PLUS_R_FOR_CLIP else (f"{ques} {resp}".strip()))
            q_texts.append(ques)

        # Encode CLIP embeddings once per file
        resp_emb = clip.encode(resp_texts)  # [N, 512]
        # CLIP cosine(question, response)
        if any(str(q).strip() for q in q_texts):
            clip_cos = clip.cosine_sim(q_texts, resp_texts if USE_Q_PLUS_R_FOR_CLIP else responses.tolist())
        else:
            clip_cos = np.zeros(len(resp_texts), dtype=np.float32)

        for i, ((resp, ques, y), rvec, cos_sim) in enumerate(zip(zip(responses, questions, ratings), resp_emb, clip_cos)):
            feats = compute_intrinsic_features(resp)
            for k in range(rvec.shape[0]):
                feats[f"clip_dim_{k:03d}"] = float(rvec[k])
            feats["clip_qr_cosine"] = float(cos_sim)
            feats.update({"user_id": i, "file_id": j, "rating": float(y)})
            all_rows.append(feats)

    data = pd.DataFrame(all_rows)
    for c in data.columns:
        if c in {"user_id","file_id"}: continue
        if pd.api.types.is_numeric_dtype(data[c]):
            data[c] = data[c].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return data

# -------------------------
# TRAIN & EVAL
# -------------------------
def main():
    print("Loading CLIP text encoder ...")
    clip = ClipTextEncoder(CLIP_NAME)

    print("Building master feature table (intrinsic + CLIP) ...")
    data = build_master_table(FILES, clip)
    full_path = OUT_DIR / "allvars_clip_master_features.csv"
    data.to_csv(full_path, index=False)
    print(f"Master features saved to: {full_path} (rows={len(data)}, cols={len(data.columns)})")

    key_cols = {"user_id","file_id","rating"}
    feat_cols = [c for c in data.columns if c not in key_cols and pd.api.types.is_numeric_dtype(data[c])]
    X = data[feat_cols].astype(float).values
    y = data["rating"].astype(float).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    rf = RandomForestRegressor(
        n_estimators=800, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(Xtr, ytr)

    yhat = rf.predict(Xte)
    mse = mean_squared_error(yte, yhat)
    mae = mean_absolute_error(yte, yhat)
    r2  = r2_score(yte, yhat)
    print(f"\n=== All-Variables + CLIP model ===")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print(f"Features used: {len(feat_cols)}")

    data_out = data[["user_id","file_id","rating"]].copy()
    preds = np.clip(rf.predict(X), 1.0, 5.0)
    data_out["pred"] = np.round(preds, 2)
    rows_path = OUT_DIR / "allvars_clip_rows.csv"
    data_out.to_csv(rows_path, index=False)
    print(f"Saved row-level predictions to: {rows_path.resolve()}")

    per_user = (data_out
                .groupby("user_id", as_index=False)
                .agg(true_avg=("rating","mean"),
                     pred_avg=("pred","mean"),
                     n_rows=("rating","size")))
    per_user["true_avg"] = per_user["true_avg"].round(2)
    per_user["pred_avg"] = per_user["pred_avg"].round(2)
    per_user_path = OUT_DIR / "allvars_clip_per_user.csv"
    per_user.to_csv(per_user_path, index=False)
    print(f"Saved per-user averages to: {per_user_path.resolve()}")

if __name__ == "__main__":
    main()
