#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KodCode-Light-RL-10K → concept vectors (global & per-subset) with CodeLlama.

Outputs:
  out/
    - embeddings_X.npy                      (N, D)
    - teacher_basis_BH.npy                  (D, k)
    - svd_meta.json
    - prevalence_by_subset.csv
    - cavs_global.json                      {concept: {"mean": [...], "probe": [...]} }
    - cavs_per_subset.json                  {subset: {concept: {"mean":[...], "probe":[...]}}}

Run:
  python kodcode_subset_concepts.py \
    --model codellama/CodeLlama-7b-Instruct-hf \
    --out out --k 64 --max_len 2048 --max_items 0
"""

import os, json, re, argparse, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

# ----------------- Concepts & Heuristics -----------------

CONCEPTS = [
    "boundary_values",
    "invalid_types",
    "empty_null",
    "negative_paths",
    "parametrization",
    "randomized_property",
    "float_tolerance",
    "file_io",
    "large_input_stress",
    "idempotency",
]

# Regexes (literal-aware empties, exceptions, etc.)
RE_INVALID_TYPES  = re.compile(r"pytest\.raises\s*\(\s*TypeError\s*\)", re.I)
RE_ANY_RAISES     = re.compile(r"pytest\.raises\s*\(\s*[A-Za-z_]\w*\s*\)", re.I)
RE_PARAMETRIZE    = re.compile(r"@pytest\.mark\.parametrize", re.I)
RE_HYPOTHESIS     = re.compile(r"(from\s+hypothesis|@given\()", re.I)
RE_RANDOM         = re.compile(r"\brandom\.", re.I)
RE_ISCLOSE        = re.compile(r"(math\.isclose|numpy\.isclose|np\.isclose)", re.I)
RE_FILEIO         = re.compile(r"\b(open|tempfile|NamedTemporaryFile|mkstemp)\b")
RE_NONE_LITERAL   = re.compile(r'(?<!["\'])\bNone\b(?!["\'])')
RE_EMPTY_STR      = re.compile(r'(?<!\S)(["\']{2})(?!\S)')
RE_EMPTY_LIST     = re.compile(r'\[\s*\]')
RE_EMPTY_TUPLE    = re.compile(r'\(\s*\)')
RE_EMPTY_DICT     = re.compile(r'\{\s*\}')
RE_FUNC_CALL      = re.compile(r'([A-Za-z_]\w*)\s*\(')
BOUNDARY_LITS     = {"-1","0","1"}

def tag_concepts(test_code: str, spec: str) -> Dict[str,int]:
    c = {k: 0 for k in CONCEPTS}
    # boundary values (simple; extend if you like)
    if any(l in test_code for l in BOUNDARY_LITS) or re.search(r"(min|max|len)\s*\(", test_code):
        c["boundary_values"] = 1
    # invalid types / negative paths
    if RE_INVALID_TYPES.search(test_code): c["invalid_types"] = 1
    if RE_ANY_RAISES.search(test_code):    c["negative_paths"] = 1
    # empty/null (literal-aware)
    if (RE_NONE_LITERAL.search(test_code) or RE_EMPTY_STR.search(test_code) or
        RE_EMPTY_LIST.search(test_code) or RE_EMPTY_TUPLE.search(test_code) or RE_EMPTY_DICT.search(test_code)):
        c["empty_null"] = 1
    # parametrization (explicit or manual)
    if (RE_PARAMETRIZE.search(test_code) or
        (test_code.count("assert") >= 3 and len(RE_FUNC_CALL.findall(test_code)) >= 4)):
        c["parametrization"] = 1
    # randomized/property
    if RE_HYPOTHESIS.search(test_code) or RE_RANDOM.search(test_code):
        c["randomized_property"] = 1
    # float tolerance
    if RE_ISCLOSE.search(test_code): c["float_tolerance"] = 1
    # file io
    if RE_FILEIO.search(test_code):  c["file_io"] = 1
    # large input stress (very coarse)
    if ("large" in spec.lower() or "big" in spec.lower() or re.search(r"range\(\s*\d{3,}\s*\)", test_code)):
        c["large_input_stress"] = 1
    # idempotency: two calls + equality
    if re.search(r"assert\s+.+==.+", test_code) and len(RE_FUNC_CALL.findall(test_code)) >= 2:
        c["idempotency"] = 1
    return c

# ----------------- Data Schema -----------------

@dataclass
class Row:
    i: int
    subset: str
    question: str
    solution: str
    test: str
    tags: Dict[str,int]

# ----------------- Embeddings (CodeLlama) -----------------

def load_llm(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    # 4-bit option (uncomment if needed)
    # from transformers import BitsAndBytesConfig
    # quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
    #                            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16, output_hidden_states=True
        # , quantization_config=quant
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def encode_hidden_means(texts: List[str], tok, model, max_len: int, layer_idx: int = -2) -> np.ndarray:
    import torch
    vecs = []
    model.eval()
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len, padding=False)
        enc = {k: v.to(model.device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            h  = out.hidden_states[layer_idx].mean(dim=1).squeeze(0).float().cpu().numpy()
            vecs.append(h)
    return np.stack(vecs, 0)

# ----------------- CAVs (mean-diff & probe) -----------------

def compute_svd(X: np.ndarray, k: int):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=k, random_state=13)
    Z = svd.fit_transform(X)   # (N, k)
    BH = svd.components_.T     # (D, k)
    meta = {"explained_variance_ratio": svd.explained_variance_ratio_.tolist()}
    return Z, BH, meta

def cav_mean_diff(X: np.ndarray, y: np.ndarray):
    pos, neg = X[y==1], X[y==0]
    if len(pos) < 10 or len(neg) < 10:  # guard for stability
        return None
    v = pos.mean(0) - neg.mean(0)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else None

def cav_probe(Z: np.ndarray, BH: np.ndarray, y: np.ndarray):
    # L2-logistic on SVD space; lift back to hidden space
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return None
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=200)
    clf.fit(Z, y)
    w = clf.coef_.reshape(-1)     # (k,)
    v = BH @ w                    # (D,)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else None

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument("--concat_spec", action="store_true", help="prepend question to tests before embedding")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) Load dataset (columns per HF card: question, solution, test, subset)
    #    Source: KodCode/KodCode-Light-RL-10K dataset page.  (uses 12 subset classes)
    #    We'll use the 'train' split (10k rows).
    from datasets import load_dataset
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")  # 10k
    if args.max_items and len(ds) > args.max_items:
        ds = ds.select(range(args.max_items))

    rows: List[Row] = []
    for i, r in enumerate(ds):
        subset = r.get("subset", "unknown")
        question = r.get("question", "")
        solution = r.get("solution", "")
        test = r.get("test", "")
        if not test:
            continue
        tags = tag_concepts(test, question)
        rows.append(Row(i=i, subset=subset, question=question, solution=solution, test=test, tags=tags))

    # 2) Prevalence table (subset × concept)
    subs = sorted(set(r.subset for r in rows))
    prev = []
    for s in subs:
        r_s = [r for r in rows if r.subset == s]
        for c in CONCEPTS:
            pos = sum(rr.tags[c] for rr in r_s)
            prev.append({"subset": s, "concept": c, "positives": pos, "total": len(r_s)})
    df_prev = pd.DataFrame(prev)
    df_prev.to_csv(os.path.join(args.out, "prevalence_by_subset.csv"), index=False)

    # 3) Embed tests with CodeLlama
    print(f"[embed] Using {args.model} …")
    tok, model = load_llm(args.model)
    texts = [ (r.question + "\n\n" + r.test) if args.concat_spec else r.test for r in rows ]
    X = encode_hidden_means(texts, tok, model, max_len=args.max_len, layer_idx=-2)  # (N, D)
    np.save(os.path.join(args.out, "embeddings_X.npy"), X)

    # 4) Teacher basis (SVD)
    Z, BH, meta = compute_svd(X, k=args.k)
    np.save(os.path.join(args.out, "teacher_basis_BH.npy"), BH)
    with open(os.path.join(args.out, "svd_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 5) Build CAVs — global
    Y = {c: np.array([r.tags[c] for r in rows], dtype=np.int32) for c in CONCEPTS}
    cavs_global = {}
    for c in CONCEPTS:
        v_mean  = cav_mean_diff(X, Y[c])
        v_probe = cav_probe(Z, BH, Y[c])
        cavs_global[c] = {
            "mean":  (v_mean.tolist()  if v_mean  is not None else None),
            "probe": (v_probe.tolist() if v_probe is not None else None),
        }
    with open(os.path.join(args.out, "cavs_global.json"), "w") as f:
        json.dump(cavs_global, f, indent=2)

    # 6) Build CAVs — per-subset (stratified)
    cavs_per_subset = defaultdict(dict)
    for s in subs:
        idx = [i for i, r in enumerate(rows) if r.subset == s]
        if len(idx) < 50:   # tiny subsets: skip
            continue
        Xs, Zs = X[idx], Z[idx]
        for c in CONCEPTS:
            ys = Y[c][idx]
            v_mean  = cav_mean_diff(Xs, ys)
            v_probe = cav_probe(Zs, BH, ys)   # use same BH; Zs already in SVD space
            cavs_per_subset[s][c] = {
                "mean":  (v_mean.tolist()  if v_mean  is not None else None),
                "probe": (v_probe.tolist() if v_probe is not None else None),
            }
    with open(os.path.join(args.out, "cavs_per_subset.json"), "w") as f:
        json.dump(cavs_per_subset, f, indent=2)

    # 7) Little report
    print("\n[done]")
    print(f"Rows used: {len(rows)}  |  Embedding X: {X.shape}  |  Basis BH: {BH.shape}")
    print(f"Saved: prevalence_by_subset.csv, cavs_global.json, cavs_per_subset.json")

if __name__ == "__main__":
    main()
