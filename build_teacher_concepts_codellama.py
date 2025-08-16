#!/usr/bin/env python3
"""
Build a Teacher concept space with CodeLlama.

Pipeline:
1) Load MBPP + HumanEval (spec, solution, tests) and unify into one schema.
2) Auto-tag tests with simple concept heuristics.
3) Encode tests with CodeLlama (hidden activations).
4) Build Teacher basis (TruncatedSVD).
5) Build concept vectors (mean-difference per concept).
6) Save artifacts for downstream use.

Reqs:
  pip install torch transformers accelerate datasets scikit-learn numpy regex

Optional (VRAM saver):
  pip install bitsandbytes  # then enable 4-bit loading below
"""

import os, json, re, math, argparse, random, numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# ---------- CONFIG ----------

DEFAULT_MODEL = "codellama/CodeLlama-7b-Instruct-hf"  # or "codellama/CodeLlama-7b-hf"
RANDOM_SEED = 13
MAX_LEN = 2048
LAYER_IDX = -2            # penultimate layer tends to work well
SVD_K = 64                # teacher basis dim
CONCEPTS = [
    "boundary_values", "invalid_types", "empty_null",
    "negative_paths", "parametrization", "randomized_property",
    "idempotency"
]

# ---------- DATA SCHEMA ----------

@dataclass
class TeacherItem:
    uid: str                 # unique id
    source: str              # e.g., "mbpp", "humaneval"
    spec: str                # natural-language problem description
    entry_point: str         # function name if known ("" if unknown)
    solution: str            # reference solution (if available; optional)
    tests_code: str          # full pytest-like code or assert checks
    concepts: Dict[str, int] # binary tags from heuristics (0/1)

# ---------- UTIL ----------

def set_seed(s=RANDOM_SEED):
    random.seed(s); np.random.seed(s)

def safe_join_tests(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(map(str, x))
    return str(x)

# ---------- HEURISTIC TAGGING ----------

# Light but effective regexes to tag test concepts
RE_INVALID_TYPES  = re.compile(r"pytest\.raises\s*\(\s*TypeError\s*\)", re.I)
RE_ANY_RAISES     = re.compile(r"pytest\.raises\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)", re.I)
RE_PARAMETRIZE    = re.compile(r"@pytest\.mark\.parametrize", re.I)
RE_HYPOTHESIS     = re.compile(r"(from\s+hypothesis|@given\()", re.I)
RE_RANDOM         = re.compile(r"\brandom\.", re.I)
RE_IDEMPOTENCY    = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\(", re.I)  # crude: repeated call detection later

EMPTY_NULL_TOKENS = {'""', "''", "[]", "{}", "()", "None"}
BOUNDARY_LITS     = {"-1", "0", "1"}  # extend as you like

def tag_concepts(test_code: str, spec: str = "", entry_point: str = "") -> Dict[str, int]:
    code = test_code
    tags = {c: 0 for c in CONCEPTS}

    # invalid types / negative paths
    if RE_INVALID_TYPES.search(code): tags["invalid_types"] = 1
    if RE_ANY_RAISES.search(code):    tags["negative_paths"] = 1

    # empty/null
    tags["empty_null"] = 1 if any(tok in code for tok in EMPTY_NULL_TOKENS) else 0

    # parametrization
    if RE_PARAMETRIZE.search(code): tags["parametrization"] = 1

    # randomized/property
    if RE_HYPOTHESIS.search(code) or RE_RANDOM.search(code):
        tags["randomized_property"] = 1

    # boundary values (very crude heuristic)
    if any(lit in code for lit in BOUNDARY_LITS):
        tags["boundary_values"] = 1
    # also try to pick up min/max/len-based boundaries
    if re.search(r"(min|max|len)\s*\(", code):
        tags["boundary_values"] = 1

    # idempotency: same function called twice on same args then compared
    if entry_point:
        patt = re.compile(fr"{re.escape(entry_point)}\s*\(", re.I)
        calls = len(patt.findall(code))
        if calls >= 2 and re.search(r"assert\s+.+==.+", code):
            tags["idempotency"] = 1
    else:
        # fallback heuristic: generic double call + equality
        if len(RE_IDEMPOTENCY.findall(code)) >= 2 and re.search(r"assert\s+.+==.+", code):
            tags["idempotency"] = 1

    return tags

# ---------- DATA LOADERS (MBPP, HumanEval) ----------

def load_mbpp_split(split="train") -> List[TeacherItem]:
    """Loads MBPP from Hugging Face and converts to TeacherItem.
    Handles both classic MBPP and variants that expose 'test_list' or 'test'.
    """
    from datasets import load_dataset
    ds = load_dataset("mbpp", split=split)  # 'train' has ~974 problems
    items = []
    for r in ds:
        uid = f"mbpp_{r.get('task_id', r.get('id', 'unk'))}"
        spec = r.get("text", r.get("prompt", ""))
        entry_point = r.get("entry_point", "")  # may not exist in MBPP
        solution = r.get("code", r.get("solution", ""))
        tests = r.get("test_list", None)
        tests_code = safe_join_tests(tests) if tests else r.get("test", "")

        # Minimal guard: if tests_code empty, try building from 'test_list' or skip
        if not tests_code:
            continue

        concepts = tag_concepts(tests_code, spec, entry_point)
        items.append(TeacherItem(uid, "mbpp", spec, entry_point, solution, tests_code, concepts))
    return items

def load_humaneval() -> List[TeacherItem]:
    """Loads HumanEval from Hugging Face and converts to TeacherItem."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval")["test"]  # standard split name in HF
    items = []
    for r in ds:
        uid = f"humaneval_{r['task_id']}"
        spec = r.get("prompt", "")
        entry_point = r.get("entry_point", "")
        solution = r.get("canonical_solution", r.get("solution", ""))
        tests_code = r.get("test", "")
        if not tests_code:
            continue
        concepts = tag_concepts(tests_code, spec, entry_point)
        items.append(TeacherItem(uid, "humaneval", spec, entry_point, solution, tests_code, concepts))
    return items

# ---------- (Optional) ADD MORE LOADERS ----------
# Add adapters for KodCode / QuixBugs / BugsInPy here if you have local paths.
# Make sure to return List[TeacherItem] in the same schema.

# ---------- CODELLAMA ACTIVATIONS ----------

def load_causallm(model_name: str = DEFAULT_MODEL):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Uncomment this block to use 4-bit loading (saves VRAM)
    # from transformers import BitsAndBytesConfig
    # quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
    #                                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
    #                                              quantization_config=quant_cfg, output_hidden_states=True)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16,
        output_hidden_states=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def encode_texts_hidden_means(texts: List[str], tok, model, layer_idx: int = LAYER_IDX, max_len: int = MAX_LEN) -> np.ndarray:
    import torch
    vecs = []
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len, padding=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc)
            hs = out.hidden_states[layer_idx]  # (1, seq, dim)
            v = hs.mean(dim=1).squeeze(0)      # (dim,)
            vecs.append(v.float().cpu().numpy())
    return np.stack(vecs, axis=0)

# ---------- SVD + CONCEPT VECTORS ----------

def compute_teacher_basis(X: np.ndarray, k: int = SVD_K) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
    Z = svd.fit_transform(X)           # (N, k)
    B_H = svd.components_.T            # (D, k)
    meta = {"explained_variance": svd.explained_variance_.tolist(),
            "explained_variance_ratio": svd.explained_variance_ratio_.tolist()}
    return B_H, {"svd_meta": meta, "k": k}

def mean_diff_concept(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) < 3 or len(neg) < 3:
        return None
    v = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)

# ---------- MAIN PIPELINE ----------

def main():
    set_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outdir", default="teacher_concepts_out")
    ap.add_argument("--max", type=int, default=0, help="cap number of items (0 = all)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load datasets (extend here as needed)
    print("[1] Loading datasets...")
    items = []
    items += load_mbpp_split("train")
    items += load_humaneval()
    if args.max and len(items) > args.max:
        items = items[:args.max]
    print(f"   loaded {len(items)} items")

    # 2) Save unified corpus (for transparency)
    print("[2] Saving unified corpus...")
    corpus_path = os.path.join(args.outdir, "teacher_corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    print(f"   wrote {corpus_path}")

    # 3) Encode tests with CodeLlama
    print("[3] Loading CodeLlama...")
    tok, model = load_causallm(args.model)

    print("[4] Encoding tests...")
    texts = [it.tests_code for it in items]
    X = encode_texts_hidden_means(texts, tok, model, layer_idx=LAYER_IDX, max_len=MAX_LEN)
    np.save(os.path.join(args.outdir, "teacher_X.npy"), X)
    print(f"   X shape: {X.shape}")

    # 4) Teacher basis
    print("[5] Computing Teacher basis (SVD)...")
    B_H, svd_meta = compute_teacher_basis(X, k=SVD_K)
    np.save(os.path.join(args.outdir, "teacher_basis_BH.npy"), B_H)
    with open(os.path.join(args.outdir, "svd_meta.json"), "w") as f:
        json.dump(svd_meta, f, indent=2)
    print(f"   B_H shape: {B_H.shape}")

    # 5) Concept vectors via mean-difference per tag
    print("[6] Building concept vectors...")
    Y = {c: np.array([it.concepts[c] for it in items], dtype=np.int32) for c in CONCEPTS}
    V = {}
    for c in CONCEPTS:
        vc = mean_diff_concept(X, Y[c])
        if vc is None:
            print(f"   [warn] concept '{c}' skipped (too few positives/negatives)")
            continue
        # Normalize again for safety
        vc = vc / (np.linalg.norm(vc) + 1e-9)
        V[c] = vc.tolist()
    with open(os.path.join(args.outdir, "concept_vectors.json"), "w") as f:
        json.dump({"concepts": CONCEPTS, "vectors": V}, f, indent=2)
    print(f"   saved {len(V)} concept vectors")

    # 6) Quick sanity: top-5 retrieval for each concept
    print("[7] Sanity: precision@5 per concept (heuristic check)")
    def topk_cos(vec, k=5):
        sims = (X @ vec) / (np.linalg.norm(X, axis=1) * (np.linalg.norm(vec) + 1e-9))
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]
    prec = {}
    for c, vec in V.items():
        vec = np.array(vec, dtype=np.float32)
        idx, sims = topk_cos(vec, k=5)
        y = Y[c][idx]
        prec[c] = int(y.sum())
        print(f"   {c:22s}  hits@5 = {int(y.sum())}/5  sims={np.round(sims,3).tolist()}")
    with open(os.path.join(args.outdir, "precision_at5.json"), "w") as f:
        json.dump(prec, f, indent=2)

    print("\nDone. Artifacts:")
    print(f"  - {corpus_path}")
    print(f"  - {os.path.join(args.outdir, 'teacher_X.npy')}")
    print(f"  - {os.path.join(args.outdir, 'teacher_basis_BH.npy')}")
    print(f"  - {os.path.join(args.outdir, 'concept_vectors.json')}")
    print(f"  - {os.path.join(args.outdir, 'precision_at5.json')}")
    print("Use these in your student-diagnosis loop.")
    
if __name__ == "__main__":
    main()
