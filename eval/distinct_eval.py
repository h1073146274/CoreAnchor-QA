import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

def load_any(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"unsupported file type: {path.suffix}")

def normalize_qa_rows(loaded) -> List[Dict[str, Any]]:
    if isinstance(loaded, list):
        if loaded and isinstance(loaded[0], dict):
            return loaded
        raise ValueError("input list must contain dict items")
    if isinstance(loaded, dict):
        if isinstance(loaded.get("chunk_results"), list):
            rows = []
            for ch in loaded["chunk_results"]:
                for r in ch.get("accepted_results") or []:
                    rows.append(dict(r))
            if not rows:
                raise ValueError("no accepted_results found in chunk_results")
            return rows
        rows = []
        for v in loaded.values():
            if isinstance(v, list):
                rows.extend([x for x in v if isinstance(x, dict)])
        if rows:
            return rows
    raise ValueError("cannot extract QA rows from JSON")

def char_tokens(text: str) -> List[str]:
    return [ch for ch in (text or "").strip() if not ch.isspace()]

def ngram_list(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if n>=1 else []

_Q_TEMPLATE_PREFIX = re.compile(r"^(please|according to|based on|below|following|combine|in the text|in the passage).{0,64}?[：:，,]?", re.I)
_PUNCS = re.compile(r"[\s,.;:!?，。；：！？”“‘’、（）()【】《》—\-]+")
_STOP_CHARS = set(list("theofandtoaisareforfromwithbyonthisthatthesethosean aornorbutsoyetasatinintoontooverunderaboutacrossaroundafterbeforebehindbelowabovebetweenbeyonddespiteduringexceptinsideoutsideperplusminusviavsthanthenelsebotheitherneithernotonlyalsoveryratherquitejustevenstillalreadyeverneverupdownoutoffownsamesuchmanymuchmoremostsomeanyeacheveryfewlittleseveralwhosewhomwhowhichwhatwhenwherewhyhow"))

def _strip_template(q: str) -> str:
    q2 = _Q_TEMPLATE_PREFIX.sub("", q or "")
    q2 = _PUNCS.sub("", q2)
    return q2.strip()

def _keep_content_chars(q: str) -> str:
    q = _strip_template(q)
    return "".join([ch for ch in q if ch not in _STOP_CHARS])

def _remove_answer_chars(q: str, ans: str) -> str:
    keyset = set([ch for ch in (ans or "") if not ch.isspace()])
    return "".join([ch for ch in (q or "") if ch not in keyset])

def _normalize_for_diversity(text: str, counterpart: str, mode: str) -> str:
    if mode == "raw":
        return (text or "").strip()
    if mode == "strip_template":
        return _strip_template(text or "")
    if mode == "content":
        return _keep_content_chars(text or "")
    if mode == "conditional":
        return _remove_answer_chars(_strip_template(text or ""), counterpart or "")
    return (text or "").strip()

def _distinct_n(texts: List[str], n=2) -> float:
    all_ngrams, total = set(), 0
    for t in texts:
        toks = char_tokens(t)
        if len(toks) < n:
            continue
        grams = ngram_list(toks, n)
        total += len(grams)
        all_ngrams.update(grams)
    return (len(all_ngrams) / total) if total else 0.0

def compute_distinct_bundle(df: pd.DataFrame, *, target: str = "question", n: int = 2, group_by: str = "chunk_id") -> Dict[str, float]:
    tgt_col = "question" if target == "question" else "answer"
    cnt_col = "answer" if target == "question" else "question"
    texts = df.get(tgt_col, pd.Series([], dtype=str)).astype(str).tolist()
    cntp  = df.get(cnt_col, pd.Series([""]*len(texts), dtype=str)).astype(str).tolist()
    gids  = df.get(group_by, pd.Series([""]*len(texts), dtype=str)).astype(str).tolist()
    vals_raw = [_normalize_for_diversity(t, c, "raw") for t, c in zip(texts, cntp)]
    d_raw = _distinct_n(vals_raw, n=n)
    vals_content = [_normalize_for_diversity(t, c, "content") for t, c in zip(texts, cntp)]
    d_content = _distinct_n(vals_content, n=n)
    vals_cond = [_normalize_for_diversity(t, c, "conditional") for t, c in zip(texts, cntp)]
    d_cond = _distinct_n(vals_cond, n=n)
    bucket = {}
    for t, g in zip(texts, gids):
        bucket.setdefault(g, []).append(_normalize_for_diversity(t, "", "strip_template"))
    d_macro = 0.0
    if bucket:
        per = []
        for _, arr in bucket.items():
            per.append(_distinct_n(arr, n=n))
        d_macro = float(np.mean(per)) if per else 0.0
    name_prefix = f"distinct_{target}-{n}"
    return {
        f"{name_prefix}_raw": d_raw,
        f"{name_prefix}_content": d_content,
        f"{name_prefix}_conditional": d_cond,
        f"{name_prefix}_macro_by_{group_by}": d_macro,
    }

def main():
    parser = argparse.ArgumentParser(description="Distinct metrics")
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--group_by", type=str, default="")
    parser.add_argument("--keep", type=str, nargs="+", default=None)
    args = parser.parse_args()

    loaded = load_any(Path(args.qa_file))
    rows = normalize_qa_rows(loaded)
    df = pd.DataFrame(rows)

    if not args.target:
        raise SystemExit("target is required")
    if args.n is None:
        raise SystemExit("n is required")
    if not args.group_by:
        raise SystemExit("group_by is required")
    if args.keep is None:
        raise SystemExit("keep is required")

    keep_set = set(args.keep)
    out_items = []

    if args.target in ("question","both"):
        qb = compute_distinct_bundle(df, target="question", n=args.n, group_by=args.group_by)
        for k, v in qb.items():
            if ("_content" in k and "content" in keep_set) or ("_macro_by_" in k and "macro" in keep_set) or ("_raw" in k and "raw" in keep_set) or ("_conditional" in k and "conditional" in keep_set):
                out_items.append({"metric": k, "score": float(v), "score_pct": float(max(0.0, min(100.0, 100.0 * v)))})

    if args.target in ("answer","both"):
        ab = compute_distinct_bundle(df, target="answer", n=args.n, group_by=args.group_by)
        for k, v in ab.items():
            if ("_content" in k and "content" in keep_set) or ("_macro_by_" in k and "macro" in keep_set) or ("_raw" in k and "raw" in keep_set) or ("_conditional" in k and "conditional" in keep_set):
                out_items.append({"metric": k, "score": float(v), "score_pct": float(max(0.0, min(100.0, 100.0 * v)))})

    out = {"summary": out_items}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[OK] wrote", args.out_file)

if __name__ == "__main__":
    main()
