import os, json, re, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
try:
    from ragas.metrics import Faithfulness as _FaithfulnessClass
except Exception:
    _FaithfulnessClass = None
try:
    from ragas.metrics import AnswerRelevancy as _AnswerRelevancyClass
except Exception:
    _AnswerRelevancyClass = None
try:
    from ragas.metrics import faithfulness as _faithfulness_metric
except Exception:
    _faithfulness_metric = None
try:
    from ragas.metrics import answer_relevancy as _answer_relevancy_metric
except Exception:
    _answer_relevancy_metric = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from config import SILICON_API_KEY, SILICON_API_URL

@dataclass
class Keys:
    question: str = "question"
    answer: str = "answer"
    chunk_text: str = "chunk_text"
    chunk_id: str = "chunk_id"
    uid: str = "id"

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
                    rr = dict(r)
                    if "id" not in rr:
                        rid = rr.get("candidate_id")
                        rr["id"] = f"{rr.get('chunk_id','NA')}-{rid if rid is not None else 'NA'}"
                    rows.append(rr)
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

def load_chunks_map(chunk_path: Optional[str], key_id="id", key_text="content") -> Optional[Dict[str, str]]:
    if not chunk_path:
        return None
    loaded = load_any(Path(chunk_path))
    if isinstance(loaded, list):
        out = {}
        for it in loaded:
            if isinstance(it, dict):
                cid = it.get(key_id) or it.get("chunk_id")
                ctext = it.get(key_text) or it.get("chunk_text") or it.get("content")
                if cid and ctext:
                    out[str(cid)] = str(ctext)
        return out
    if isinstance(loaded, dict):
        return {str(k): str(v) for k, v in loaded.items()}
    raise ValueError("invalid chunks file format")

def to_pct_raw(x):
    try:
        v = float(x)
    except Exception:
        return None
    return float(max(0.0, min(100.0, 100.0 * v)))

def _first_nonempty_str(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, list):
                v = "\n".join([str(x) for x in v if x is not None])
            if isinstance(v, (str, int, float)):
                v = str(v).strip()
                if v:
                    return v
    return None

def _first_present(d: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return None

def make_samples(qa_rows: List[Dict[str, Any]], keys: Keys, chunk_map: Optional[Dict[str, str]] = None) -> Tuple[List[SingleTurnSample], List[Dict[str, Any]]]:
    q_keys = [keys.question, "question", "q", "query", "prompt", "user_input", "instruction", "input", "ask"]
    a_keys = [keys.answer, "answer", "final_answer", "response", "output", "pred", "prediction", "generated_answer", "model_answer", "completion"]
    ctx_keys = [keys.chunk_text, "chunk_text", "context", "contexts", "evidence", "passage", "retrieved_contexts", "retrieved", "source", "sources", "content", "text"]
    cid_keys = [keys.chunk_id, "chunk_id", "context_id", "doc_id", "source_id", "chunkId", "cid", "sid"]
    samples, kept_meta = [], []
    for it in qa_rows:
        if not isinstance(it, dict):
            continue
        uid = it.get(keys.uid)
        q = _first_nonempty_str(it, q_keys)
        a = _first_nonempty_str(it, a_keys)
        ctx_text = _first_nonempty_str(it, ctx_keys)
        if not ctx_text and chunk_map:
            cid_val = _first_present(it, cid_keys)
            if cid_val is not None:
                ctx_text = chunk_map.get(str(cid_val))
        if not q or not a or not ctx_text:
            continue
        samples.append(SingleTurnSample(user_input=q, retrieved_contexts=[ctx_text], response=a))
        kept_meta.append({
            "id": uid, keys.question: q, keys.answer: a,
            "context_text": ctx_text,
            keys.chunk_id: _first_present(it, [keys.chunk_id, "chunk_id", "context_id", "doc_id", "source_id"]),
        })
    return samples, kept_meta

def _is_rate_limit_error(err: Exception) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    return ("429" in msg) or ("rate limit" in msg) or ("tpm" in msg) or ("request was rejected due to rate limiting" in msg)

def _is_transient_server_error(err: Exception) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    return ("503" in msg) or ("50603" in msg) or ("too busy" in msg) or ("temporarily" in msg) or ("timeout" in msg) or ("timed out" in msg) or ("read timeout" in msg)

def _sleep_with_jitter(base_s: float, factor: float, attempt: int, max_s: float, jitter: float = 0.2):
    import random, time
    wait = min(max_s, base_s * (factor ** attempt))
    wait = max(0.0, wait * (1.0 + random.uniform(-jitter, jitter)))
    time.sleep(wait)

def run_evaluation_robust(samples: List[SingleTurnSample], metrics, llm_wrapper, emb_wrapper, *, batch_size: Optional[int], show_progress: bool, chunk_size: int, max_retries: int, backoff_base: float, backoff_factor: float, backoff_max: float, retry_jitter: float = 0.2) -> pd.DataFrame:
    n = len(samples)
    if n == 0:
        return pd.DataFrame()
    all_items = []
    start = 0
    block_idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        block_idx += 1
        dataset_chunk = EvaluationDataset(samples=samples[start:end])
        attempt = 0
        while True:
            try:
                res = evaluate(dataset=dataset_chunk, metrics=metrics, show_progress=False, batch_size=batch_size, llm=llm_wrapper, embeddings=emb_wrapper)
                df_chunk = res.to_pandas() if hasattr(res, "to_pandas") else pd.DataFrame()
                all_items.append(df_chunk)
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}".lower()
                if ("403" in msg) or ("insufficient" in msg):
                    raise
                attempt += 1
                _sleep_with_jitter(backoff_base, backoff_factor, attempt - 1, backoff_max, jitter=retry_jitter)
        start = end
    return pd.concat(all_items, ignore_index=True) if all_items else pd.DataFrame()

def _resolve_metric(name: str):
    name = name.lower()
    if name == "faithfulness":
        if _FaithfulnessClass:
            return _FaithfulnessClass()
        if _faithfulness_metric:
            return _faithfulness_metric
    if name == "answer_relevancy":
        if _AnswerRelevancyClass:
            return _AnswerRelevancyClass()
        if _answer_relevancy_metric:
            return _answer_relevancy_metric
    return None

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--chunks_file", type=str, default="")
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--qa_question_key", type=str, default="")
    parser.add_argument("--qa_answer_key", type=str, default="")
    parser.add_argument("--qa_chunk_id_key", type=str, default="")
    parser.add_argument("--qa_chunk_text_key", type=str, default="")
    parser.add_argument("--chunks_id_key", type=str, default="")
    parser.add_argument("--chunks_text_key", type=str, default="")
    parser.add_argument("--metrics", type=str, nargs="+", default="", choices=["faithfulness", "answer_relevancy"])
    parser.add_argument("--eval_llm_model", type=str, default="")
    parser.add_argument("--emb_model", type=str, default="")
    parser.add_argument("--temperature", type=float, default="")
    parser.add_argument("--max_tokens", type=int, default="")
    parser.add_argument("--batch_size", type=int, default="")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--chunk_size", type=int, default="")
    parser.add_argument("--max_retries", type=int, default="")
    parser.add_argument("--backoff_base", type=float, default="")
    parser.add_argument("--backoff_factor", type=float, default="")
    parser.add_argument("--backoff_max", type=float, default="")
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = SILICON_API_KEY or ""
    os.environ["OPENAI_BASE_URL"] = SILICON_API_URL or ""
    os.environ["OPENAI_API_BASE"] = SILICON_API_URL or ""

    loaded = load_any(Path(args.qa_file))
    qa_rows = normalize_qa_rows(loaded)
    chunk_map = load_chunks_map(args.chunks_file, key_id=(args.chunks_id_key or "id"), key_text=(args.chunks_text_key or "content"))
    keys = Keys(
        question=(args.qa_question_key or "question"),
        answer=(args.qa_answer_key or "answer"),
        chunk_id=(args.qa_chunk_id_key or "chunk_id"),
        chunk_text=(args.qa_chunk_text_key or "chunk_text"),
    )
    samples, kept_meta = make_samples(qa_rows, keys, chunk_map)
    if not samples:
        raise SystemExit("no valid samples")

    llm = ChatOpenAI(model=(args.eval_llm_model or ""), base_url=(SILICON_API_URL or None), temperature=(args.temperature or 0.0), max_tokens=(args.max_tokens or 0), max_retries=6)
    llm_wrapper = LangchainLLMWrapper(llm)
    if not args.emb_model:
        raise SystemExit("emb_model is required")
    hf_emb = HuggingFaceEmbeddings(model_name=args.emb_model)
    emb_wrapper = LangchainEmbeddingsWrapper(hf_emb)

    if not args.metrics:
        raise SystemExit("metrics are required")
    metrics = []
    for name in args.metrics:
        m = _resolve_metric(name)
        if m is not None:
            metrics.append(m)
    if not metrics:
        raise SystemExit("no ragas metrics selected")

    df_items = run_evaluation_robust(
        samples=samples,
        metrics=metrics,
        llm_wrapper=llm_wrapper,
        emb_wrapper=emb_wrapper,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        chunk_size=(args.chunk_size or len(samples)),
        max_retries=(args.max_retries or 1),
        backoff_base=(args.backoff_base or 1.0),
        backoff_factor=(args.backoff_factor or 1.0),
        backoff_max=(args.backoff_max or 1.0),
    )

    df_detail = pd.DataFrame(kept_meta).reset_index(drop=True)
    if not df_items.empty:
        cols = [c for c in df_items.columns if c in ("faithfulness", "answer_relevancy")]
        df_detail = pd.concat([df_detail, df_items[cols].reset_index(drop=True)], axis=1)

    for col in ["faithfulness", "answer_relevancy"]:
        if col in df_detail.columns:
            df_detail[col + "_pct"] = df_detail[col].apply(to_pct_raw)

    summary = []
    for met in (args.metrics or []):
        if met in df_detail.columns:
            s = pd.to_numeric(df_detail[met], errors="coerce")
            if s.notna().any():
                mean_val = float(s.mean(skipna=True))
                summary.append({"metric": met, "score": mean_val, "score_pct": to_pct_raw(mean_val)})

    out = {"summary": summary, "details": df_detail.to_dict(orient="records")}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[OK] wrote", args.out_file)

if __name__ == "__main__":
    main()
