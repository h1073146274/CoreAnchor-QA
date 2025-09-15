import os, argparse, json, re, time, torch
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain_openai import ChatOpenAI
from config import SILICON_API_KEY, SILICON_API_URL

_UNIEVAL_SYS = "You are a strict evaluator. Score on a specific criterion. Output a single integer in [1,5] with no extra text."

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

def to_pct_raw(x):
    try:
        v = float(x)
    except Exception:
        return None
    return float(max(0.0, min(100.0, 100.0 * v)))

def _unieval_prompt(dim: str, q: str, a: str, c: Optional[str], with_ctx_for_coh: bool) -> str:
    if dim == "nat":
        return f"Dimension: NAT.\nScore the answer from 1 to 5.\nAnswer:\n{a}\n"
    if dim == "und":
        return f"Dimension: UND.\nScore the answer from 1 to 5.\nAnswer:\n{a}\n"
    if dim == "coh":
        if with_ctx_for_coh:
            return f"Dimension: COH.\nConsider question and context. Score 1..5.\nQuestion:\n{q}\n\nContext:\n{c or ''}\n\nAnswer:\n{a}\n"
        else:
            return f"Dimension: COH.\nConsider question only. Score 1..5.\nQuestion:\n{q}\n\nAnswer:\n{a}\n"
    return f"Score 1-5: {a}"

def _dtype_from_str(name: str):
    name = (name or "auto").lower()
    if name == "auto": return "auto"
    if name == "float16": return torch.float16
    if name == "bfloat16": return torch.bfloat16
    if name == "float32": return torch.float32
    return "auto"

@torch.no_grad()
def _hf_first_token_logits(prompts: List[str], tok, model, *, batch_size: int = 8, input_max_len: int = 1024) -> List[torch.Tensor]:
    outs = []
    dev = model.device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=input_max_len).to(dev)
        try:
            out = model.generate(**enc, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)
            step = out.scores[0]
            for r in range(step.shape[0]):
                outs.append(step[r].detach().cpu())
        except Exception:
            outputs = model(**enc, use_cache=False)
            logits = outputs.logits
            for r in range(logits.shape[0]):
                outs.append(logits[r, -1, :].detach().cpu())
    return outs

def _map_logits_to_unit(tok, logits_batch: List[torch.Tensor]) -> List[float]:
    base_digits = ["1", "2", "3", "4", "5"]
    prefixes = ["", "▁", " "]
    cand_map = {}
    for i, d in enumerate(base_digits, start=1):
        for pref in prefixes:
            ids = tok.encode(pref + d, add_special_tokens=False)
            if len(ids) == 1:
                cand_map[ids[0]] = i
    if not cand_map:
        return [0.0] * len(logits_batch)
    tids = torch.tensor(list(cand_map.keys()))
    n_of_tid = [cand_map[t.item()] for t in tids]
    scores = []
    for logit in logits_batch:
        sub = logit[tids]
        prob = torch.softmax(sub, dim=-1)
        j = int(torch.argmax(prob).item())
        n = n_of_tid[j]
        scores.append((n - 1) / 4.0)
    return scores

def _load_hf_unieval_model(path: str, device: Optional[str], dtype_str: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"missing local model: {path}")
    dtype = _dtype_from_str(dtype_str)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=(dtype if dtype != "auto" else None))
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=(dtype if dtype != "auto" else None))
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    if hasattr(model.config, "is_encoder_decoder") and model.config.is_encoder_decoder:
        if getattr(model.config, "decoder_start_token_id", None) is None:
            model.config.decoder_start_token_id = getattr(tok, "bos_token_id", None) or tok.eos_token_id or model.config.pad_token_id
    if device:
        model = model.to(device)
    elif torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return tok, model

def _score_to_unit(s: str) -> float:
    if not s:
        return 0.0
    s = str(s).strip()
    s = s.translate(str.maketrans("１２３４５０", "123450"))
    m = re.search(r"([1-5])", s)
    if m:
        n = int(m.group(1))
        return (n - 1) / 4.0
    zhmap = {"一":1, "二":2, "三":3, "四":4, "五":5}
    for ch in s:
        if ch in zhmap:
            n = zhmap[ch]
            return (n - 1) / 4.0
    return 0.0

def run_unieval(df_detail: pd.DataFrame, args, llm_for_default) -> pd.DataFrame:
    dims = set(args.unieval_dims)
    nat_list, und_list, coh_list = [], [], []
    nat_prompts, und_prompts, coh_prompts = [], [], []
    for _, row in df_detail.iterrows():
        q = str(row.get("question", "") or "")
        a = str(row.get("answer", "") or "")
        c = str(row.get("context_text", "") or "")
        if "nat" in dims:
            nat_prompts.append(_unieval_prompt("nat", q, a, c, args.coh_with_context))
        if "und" in dims:
            und_prompts.append(_unieval_prompt("und", q, a, c, args.coh_with_context))
        if "coh" in dims:
            coh_prompts.append(_unieval_prompt("coh", q, a, c, args.coh_with_context))

    if args.unieval_backend == "hf":
        local_path = args.unieval_local_path or args.unieval_model
        if not local_path:
            raise ValueError("backend=hf requires --unieval_local_path or --unieval_model pointing to a local folder")
        tok, mdl = _load_hf_unieval_model(local_path, args.unieval_device, args.unieval_dtype)
        if "nat" in dims:
            nat_logits = _hf_first_token_logits(nat_prompts, tok, mdl, batch_size=(args.unieval_batch_size or 1), input_max_len=(args.unieval_input_max_len or 256))
            nat_list = _map_logits_to_unit(tok, nat_logits)
        if "und" in dims:
            und_logits = _hf_first_token_logits(und_prompts, tok, mdl, batch_size=(args.unieval_batch_size or 1), input_max_len=(args.unieval_input_max_len or 256))
            und_list = _map_logits_to_unit(tok, und_logits)
        if "coh" in dims:
            coh_logits = _hf_first_token_logits(coh_prompts, tok, mdl, batch_size=(args.unieval_batch_size or 1), input_max_len=(args.unieval_input_max_len or 256))
            coh_list = _map_logits_to_unit(tok, coh_logits)
    else:
        judge = ChatOpenAI(model=(args.unieval_model or getattr(llm_for_default, "model_name", None) or ""), base_url=os.environ.get("OPENAI_BASE_URL"), temperature=0.0, max_tokens=16, max_retries=0)
        def _invoke_with_retry(p):
            max_attempts = args.unieval_max_retries if args.unieval_max_retries and args.unieval_max_retries > 0 else 1
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = judge.invoke([{"role": "system", "content": _UNIEVAL_SYS}, {"role": "user", "content": p}])
                    return getattr(resp, "content", "") or ""
                except Exception as e:
                    time.sleep(1.0 + 0.2 * attempt)
            return ""
        if "nat" in dims:
            for p in nat_prompts:
                text = _invoke_with_retry(p)
                nat_list.append(_score_to_unit(text))
                if args.unieval_sleep and args.unieval_sleep > 0:
                    time.sleep(args.unieval_sleep)
        if "und" in dims:
            for p in und_prompts:
                text = _invoke_with_retry(p)
                und_list.append(_score_to_unit(text))
                if args.unieval_sleep and args.unieval_sleep > 0:
                    time.sleep(args.unieval_sleep)
        if "coh" in dims:
            for p in coh_prompts:
                text = _invoke_with_retry(p)
                coh_list.append(_score_to_unit(text))
                if args.unieval_sleep and args.unieval_sleep > 0:
                    time.sleep(args.unieval_sleep)

    if "nat" in dims: df_detail["unieval_nat"] = pd.Series(nat_list)
    if "und" in dims: df_detail["unieval_und"] = pd.Series(und_list)
    if "coh" in dims: df_detail["unieval_coh"] = pd.Series(coh_list)
    for col in ["unieval_nat","unieval_und","unieval_coh"]:
        if col in df_detail.columns:
            df_detail[col + "_pct"] = df_detail[col].apply(lambda v: to_pct_raw(v))
    return df_detail

def main():
    parser = argparse.ArgumentParser(description="UniEval evaluation")
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--enable_dims", type=str, nargs="+", default="")
    parser.add_argument("--coh_with_context", action="store_true")
    parser.add_argument("--unieval_backend", type=str, default="")
    parser.add_argument("--eval_llm_model", type=str, default="")
    parser.add_argument("--unieval_model", type=str, default="")
    parser.add_argument("--unieval_max_retries", type=int, default="")
    parser.add_argument("--unieval_sleep", type=float, default="")
    parser.add_argument("--unieval_local_path", type=str, default="")
    parser.add_argument("--unieval_device", type=str, default="")
    parser.add_argument("--unieval_dtype", type=str, default="")
    parser.add_argument("--unieval_batch_size", type=int, default="")
    parser.add_argument("--unieval_input_max_len", type=int, default="")
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = SILICON_API_KEY or ""
    os.environ["OPENAI_BASE_URL"] = SILICON_API_URL or ""
    os.environ["OPENAI_API_BASE"] = SILICON_API_URL or ""

    loaded = load_any(Path(args.qa_file))
    rows = normalize_qa_rows(loaded)
    df = pd.DataFrame(rows)
    df_detail = pd.DataFrame({
        "question": df.get("question", pd.Series([], dtype=str)),
        "answer": df.get("final_answer", df.get("answer", pd.Series([], dtype=str))),
        "context_text": df.get("chunk_text", df.get("context_text", pd.Series([], dtype=str)))
    })
    class _Dummy: pass
    dummy = _Dummy()
    dummy.model_name = args.eval_llm_model or ""
    if not args.enable_dims:
        raise SystemExit("enable_dims is required")
    args.unieval_dims = args.enable_dims
    if not args.unieval_backend:
        raise SystemExit("unieval_backend is required")
    if not args.unieval_dtype:
        args.unieval_dtype = "auto"
    df_out = run_unieval(df_detail, args, dummy)
    summary = []
    for col in ["unieval_nat","unieval_und","unieval_coh"]:
        if col in df_out.columns:
            s = pd.to_numeric(df_out[col], errors="coerce")
            if s.notna().any():
                mean_val = float(s.mean(skipna=True))
                summary.append({"metric": f"{col}_mean", "score": mean_val, "score_pct": to_pct_raw(mean_val)})
    out = {"summary": summary, "details": df_out.to_dict(orient="records")}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[OK] wrote", args.out_file)

if __name__ == "__main__":
    main()
