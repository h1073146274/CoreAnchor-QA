#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from alignscore import AlignScore
except Exception:
    raise


def load_results_file(path: str) -> List[Dict]:
    import json

    def _map_row(obj: dict) -> Dict:
        cid = obj.get("chunk_id") or obj.get("chunkId") or obj.get("cid")
        q = (obj.get("question") or obj.get("query") or "").strip()
        a = (obj.get("answer")
             or obj.get("generated_answer")
             or obj.get("final_answer")
             or obj.get("prediction")
             or "").strip()
        if not (q or a):
            return {}
        return {"chunk_id": cid, "question": q, "answer": a}

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        out: List[Dict] = []

        if isinstance(data, dict) and "chunk_results" in data:
            for ch in data.get("chunk_results", []):
                cid = ch.get("chunk_id")
                rows = ch.get("accepted_results") or ch.get("candidate_results") or []
                for r in rows:
                    mapped = _map_row({
                        "chunk_id": cid,
                        "question": r.get("question"),
                        "answer": (r.get("final_answer")
                                   or r.get("generated_answer")
                                   or r.get("candidate_answer"))
                    })
                    if mapped:
                        out.append(mapped)
            return out

        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    mapped = _map_row(obj)
                    if mapped:
                        out.append(mapped)
            return out

        if isinstance(data, dict):
            mapped = _map_row(data)
            return [mapped] if mapped else []

    except json.JSONDecodeError:
        out: List[Dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mapped = _map_row(obj)
            if mapped:
                out.append(mapped)
        if out:
            return out
        raise

    return []


def load_anchor_file(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    idx = {}
    for item in arr:
        cid = item.get("id")
        csum = (item.get("summary") or "").strip()
        ckw = item.get("paragraph_keywords") or []
        chunk_anchor = " ".join([csum] + list(ckw))
        meta = item.get("meta") or {}
        dabs = (meta.get("abstract") or "").strip()
        dkw = meta.get("keywords") or []
        doc_anchor = " ".join([dabs] + list(dkw))
        idx[cid] = {"doc_anchor": doc_anchor.strip(), "chunk_anchor": chunk_anchor.strip()}
    return idx


def batch_alignscore(scorer, contexts: List[str], claims: List[str], batch_size: int = 16) -> List[float]:
    assert len(contexts) == len(claims)
    scores = []
    n = len(claims)
    for i in range(0, n, batch_size):
        ctx_batch = contexts[i:i + batch_size]
        clm_batch = claims[i:i + batch_size]
        s = scorer.score(contexts=ctx_batch, claims=clm_batch)
        scores.extend(list(map(float, s)))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--anchors_json", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--device", default="")
    ap.add_argument("--batch_size", type=int, default="")
    ap.add_argument("--tau", type=float, default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--fig_prefix", default="")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_csv_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(out_dir, args.out_csv)
    fig_prefix = args.fig_prefix if os.path.isabs(args.fig_prefix) else os.path.join(out_dir, args.fig_prefix)

    triples = load_results_file(args.results_json)
    anchors = load_anchor_file(args.anchors_json)

    cand_texts, doc_ctx, chunk_ctx, rows = [], [], [], []
    missing = 0
    for t in triples:
        cid = t["chunk_id"]
        anc = anchors.get(cid)
        if not anc:
            missing += 1
            continue
        qa = (t["question"] + "\n" + t["answer"]).strip()
        cand_texts.append(qa)
        doc_ctx.append(anc["doc_anchor"])
        chunk_ctx.append(anc["chunk_anchor"])
        rows.append({"chunk_id": cid, "question": t["question"], "answer": t["answer"]})

    scorer = AlignScore(
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=args.ckpt_path,
        evaluation_mode="nli_sp",
    )

    doc_scores = batch_alignscore(scorer, doc_ctx, cand_texts, args.batch_size)
    chunk_scores = batch_alignscore(scorer, chunk_ctx, cand_texts, args.batch_size)

    df = pd.DataFrame(rows)
    df["align_doc"] = doc_scores
    df["align_chunk"] = chunk_scores
    df["tdr_flag"] = ((df["align_doc"] < args.tau) & (df["align_chunk"] < args.tau)).astype(int)
    df["align_bottleneck"] = df[["align_doc", "align_chunk"]].min(axis=1)

    n = len(df)
    mean_doc = float(df["align_doc"].mean()) if n else 0.0
    mean_chunk = float(df["align_chunk"].mean()) if n else 0.0
    med_doc = float(df["align_doc"].median()) if n else 0.0
    med_chunk = float(df["align_chunk"].median()) if n else 0.0
    q_doc = df["align_doc"].quantile([0.1, 0.25, 0.75, 0.9]).to_dict() if n else {}
    q_chunk = df["align_chunk"].quantile([0.1, 0.25, 0.75, 0.9]).to_dict() if n else {}
    mean_bneck = float(df["align_bottleneck"].mean()) if n else 0.0
    med_bneck = float(df["align_bottleneck"].median()) if n else 0.0
    tdr = df["tdr_flag"].mean() if n else 0.0

    summary = {
        "n_samples": n,
        "tau": args.tau,
        "align_doc": {
            "mean": mean_doc,
            "median": med_doc,
            "p10": float(q_doc.get(0.1, 0.0)),
            "p25": float(q_doc.get(0.25, 0.0)),
            "p75": float(q_doc.get(0.75, 0.0)),
            "p90": float(q_doc.get(0.9, 0.0)),
        },
        "align_chunk": {
            "mean": mean_chunk,
            "median": med_chunk,
            "p10": float(q_chunk.get(0.1, 0.0)),
            "p25": float(q_chunk.get(0.25, 0.0)),
            "p75": float(q_chunk.get(0.75, 0.0)),
            "p90": float(q_chunk.get(0.9, 0.0)),
        },
        "align_bottleneck": {"mean": mean_bneck, "median": med_bneck},
        "TDR": tdr,
    }

    with open(os.path.join(out_dir, "summary_alignscore.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [{
            "n_samples": n,
            "tau": args.tau,
            "align_doc_mean": mean_doc,
            "align_doc_median": med_doc,
            "align_doc_p10": float(q_doc.get(0.1, 0.0)),
            "align_doc_p25": float(q_doc.get(0.25, 0.0)),
            "align_doc_p75": float(q_doc.get(0.75, 0.0)),
            "align_doc_p90": float(q_doc.get(0.9, 0.0)),
            "align_chunk_mean": mean_chunk,
            "align_chunk_median": med_chunk,
            "align_chunk_p10": float(q_chunk.get(0.1, 0.0)),
            "align_chunk_p25": float(q_chunk.get(0.25, 0.0)),
            "align_chunk_p75": float(q_chunk.get(0.75, 0.0)),
            "align_chunk_p90": float(q_chunk.get(0.9, 0.0)),
            "align_bottleneck_mean": mean_bneck,
            "align_bottleneck_median": med_bneck,
            "TDR": tdr,
        }]
    ).to_csv(os.path.join(out_dir, "summary_alignscore.csv"), index=False, encoding="utf-8-sig")

    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    plt.figure()
    plt.hist(df["align_doc"], bins=30, alpha=0.7, label="AlignScore-Doc")
    plt.hist(df["align_chunk"], bins=30, alpha=0.7, label="AlignScore-Chunk")
    plt.xlabel("AlignScore")
    plt.ylabel("Count")
    plt.title("AlignScore Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_hist.png", dpi=200)

    plt.figure()
    plt.scatter(df["align_doc"], df["align_chunk"], s=8)
    plt.axvline(args.tau, linestyle="--")
    plt.axhline(args.tau, linestyle="--")
    plt.xlabel("AlignScore-Doc")
    plt.ylabel("AlignScore-Chunk")
    plt.title("Doc vs Chunk AlignScore")
    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_scatter.png", dpi=200)


if __name__ == "__main__":
    main()
