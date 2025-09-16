import json
import os
import sys
import argparse
from qa_pipeline import process_all_chunks

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate QA pairs from chunks")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="qa_results.json")
    parser.add_argument("--similarity_analysis", default="similarity_analysis.json")
    parser.add_argument("--chars_per_candidate", type=int, default="")
    parser.add_argument("--max_retry", type=int, default="", help)
    parser.add_argument("--threshold", type=float, default="")
    parser.add_argument("--min_score", type=float, default="")
    parser.add_argument("--api_delay", type=float, default="")
    parser.add_argument("--failmem_sim_th", type=float, default="")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("[ERROR] input file does not exist:", args.input)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        try:
            chunks = json.load(f)
            if not isinstance(chunks, list):
                raise ValueError("input must be a JSON array of chunks")
        except Exception as e:
            print("[ERROR] failed to parse input:", e)
            sys.exit(1)

    print("[INFO] loaded", len(chunks), "chunks")
    process_all_chunks(
        chunks,
        output_path=args.output,
        similarity_analysis_path=args.similarity_analysis,
        chars_per_candidate=args.chars_per_candidate,
        max_retry=args.max_retry,
        similarity_threshold=args.threshold,
        min_score=args.min_score,
        api_delay=args.api_delay,
        failmem_sim_th=args.failmem_sim_th
    )

if __name__ == "__main__":
    main()
