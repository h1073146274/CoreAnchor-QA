# CoreAnchor-QA

## 1. Environment Initialization
```bash
conda create -y -n CoreAnchor-QA python=3.10
conda activate CoreAnchor-QA
pip install -r requirements.txt
```

---

## 2. Data Preparation

### a) Multi-format files/dirs → Chunks
```bash
python extract_text.py /path/to/dir   --global_output out/all_chunks.json   --min 800   --max 1600
```

### b) JSON/JSONL → Chunks
```bash
python json_split.py data/articles.jsonl   --output out/news_all_chunks.json   --min 800   --max 1600
```

---

## 3. QA Generation Pipeline
```bash
python main.py --input input.json --output qa_results.json
```

---

## 4. Evaluation

### a) AlignScore Evaluation
> Note: AlignScore must be installed from its GitHub repository.
```bash
python alignscore_anchor_eval.py   --results_json <results.json>   --anchors_json <anchors.json>   --ckpt_path <alignscore_ckpt>   --model roberta-base   --device cuda:0   --out_csv results_alignscore.csv   --fig_prefix figs/alignscore   --out_dir results/
```

### b) UniEval Evaluation
```bash
python unieval_eval.py   --qa_file <qa.json>   --out_file results_unieval.json   --enable_dims nat und coh   --unieval_backend hf   --unieval_local_path <path_to_local_model>   --unieval_device cuda   --unieval_dtype float16
```

### c) Distinct Evaluation
```bash
python distinct_eval.py   --qa_file <qa.json>   --out_file results_distinct.json
```

### d) RAGAS Evaluation
```bash
python ragas_eval.py   --qa_file <qa.json>   --chunks_file <chunks.json>   --out_file results_ragas.json   --metrics faithfulness answer_relevancy   --eval_llm_model <model_name>   --emb_model <embedding_model>
```

---

