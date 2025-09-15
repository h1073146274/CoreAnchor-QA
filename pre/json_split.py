import re
import requests
from typing import List, Dict, Callable, Tuple
import sys
import os
import json
import uuid
from datetime import datetime
import time

from config import SILICON_API_KEY, SILICON_API_URL

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

def detect_language(text: str) -> str:
    if not text:
        return "en"
    cjk = len(_CJK_RE.findall(text))
    latin = len(_ASCII_LETTER_RE.findall(text))
    total = max(len(text), 1)
    if cjk / total >= 0.05 or cjk >= 16:
        return "zh"
    if latin / total >= 0.05 or latin >= 20:
        return "en"
    return "en"

def _build_article_prompt(text: str) -> str:
    return f"""You are an expert in summarization and keyword extraction.
Analyze the article below and produce a concise summary and keywords.

Article:
{text}

Requirements:
1) Write a single-paragraph summary of the core content in about 100–200 words.
2) Generate 3–5 keywords that best capture the core topics.

Return exactly:
Summary: ...
Keywords: keyword1, keyword2, keyword3, ...
"""

def _build_chunk_prompt(text: str) -> str:
    return f"""You are an expert in summarization and keyword extraction for paragraphs.

Paragraph:
{text}

Requirements:
1) Provide a one-sentence summary under 50 words; keep it concise.
2) Provide 2–3 keywords capturing the core idea.

Return exactly:
Summary: ...
Keywords: keyword1, keyword2, ...
"""

def _parse_model_output(raw: str) -> Tuple[str, List[str]]:
    if not raw:
        return ("", [])
    raw_stripped = raw.strip()
    if raw_stripped.startswith("{") and raw_stripped.endswith("}"):
        try:
            obj = json.loads(raw_stripped)
            s = obj.get("summary", "")
            kws = obj.get("keywords", [])
            if isinstance(kws, str):
                kws = re.split(r"[,\n\r\t;]+", kws)
            kws = [k.strip() for k in kws if k and k.strip()]
            return (s.strip(), kws)
        except Exception:
            pass
    sum_en = re.search(r"(?i)Summary[: ]?\s*(.+?)(?:\n[A-Z][a-zA-Z ]*[: ]|$)", raw_stripped, re.DOTALL)
    kw_en  = re.search(r"(?i)Keywords?[: ]?\s*(.+)", raw_stripped)
    summary = sum_en.group(1).strip() if sum_en else ""
    keywords_str = kw_en.group(1).strip() if kw_en else ""
    keywords_str = re.sub(r"(?:^|\n)\s*(?:\d+\.\s*|[-*]\s*)", "\n", keywords_str)
    keywords = []
    if keywords_str:
        keywords = [k.strip(" -\t") for k in re.split(r"[,\n\r\t;]+", keywords_str) if k.strip()]
    return (summary, keywords)

def _post_chat(prompt: str) -> Dict:
    headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": ,
        "top_p": ,
        "max_tokens": 
    }
    resp = requests.post(SILICON_API_URL, headers=headers, json=payload, timeout=120)
    return {"status": resp.status_code, "json": (resp.json() if resp.content else {}), "text": resp.text}

def call_qwen_for_summary_and_keywords(text: str, lang: str = "auto") -> Dict:
    if not text or not text.strip():
        return {"abstract": "Empty content", "keywords": []}
    _ = detect_language(text) if lang == "auto" else lang
    prompt = _build_article_prompt(text)
    try:
        print("[INFO] article prompt input preview:", text[:100], "...")
        res = _post_chat(prompt)
        print("[INFO] status:", res["status"])
        if res["status"] != 200:
            print("[ERROR] http error:", res["text"][:500])
            return {"abstract": "Failed to generate summary", "keywords": []}
        rj = res["json"]
        if "choices" not in rj:
            print("[ERROR] bad response:", rj.get("error", "unknown"))
            return {"abstract": "Failed to generate summary", "keywords": []}
        content = rj["choices"][0]["message"]["content"]
        print("[INFO] model content:\n", content)
        summary, keywords = _parse_model_output(content)
        if not summary:
            summary = "Summary unavailable"
        if not keywords:
            keywords = ["keywords unavailable"]
        return {"abstract": summary, "keywords": keywords}
    except Exception as e:
        print("[ERROR] exception:", str(e))
        return {"abstract": "Failed to generate summary", "keywords": ["keywords unavailable"]}

def call_qwen_for_chunk_summary(text: str, lang: str = "auto") -> Dict:
    if not text or not text.strip():
        return {"abstract": "Empty paragraph", "paragraph_keywords": []}
    _ = detect_language(text) if lang == "auto" else lang
    prompt = _build_chunk_prompt(text)
    try:
        res = _post_chat(prompt)
        rj = res["json"]
        if "choices" not in rj:
            print("[ERROR] bad response:", rj.get("error", "unknown"))
            return {"abstract": "Summary unavailable", "paragraph_keywords": []}
        content = rj["choices"][0]["message"]["content"]
        print("[INFO] model paragraph content:\n", content)
        summary, keywords = _parse_model_output(content)
        if not summary:
            summary = "Summary unavailable"
        if not keywords:
            keywords = ["no keywords"]
        return {"abstract": summary, "paragraph_keywords": keywords}
    except Exception as e:
        print("[ERROR] exception:", str(e))
        return {"abstract": "Summary unavailable", "paragraph_keywords": []}

def split_text_into_chunks(text: str, min_chars: int = 1000, max_chars: int = 2000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += ("\n\n" + paragraph) if current_chunk else paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(paragraph) > max_chars:
                sentences = re.split(r"(?<=[.!?])", paragraph)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= max_chars:
                        temp_chunk += sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                if temp_chunk:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    final_chunks = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        while (len(cur) < min_chars and i + 1 < len(chunks)
               and len(cur) + len(chunks[i + 1]) <= max_chars):
            i += 1
            cur += "\n\n" + chunks[i]
        final_chunks.append(cur)
        i += 1
    return final_chunks

def process_single_article(article: Dict, article_id: str, min_chars: int = 1000, max_chars: int = 2000) -> List[Dict]:
    title = (article.get("title") or "").strip()
    text = (article.get("text") or "").strip()
    if not text:
        print("[WARN] empty article:", title or "Untitled")
        return []
    lang_field = (article.get("language") or "").lower()
    lang = lang_field if lang_field in ("en", "zh") else detect_language(f"{title}\n{text}")
    print("[INFO] processing article:", title or "Untitled", "| lang=", lang)
    full_text = f"{title}\n\n{text}" if title else text
    article_meta = call_qwen_for_summary_and_keywords(full_text, lang=lang)
    time.sleep(0.5)
    text_chunks = split_text_into_chunks(text, min_chars, max_chars)
    result_chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_summary = call_qwen_for_chunk_summary(chunk_text, lang=lang)
        time.sleep(0.5)
        chunk_data = {
            "id": str(uuid.uuid4()),
            "articleId": article_id,
            "title": title,
            "content": chunk_text,
            "summary": chunk_summary.get("abstract", "Summary unavailable"),
            "paragraph_keywords": chunk_summary.get("paragraph_keywords", []),
            "length": len(chunk_text),
            "chunk_index": i + 1,
            "total_chunks": len(text_chunks),
            "meta": {
                "abstract": article_meta.get("abstract", "Summary unavailable"),
                "keywords": article_meta.get("keywords", [])
            },
            "language": lang
        }
        result_chunks.append(chunk_data)
    return result_chunks

def append_chunks_to_global_file(chunks: List[Dict], global_output_path: str):
    dir_path = os.path.dirname(global_output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(global_output_path):
        with open(global_output_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
    else:
        existing = []
    existing.extend(chunks)
    with open(global_output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print("[OK] appended to:", global_output_path)

def process_news_json_file(input_filepath: str, output_path: str = "news_all_chunks.json",
                           min_chars: int = 1000, max_chars: int = 2000):
    print("[INFO] start processing:", input_filepath)
    articles = []
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                articles = json.loads(content)
            else:
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        articles.append(article)
                    except json.JSONDecodeError as e:
                        print("[WARN] bad json at line", line_num, ":", e)
                        continue
    except Exception as e:
        print("[ERROR] read file failed:", e)
        return
    if not articles:
        print("[ERROR] no valid articles found")
        return
    print("[INFO] total articles:", len(articles))
    total_chunks = 0
    processed_articles = 0
    for i, article in enumerate(articles):
        if not isinstance(article, dict) or "text" not in article:
            print("[WARN] record missing 'text' field at index", i + 1)
            continue
        try:
            article_id = str(uuid.uuid4())
            chunks = process_single_article(article, article_id, min_chars, max_chars)
            if chunks:
                append_chunks_to_global_file(chunks, output_path)
                total_chunks += len(chunks)
                processed_articles += 1
                print("[OK] done:", (article.get("title") or "Untitled"), "->", len(chunks), "chunks")
            else:
                print("[WARN] skipped:", (article.get("title") or "Untitled"))
        except Exception as e:
            print("[ERROR] process article failed:", str(e))
            continue
    print("[INFO] finished")
    print("[INFO] stats:")
    print("  processed articles:", processed_articles, "/", len(articles))
    print("  total chunks:", total_chunks)
    print("  output file:", output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process news JSON/JSONL and generate chunks")
    parser.add_argument("input_file", help="Input JSON/JSONL file path")
    parser.add_argument("--output", default="news_all_chunks.json", help="Output JSON file path ")
    parser.add_argument("--min", type=int, default=1000, help="Minimum chunk length")
    parser.add_argument("--max", type=int, default=2000, help="Maximum chunk length")
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        print("[ERROR] input file does not exist:", args.input_file)
        sys.exit(1)
    process_news_json_file(
        input_filepath=args.input_file,
        output_path=args.output,
        min_chars=args.min,
        max_chars=args.max
    )
