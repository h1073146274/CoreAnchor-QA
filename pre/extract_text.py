import re
import requests
from typing import List, Dict, Optional
import sys
import os
import json
import uuid
from config import SILICON_API_KEY, SILICON_API_URL

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from split import chunk_markdown_from_file
from datetime import datetime

def _lazy_import_pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except Exception:
        return None

def _lazy_import_docx():
    try:
        import docx
        return docx
    except Exception:
        return None

def call_qwen_for_summary_and_keywords(text: str) -> Dict:
    prompt = f"""You are an expert in summarization and keyword extraction.

If the document has a section named "Abstract", extract the text after that heading until the next heading or a clear break as the abstract.
If it has a section named "Keywords", extract them (comma, semicolon, or newline separated).
Otherwise, write a summary of about 200 words and give 3-5 keywords.

Return exactly:
Abstract: ...
Keywords: ...

Document:
{text}
"""
    headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": ,
        "top_p": ,
        "max_tokens": 
    }
    try:
        resp = requests.post(SILICON_API_URL, headers=headers, json=payload, timeout=60)
        content = resp.json()["choices"][0]["message"]["content"]
        print("[LLM] abstract/keywords response:\n", content)
        abstract_match = re.search(r"Abstract[: ]?\s*(.*)", content, flags=re.IGNORECASE)
        keywords_match = re.search(r"Keywords[: ]?\s*(.*)", content, flags=re.IGNORECASE)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        raw_keywords = keywords_match.group(1).strip() if keywords_match else ""
        keywords = [k.strip() for k in re.split(r"[,\s;]+", raw_keywords) if k.strip()]
        return {"abstract": abstract or "Abstract generation failed", "keywords": keywords}
    except Exception as e:
        print("Failed to generate abstract/keywords:", str(e))
        return {"abstract": "Abstract generation failed", "keywords": []}

def is_valid_content(content: str) -> bool:
    content = (content or "").strip()
    if not content:
        return False
    if re.match(r"^[\d\W_]+$", content):
        return False
    return True

def read_md(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def read_txt(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()

def read_pdf(filepath: str) -> str:
    pdfplumber = _lazy_import_pdfplumber()
    if not pdfplumber:
        print("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def read_docx(filepath: str) -> str:
    docx = _lazy_import_docx()
    if not docx:
        print("python-docx not installed. Run: pip install python-docx")
        return ""
    doc = docx.Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

def read_file_any(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".md":
        return read_md(filepath)
    if ext == ".txt":
        return read_txt(filepath)
    if ext == ".pdf":
        return read_pdf(filepath)
    if ext in (".docx",):
        return read_docx(filepath)
    return read_txt(filepath)

def _split_long(text: str, max_len: int) -> List[str]:
    import re as _re
    paras = _re.split(r"\n\s*\n", text.strip())
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) <= max_len:
            cur += (("\n\n" + p) if cur else p)
        else:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            if len(p) <= max_len:
                cur = p
            else:
                sentences = _re.split(r"(?<=[.!?])", p)
                buf = ""
                for s in sentences:
                    if len(buf) + len(s) <= max_len:
                        buf += s
                    else:
                        if buf:
                            chunks.append(buf.strip())
                        buf = s
                if buf:
                    chunks.append(buf.strip())
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if is_valid_content(c)]

def call_qwen_for_summary_and_keywords_paragraph(text: str) -> Dict:
    prompt = f"""Summarize and extract keywords for the paragraph below.

Return exactly:
Abstract: <=50 characters (if extremely short, the original text is acceptable)
Keywords: k1, k2, k3

Paragraph:
{text}
"""
    headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": ,
        "top_p": ,
        "max_tokens": 
    }
    try:
        resp = requests.post(SILICON_API_URL, headers=headers, json=payload, timeout=60)
        content = resp.json()["choices"][0]["message"]["content"]
        abstract_match = re.search(r"Abstract[: ]?\s*(.*)", content, flags=re.IGNORECASE)
        keywords_match = re.search(r"Keywords[: ]?\s*(.*)", content, flags=re.IGNORECASE)
        abstract = (abstract_match.group(1).strip() if abstract_match else "") or "No abstract"
        raw_keywords = keywords_match.group(1).strip() if keywords_match else ""
        kws = [k.strip() for k in re.split(r"[,\s;]+", raw_keywords) if k.strip()]
        return {"abstract": abstract, "paragraph_keywords": kws[:3]}
    except Exception as e:
        print("Failed to generate paragraph summary:", str(e))
        return {"abstract": "No abstract", "paragraph_keywords": []}

def chunk_plain_text(text: str, min_chars: int = 1000, max_chars: int = 2000) -> List[Dict]:
    text = (text or "").strip()
    if not text:
        return []
    units = _split_long(text, max_len=max_chars)
    chunks = []
    buf = ""
    for u in units:
        if len(buf) + len(u) < min_chars:
            buf += (("\n\n" + u) if buf else u)
        else:
            if buf:
                chunks.append(buf)
                buf = u
            else:
                chunks.append(u)
    if buf:
        chunks.append(buf)
    out = []
    for c in chunks:
        summa = call_qwen_for_summary_and_keywords_paragraph(c)
        out.append({
            "content": c,
            "summary": f"Document body \nAbstract: {summa.get('abstract', 'No abstract')}",
            "paragraph_keywords": summa.get("paragraph_keywords", []),
            "length": len(c),
        })
    return out

def add_meta_to_chunks(chunks: List[Dict], meta: Dict, fileId: str, fileName: str) -> List[Dict]:
    enriched = []
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        summary = chunk.get("summary") or ("No abstract" if content else "Empty content")
        kws = chunk.get("paragraph_keywords") or []
        enriched.append({
            "id": str(uuid.uuid4()),
            "fileId": fileId,
            "fileName": fileName,
            "content": content,
            "summary": summary,
            "paragraph_keywords": kws,
            "length": chunk.get("length", len(content)),
            "meta": {
                "abstract": meta.get("abstract", "No abstract"),
                "keywords": meta.get("keywords", [])
            }
        })
    return enriched

def append_chunks_to_global_file(chunks: List[Dict], global_output_path: str):
    dir_path = os.path.dirname(global_output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(global_output_path):
        with open(global_output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.extend(chunks)
    with open(global_output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print("[OK] appended to:", global_output_path)

def process_file(filepath: str, fileId: Optional[str], global_output_path: str,
                 min_chars=1000, max_chars=2000) -> Dict:
    print("Processing file:", filepath)
    fileName = os.path.basename(filepath)
    ext = os.path.splitext(fileName)[1].lower()
    fileId = fileId or str(uuid.uuid4())
    if ext == ".md":
        chunks = chunk_markdown_from_file(filepath, min_chars, max_chars)
    else:
        raw = read_file_any(filepath)
        chunks = chunk_plain_text(raw, min_chars=min_chars, max_chars=max_chars)
    full_text = "\n\n".join(c.get("content", "") for c in chunks)
    meta = call_qwen_for_summary_and_keywords(full_text)
    enriched = add_meta_to_chunks(chunks, meta, fileId, fileName)
    append_chunks_to_global_file(enriched, global_output_path)
    return {"fileId": fileId, "fileName": fileName, "meta": meta, "num_chunks": len(enriched)}

def process_directory(dirpath: str, output_path: str, min_chars=1000, max_chars=2000):
    exts = (".md", ".txt", ".pdf", ".docx")
    files = [
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f)) and os.path.splitext(f)[1].lower() in exts
    ]
    print("Found files:", len(files))
    results = []
    for fp in files:
        res = process_file(fp, fileId=None, global_output_path=output_path,
                           min_chars=min_chars, max_chars=max_chars)
        results.append(res)
    report_path = os.path.splitext(output_path)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[OK] summary report:", report_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-format extraction + unified chunking (md/txt/pdf/docx)")
    parser.add_argument("path", help="Input file or directory")
    parser.add_argument("--fileId", help="Optional unique file ID")
    parser.add_argument("--min", type=int, default=1000, help="Minimum chunk length")
    parser.add_argument("--max", type=int, default=2000, help="Maximum chunk length")
    parser.add_argument("--global_output", default="all_chunks.json", help="Global output JSON")
    args = parser.parse_args()
    if os.path.isdir(args.path):
        process_directory(args.path, output_path=args.global_output, min_chars=args.min, max_chars=args.max)
    else:
        process_file(args.path, fileId=args.fileId, global_output_path=args.global_output,
                     min_chars=args.min, max_chars=args.max)
