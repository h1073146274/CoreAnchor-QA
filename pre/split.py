import re
from typing import List, Dict
import jieba.analyse
import requests
import time
from config import SILICON_API_KEY, SILICON_API_URL

def extract_outline(text: str) -> List[Dict]:
    pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
    return [
        {'level': len(m.group(1)), 'title': m.group(2).strip(), 'position': m.start()}
        for m in pattern.finditer(text)
    ]

def split_by_outline(text: str, outline: List[Dict]) -> List[Dict]:
    if not outline:
        return [{'heading': None, 'level': 0, 'content': text.strip(), 'position': 0}]
    sections = []
    for i, item in enumerate(outline):
        start = item['position']
        end = outline[i + 1]['position'] if i + 1 < len(outline) else len(text)
        content = text[start:end].strip()
        sections.append({
            'heading': item['title'],
            'level': item['level'],
            'content': content,
            'position': item['position']
        })
    return sections

def split_long_section(section: Dict, max_len: int) -> List[str]:
    text = section['content']
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) <= max_len:
            current += ("\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current.strip())
            if len(p) > max_len:
                sentences = re.split(r'(?<=[.!?])', p)
                temp = ""
                for s in sentences:
                    if len(temp) + len(s) <= max_len:
                        temp += s
                    else:
                        if temp:
                            chunks.append(temp.strip())
                        temp = s
                if temp:
                    chunks.append(temp.strip())
            else:
                current = p
    if current:
        chunks.append(current.strip())
    return chunks

def merge_small_sections(sections: List[Dict], min_len: int, max_len: int) -> List[Dict]:
    merged = []
    buffer = None
    for sec in sections:
        text = sec['content'].strip()
        if len(text) >= min_len:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(sec)
        else:
            if buffer:
                buffer['content'] += "\n\n" + text
            else:
                buffer = sec.copy()
                buffer['content'] = text
            if len(buffer['content']) >= min_len:
                merged.append(buffer)
                buffer = None
    if buffer:
        merged.append(buffer)
    return merged

def extract_keywords_from_text(text: str, top_k=5):
    return jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)

def call_qwen_for_summary_and_keywords(text: str) -> Dict:
    if not text.strip():
        print("[WARN] empty text to API")
        return {"abstract": "Empty paragraph", "paragraph_keywords": []}
    prompt = f"""Summarize and extract keywords for the paragraph.

Return exactly:
Abstract: one sentence (<=50 chars; if extremely short, you may return the text itself)
Keywords: k1, k2, k3

Paragraph:
{text}
"""
    headers = {
        "Authorization": f"Bearer {SILICON_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": ,
        "top_p": ,
        "max_tokens": 
    }
    try:
        print(f"[INFO] API input preview: {text[:100]}...")
        response = requests.post(SILICON_API_URL, headers=headers, json=payload, timeout=120)
        print(f"[INFO] API status: {response.status_code}")
        print(f"[INFO] API raw: {response.text[:500]}")
        response_json = response.json()
        if "choices" not in response_json:
            print(f"[ERROR] API error: {response_json.get('error', 'unknown')}")
            return {"abstract": "No abstract", "paragraph_keywords": []}
        content = response_json["choices"][0]["message"]["content"]
        print("[INFO] API content:\n", content)
        abstract_match = re.search(r"Abstract[: ]?\s*(.*)", content, flags=re.IGNORECASE)
        keywords_match = re.search(r"Keywords[: ]?\s*(.*)", content, flags=re.IGNORECASE)
        abstract = abstract_match.group(1).strip() if abstract_match else "No abstract"
        raw_keywords = keywords_match.group(1).strip() if keywords_match else ""
        keywords = [k.strip() for k in re.split(r"[,\s;]+", raw_keywords) if k.strip()]
        if not keywords:
            print("[INFO] fallback to TF-IDF keywords")
            keywords = extract_keywords_from_text(text)
        return {"abstract": abstract, "paragraph_keywords": keywords}
    except Exception as e:
        print(f"[ERROR] API call failed: {str(e)}")
        return {"abstract": "No abstract", "paragraph_keywords": []}

def generate_summary(section: Dict, outline: List[Dict], part_index=None, total_parts=None) -> Dict:
    if not section.get('heading'):
        return {"summary": "Preface", "keywords": []}
    summary_result = call_qwen_for_summary_and_keywords(section['content'])
    summary = summary_result.get('abstract', 'No abstract')
    keywords = summary_result.get('paragraph_keywords', [])
    parents = []
    current_level = section['level']
    for item in reversed(outline):
        if item['position'] < section['position'] and item['level'] < current_level:
            parents.insert(0, item['title'])
            current_level = item['level']
    path = " > ".join(parents + [section['heading']])
    if part_index is not None and total_parts is not None and total_parts > 1:
        path += f" - Part {part_index}/{total_parts}"
    return {"summary": f"{path} \nAbstract: {summary}", "keywords": keywords}

def chunk_markdown(text: str, min_chars: int = 1000, max_chars: int = 2000) -> List[Dict]:
    outline = extract_outline(text)
    raw_sections = split_by_outline(text, outline)
    merged_sections = merge_small_sections(raw_sections, min_chars, max_chars)
    final_chunks = []
    for sec in merged_sections:
        content_clean = sec['content'].strip()
        if not content_clean or re.match(r'^[\d\W_]+$', content_clean):
            print(f"[WARN] skip invalid content: {content_clean[:50]}...")
            final_chunks.append({
                **sec,
                'summary': f"{sec.get('heading', 'Preface')} \nAbstract: Empty or invalid content",
                'paragraph_keywords': [],
                'length': len(sec['content'])
            })
            continue
        if len(content_clean) <= max_chars:
            time.sleep(0.5)
            summary_result = generate_summary(sec, outline)
            final_chunks.append({
                **sec,
                'summary': summary_result['summary'],
                'paragraph_keywords': summary_result['keywords'],
                'length': len(sec['content'])
            })
        else:
            sub_chunks = split_long_section(sec, max_chars)
            for i, sub in enumerate(sub_chunks):
                time.sleep(0.5)
                sec_sub = {
                    'heading': sec['heading'],
                    'level': sec['level'],
                    'content': sub,
                    'position': sec['position']
                }
                summary_result = generate_summary(sec_sub, outline, part_index=i+1, total_parts=len(sub_chunks))
                final_chunks.append({
                    'heading': sec['heading'],
                    'level': sec['level'],
                    'content': sub,
                    'summary': summary_result['summary'],
                    'paragraph_keywords': summary_result['keywords'],
                    'length': len(sub),
                    'position': sec['position']
                })
    return final_chunks

def chunk_markdown_from_file(filepath: str, min_chars: int = 1000, max_chars: int = 2000) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return chunk_markdown(content, min_chars, max_chars)
