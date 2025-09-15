import jieba.posseg as pseg
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.tree import Tree
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def deduplicate_keywords(keywords: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for keyword in keywords:
        if keyword not in seen and keyword.strip():
            seen.add(keyword)
            deduped.append(keyword)
    return deduped

def extract_core_keywords_chinese(text: str) -> List[str]:
    words = pseg.cut(text)
    keywords = [w.word for w in words if w.flag in {"n", "vn", "nt", "nz"}]
    return deduplicate_keywords(keywords)

def extract_core_keywords_english(text: str) -> List[str]:
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    chunks = ne_chunk(pos_tags)
    keywords = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            word, tag = chunk
            if tag in ["NN", "NNS", "NNP", "NNPS"]:
                keywords.append(word)
        elif isinstance(chunk, Tree):
            entity = " ".join(c[0] for c in chunk)
            if chunk.label() not in ["GPE", "PERSON"]:
                keywords.append(entity)
    return deduplicate_keywords(keywords)

def extract_core_keywords(text: str) -> List[str]:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return extract_core_keywords_chinese(text)
    else:
        return extract_core_keywords_english(text)

def extract_representative_keywords(text: str, candidate_answer: str, top_k: int = 10) -> List[str]:
    keywords_text = extract_core_keywords(text)
    keywords_answer = extract_core_keywords(candidate_answer)
    keywords_representative = []
    emb_answer = sbert_model.encode([candidate_answer])
    for kw in keywords_text:
        emb_kw = sbert_model.encode([kw])
        sim = util.cos_sim(emb_kw, emb_answer)[0][0].item()
        if sim > 0.4:
            keywords_representative.append(kw)
    keywords_representative = deduplicate_keywords(keywords_representative)
    return keywords_representative[:top_k]

def keyword_cosine_similarity(text1: str, text2: str) -> float:
    keywords1 = extract_core_keywords(text1)
    keywords2 = extract_core_keywords(text2)
    if not keywords1 or not keywords2:
        return 0.0
    emb1 = sbert_model.encode(keywords1)
    emb2 = sbert_model.encode(keywords2)
    sim_matrix = cosine_similarity(emb1, emb2)
    max_sim1 = np.max(sim_matrix, axis=1)
    max_sim2 = np.max(sim_matrix, axis=0)
    return float((np.mean(max_sim1) + np.mean(max_sim2)) / 2)

def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = sbert_model.encode(text1, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0])

def compute_similarity(text1: str, text2: str, weight_keyword: Optional[float] = None, weight_semantic: Optional[float] = None) -> float:
    if weight_keyword is None or weight_semantic is None:
        raise ValueError("weight_keyword and weight_semantic must be provided explicitly.")
    sim_keyword = keyword_cosine_similarity(text1, text2)
    sim_semantic = semantic_similarity(text1, text2)
    return float(weight_keyword) * sim_keyword + float(weight_semantic) * sim_semantic
