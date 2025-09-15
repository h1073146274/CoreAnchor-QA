from typing import Dict, List
from similarity_utils import compute_similarity, extract_representative_keywords

def enhanced_compute_similarity_with_keywords(candidate_answer: str, generated_answer: str, chunk_content: str) -> Dict:
    candidate_keywords: List[str] = extract_representative_keywords(chunk_content, candidate_answer)
    generated_keywords: List[str] = extract_representative_keywords(chunk_content, generated_answer)
    similarity_score = compute_similarity(candidate_answer, generated_answer)
    common_keywords = list(set(candidate_keywords) & set(generated_keywords))
    if candidate_keywords and generated_keywords:
        keyword_overlap = len(common_keywords) / max(len(candidate_keywords), len(generated_keywords))
    else:
        keyword_overlap = 0.0
    return {
        "similarity_score": similarity_score,
        "candidate_keywords": candidate_keywords,
        "generated_keywords": generated_keywords,
        "common_keywords": common_keywords,
        "keyword_overlap": keyword_overlap,
        "candidate_answer": candidate_answer,
        "generated_answer": generated_answer
    }
