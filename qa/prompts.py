def prompt_extract_candidates(text: str, keywords: list, n: int) -> str:
    return f"""Extract {n} distinct factual points from the text in decreasing importance.

Rules:
1) Each point must be explicitly stated in the text
2) Each point should be a complete fact or definition
3) Order by importance, most important first
4) Each point 1-2 sentences
5) Prefer points related to the keywords
6) Do not repeat content

Output JSON array only, e.g. ["point1","point2",...].

Keywords: {", ".join(keywords)}
Text:
{text}
Return {n} candidates:"""

def prompt_extract_single(text: str, keywords: list) -> str:
    return f"""Extract the single most important factual point from the text.

Keywords: {", ".join(keywords)}
Text:
{text}
Return one sentence:"""

def prompt_extract_supplement(existing: list, text: str, keywords: list, n: int) -> str:
    return f"""Existing candidates: {existing}
Extract {n} additional, different, less important factual points that do not duplicate the existing ones.
Output JSON array only.

Keywords: {", ".join(keywords)}
Text:
{text}"""

def prompt_generate_questions(text: str, abstract: str, keywords: list, rep_keywords: list, n: int, strategy: str) -> str:
    return f"""Generate {n} concrete, fact-answerable questions based on the body.

Rules:
1) Answers must be directly supported by the body
2) Be specific, not vague
3) Keep questions factual with clear answers
4) Ensure tight relevance to these keywords: {", ".join(rep_keywords)}
5) Strategy: {strategy}

Output JSON array only, e.g. ["q1","q2","q3"].

Abstract: {abstract}
Article keywords: {", ".join(keywords)}

Body:
{text}"""

def prompt_generate_answers(question: str, text: str, candidate_answer: str) -> str:
    return f"""Generate 3 accurate answers to the question using the body only.

Rules:
1) Answers must be grounded in the body, no fabrication
2) Each answer should differ in phrasing and structure
3) Answers should directly address the question

Output JSON array only, e.g. ["a1","a2","a3"].

Question: {question}
Reference answer: {candidate_answer}

Body:
{text}"""

def prompt_score_qa(question: str, answer: str, text: str, abstract: str, keywords: list) -> str:
    return f"""Score the QA pair on a 0-10 scale.

Criteria:
1) Correctness (0-3)
2) Completeness (0-2)
3) Match (0-2)
4) Topical Relevance (0-3)

Return JSON:
{{
  "total_score": 8.5,
  "dimension_scores": {{
    "Correctness": 3,
    "Completeness": 2,
    "Match": 2,
    "TopicalRelevance": 1.5
  }},
  "comment": "short remark",
  "reject_reason": "reason if total_score < 6"
}}

Question: {question}
Answer: {answer}

Body:
{text}

Abstract: {abstract}
Keywords: {", ".join(keywords)}"""
