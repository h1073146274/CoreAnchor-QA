import json
import re
import time
from typing import List, Dict
from failure_memory import FailureMemory
from llm_client import call_silicon_llm
from prompts import (
    prompt_extract_candidates,
    prompt_extract_single,
    prompt_extract_supplement,
    prompt_generate_questions,
    prompt_generate_answers,
    prompt_score_qa,
)
from parsing import parse_json_array_block
from similarity_enhanced import enhanced_compute_similarity_with_keywords

ANGLE_STRATEGIES = {
    1: "Definition, cause, characteristics, impact",
    2: "Mechanism, process, steps, workflow",
    3: "Applications, use cases, effects, significance",
    4: "Comparison, pros and cons, differences"
}

def extract_multiple_candidates(chunk_content: str, chunk_keywords: List[str], candidate_count: int, api_delay: float = 2.0) -> List[Dict]:
    candidates = []
    prompt = prompt_extract_candidates(chunk_content, chunk_keywords, candidate_count)
    res = call_silicon_llm(prompt, delay=api_delay)
    answers = parse_json_array_block(res)
    if not answers:
        single = call_silicon_llm(prompt_extract_single(chunk_content, chunk_keywords), delay=api_delay).strip()
        answers = [single] if single else []
    if len(answers) < candidate_count:
        remain = candidate_count - len(answers)
        sup = call_silicon_llm(prompt_extract_supplement(answers, chunk_content, chunk_keywords, remain), delay=api_delay)
        supp = parse_json_array_block(sup)
        answers.extend(supp[:remain])
    answers = answers[:candidate_count]
    for i, a in enumerate(answers):
        if a and a.strip():
            candidates.append({
                "candidate_id": i,
                "candidate_answer": a.strip(),
                "importance_rank": i + 1,
                "extraction_method": "semantic_importance",
                "source_content": (chunk_content[:200] + "...") if len(chunk_content) > 200 else chunk_content
            })
    if not candidates:
        candidates.append({
            "candidate_id": 0,
            "candidate_answer": "No candidate extracted",
            "importance_rank": 1,
            "extraction_method": "fallback",
            "source_content": (chunk_content[:200] + "...") if len(chunk_content) > 200 else chunk_content
        })
    return candidates

def build_question_generation_prompt(text: str, abstract: str, keywords: List[str], rep_keywords: List[str], number: int, attempt_round: int) -> str:
    strategy = ANGLE_STRATEGIES.get(attempt_round, ANGLE_STRATEGIES[1])
    return prompt_generate_questions(text, abstract, keywords, rep_keywords, number, strategy)

def build_answer_generation_prompt(question: str, chunk_text: str, candidate_answer: str) -> str:
    return prompt_generate_answers(question, chunk_text, candidate_answer)

def build_qa_pair_scoring_prompt(question: str, answer: str, chunk_text: str, abstract: str, keywords: List[str]) -> str:
    return prompt_score_qa(question, answer, chunk_text, abstract, keywords)

def process_candidate_qa_generation(candidate_info: Dict, chunk_content: str, article_summary: str, article_keywords: List[str], max_retry: int, similarity_threshold: float, min_score: float, early_stop_threshold: int = 2, api_delay: float = 2.0, failmem_sim_th: float = 0.90) -> Dict:
    from similarity_utils import extract_representative_keywords
    failure_mem = FailureMemory(sim_threshold=failmem_sim_th)
    candidate_answer = candidate_info["candidate_answer"]
    candidate_id = candidate_info["candidate_id"]
    result = {
        "candidate_id": candidate_id,
        "candidate_answer": candidate_answer,
        "importance_rank": candidate_info.get("importance_rank", candidate_id + 1),
        "extraction_method": candidate_info.get("extraction_method", "semantic_importance"),
        "success": False,
        "attempt_rounds": 0,
        "qualified_qa_pairs": [],
        "all_attempts": [],
        "similarity_details": [],
        "final_reject_reason": ""
    }
    question_best_pairs = {}
    for attempt_round in range(1, max_retry + 1):
        result["attempt_rounds"] = attempt_round
        rep_keywords = extract_representative_keywords(chunk_content, candidate_answer)
        q_prompt = build_question_generation_prompt(chunk_content, article_summary, article_keywords, rep_keywords, number=3, attempt_round=attempt_round)
        q_json = call_silicon_llm(q_prompt, delay=api_delay)
        try:
            questions = parse_json_array_block(q_json)
        except Exception:
            questions = []
        if not questions:
            continue
        filtered_questions, dropped = failure_mem.filter(questions)
        if not filtered_questions:
            filtered_questions = questions[:1]
        questions = filtered_questions
        round_qa_pairs = []
        round_qualified_count = 0
        round_similarity_details = []
        for question in questions:
            a_prompt = build_answer_generation_prompt(question, chunk_content, candidate_answer)
            a_json = call_silicon_llm(a_prompt, delay=api_delay)
            try:
                answers = parse_json_array_block(a_json)
                if not answers:
                    answers = [candidate_answer]
                answers = answers[:3]
            except Exception:
                answers = [candidate_answer]
            question_best_pair = None
            question_best_score = 0.0
            for answer in answers:
                sim_detail = enhanced_compute_similarity_with_keywords(candidate_answer, answer, chunk_content)
                round_similarity_details.append({"question": question, "attempt_round": attempt_round, **sim_detail})
                s_prompt = build_qa_pair_scoring_prompt(question, answer, chunk_content, article_summary, article_keywords)
                s_json = call_silicon_llm(s_prompt, delay=api_delay)
                try:
                    score_match = re.search(r"\{.*\}", s_json, re.DOTALL)
                    score_result = json.loads(score_match.group(0)) if score_match else {"total_score": 5.0, "dimension_scores": {}, "comment": "parse-failed"}
                    total_score = float(score_result.get("total_score", 0))
                    similarity = float(sim_detail["similarity_score"])
                    pair_info = {
                        "question": question,
                        "answer": answer,
                        "total_score": total_score,
                        "dimension_scores": score_result.get("dimension_scores", {}),
                        "comment": score_result.get("comment", ""),
                        "similarity_with_candidate": similarity,
                        "similarity_detail": sim_detail,
                        "attempt_round": attempt_round,
                        "is_qualified": total_score >= min_score and similarity >= similarity_threshold
                    }
                    round_qa_pairs.append(pair_info)
                    if pair_info["is_qualified"] and total_score > question_best_score:
                        question_best_score = total_score
                        question_best_pair = pair_info
                except Exception:
                    continue
            if question_best_pair and question_best_pair["is_qualified"]:
                if question not in question_best_pairs or question_best_pair["total_score"] > question_best_pairs[question]["total_score"]:
                    question_best_pairs[question] = question_best_pair
                    round_qualified_count += 1
            else:
                failure_mem.add(question)
        attempt_result = {
            "attempt_round": attempt_round,
            "questions_generated": len(questions),
            "qa_pairs_evaluated": len(round_qa_pairs),
            "qualified_pairs_in_round": round_qualified_count,
            "total_qualified_questions": len(question_best_pairs),
            "qa_pairs": round_qa_pairs,
            "failmem_dropped_count": len(dropped) if 'dropped' in locals() and dropped else 0,
            "failmem_threshold": failure_mem.sim_threshold
        }
        result["all_attempts"].append(attempt_result)
        result["similarity_details"].extend(round_similarity_details)
        should_early_stop = False
        if len(question_best_pairs) >= early_stop_threshold:
            should_early_stop = True
            early_stop_reason = f"enough qualified pairs: {len(question_best_pairs)}"
        elif len(questions) > 0 and len(question_best_pairs) >= len(questions):
            should_early_stop = True
            early_stop_reason = "qualified answers found for all questions in this round"
        elif attempt_round >= 2 and len(question_best_pairs) >= 1:
            recent = result["all_attempts"][-1:] if result["all_attempts"] else []
            if recent and recent[0].get("qualified_pairs_in_round", 0) == 0:
                should_early_stop = True
                early_stop_reason = "no new qualified pairs in recent round"
        if should_early_stop:
            break
    result["qualified_qa_pairs"] = list(question_best_pairs.values())
    result["success"] = len(result["qualified_qa_pairs"]) > 0
    if not result["success"]:
        if result["all_attempts"]:
            all_pairs = []
            for attempt in result["all_attempts"]:
                all_pairs.extend(attempt["qa_pairs"])
            if not all_pairs:
                result["final_reject_reason"] = "no qa pairs generated"
            else:
                best_score = max([p["total_score"] for p in all_pairs]) if all_pairs else 0
                best_similarity = max([p["similarity_with_candidate"] for p in all_pairs]) if all_pairs else 0
                qualified_count = len([p for p in all_pairs if p["is_qualified"]])
                if qualified_count == 0:
                    result["final_reject_reason"] = f"insufficient score and/or similarity (best_score={best_score:.1f}, best_similarity={best_similarity:.3f})"
                else:
                    result["final_reject_reason"] = "qualified pairs existed but were not recognized"
        else:
            result["final_reject_reason"] = "no attempts generated"
    return result

def process_chunk_for_qa(chunk: Dict, chars_per_candidate: int = 240, max_retry: int = 3, similarity_threshold: float = 0.75, min_score: float = 6.0, early_stop_threshold: int = 2, api_delay: float = 2.0, failmem_sim_th: float = 0.90) -> Dict:
    chunk_id = chunk["id"]
    content = chunk["content"]
    chunk_summary = chunk["summary"]
    chunk_keywords = chunk["paragraph_keywords"]
    article_summary = chunk["meta"]["abstract"]
    article_keywords = chunk["meta"]["keywords"]
    candidate_count = max(1, len(content) // chars_per_candidate)
    print(f"[INFO] chunk {chunk_id} length={len(content)}, candidates={candidate_count}")
    candidates = extract_multiple_candidates(content, chunk_keywords, candidate_count, api_delay)
    candidate_results = []
    successful_candidates = 0
    total_qa_pairs_generated = 0
    accepted_qa_pairs = []
    all_similarity_details = []
    for i, cand in enumerate(candidates):
        print(f"[INFO] candidate {i+1}/{len(candidates)}")
        cres = process_candidate_qa_generation(
            cand, content, article_summary, article_keywords,
            max_retry, similarity_threshold, min_score,
            early_stop_threshold, api_delay, failmem_sim_th
        )
        candidate_results.append(cres)
        all_similarity_details.extend(cres["similarity_details"])
        for attempt in cres["all_attempts"]:
            total_qa_pairs_generated += attempt["qa_pairs_evaluated"]
        if cres["success"]:
            successful_candidates += 1
            for qp in cres["qualified_qa_pairs"]:
                accepted_qa_pairs.append({
                    "chunk_id": chunk_id,
                    "candidate_id": cres["candidate_id"],
                    "importance_rank": cand.get("importance_rank", i+1),
                    "question": qp["question"],
                    "candidate_answer": cres["candidate_answer"],
                    "final_answer": qp["answer"],
                    "similarity": qp["similarity_with_candidate"],
                    "similarity_detail": qp["similarity_detail"],
                    "total_score": qp["total_score"],
                    "dimension_scores": qp["dimension_scores"],
                    "comment": qp["comment"],
                    "attempt_round": qp["attempt_round"],
                    "extraction_method": cand.get("extraction_method", "semantic_importance"),
                    "status": "accepted"
                })
        else:
            pass
    candidate_success_rate = (successful_candidates / len(candidates) * 100) if candidates else 0
    avg_attempts = sum([cr["attempt_rounds"] for cr in candidate_results]) / len(candidate_results) if candidate_results else 0
    avg_pairs_per_success = (len(accepted_qa_pairs) / successful_candidates) if successful_candidates > 0 else 0
    return {
        "chunk_id": chunk_id,
        "chunk_summary": chunk_summary,
        "keywords": chunk_keywords,
        "chunk_length": len(content),
        "chars_per_candidate": chars_per_candidate,
        "all_similarity_details": all_similarity_details,
        "statistics": {
            "total_candidates_generated": len(candidates),
            "successful_candidates": successful_candidates,
            "failed_candidates": len(candidates) - successful_candidates,
            "candidate_success_rate_percent": round(candidate_success_rate, 2),
            "total_qa_pairs_evaluated": total_qa_pairs_generated,
            "accepted_qa_pairs": len(accepted_qa_pairs),
            "average_attempts_per_candidate": round(avg_attempts, 2),
            "average_pairs_per_successful_candidate": round(avg_pairs_per_success, 2)
        },
        "candidate_results": candidate_results,
        "accepted_results": accepted_qa_pairs
    }

def generate_keyword_statistics(similarity_analysis: List[Dict]) -> Dict:
    from collections import Counter
    all_candidate_keywords, all_generated_keywords, all_common_keywords = [], [], []
    for chunk_data in similarity_analysis:
        for detail in chunk_data["similarity_details"]:
            all_candidate_keywords.extend(detail["candidate_keywords"])
            all_generated_keywords.extend(detail["generated_keywords"])
            all_common_keywords.extend(detail["common_keywords"])
    ck = Counter(all_candidate_keywords)
    gk = Counter(all_generated_keywords)
    cm = Counter(all_common_keywords)
    return {
        "candidate_keywords": {"total_count": len(all_candidate_keywords), "unique_count": len(ck), "top_10": dict(ck.most_common(10))},
        "generated_keywords": {"total_count": len(all_generated_keywords), "unique_count": len(gk), "top_10": dict(gk.most_common(10))},
        "common_keywords": {"total_count": len(all_common_keywords), "unique_count": len(cm), "top_10": dict(cm.most_common(10))}
    }

def process_all_chunks(chunks: List[Dict], output_path: str = "qa_results.json", similarity_analysis_path: str = "similarity_analysis.json", chars_per_candidate: int = 240, max_retry: int = 3, similarity_threshold: float = 0.75, min_score: float = 6.0, api_delay: float = 2.0, failmem_sim_th: float = 0.90):
    results = []
    all_similarity_analysis = []
    overall_stats = {
        "total_chunks": len(chunks),
        "processed_chunks": 0,
        "total_candidates": 0,
        "successful_candidates": 0,
        "failed_candidates": 0,
        "total_qa_pairs_evaluated": 0,
        "total_accepted_pairs": 0,
        "overall_candidate_success_rate": 0,
        "overall_qa_success_rate": 0,
        "average_attempts_per_candidate": 0,
        "total_similarity_calculations": 0
    }
    for i, chunk in enumerate(chunks):
        print(f"[INFO] processing chunk {i+1}/{len(chunks)}: {chunk['id']}")
        qa_result = process_chunk_for_qa(
            chunk,
            chars_per_candidate=chars_per_candidate,
            max_retry=max_retry,
            similarity_threshold=similarity_threshold,
            min_score=min_score,
            api_delay=api_delay,
            failmem_sim_th=failmem_sim_th
        )
        results.append(qa_result)
        chunk_similarity_data = {
            "chunk_id": chunk["id"],
            "chunk_summary": qa_result["chunk_summary"],
            "similarity_details": qa_result["all_similarity_details"],
            "accepted_pairs": qa_result["accepted_results"],
            "total_similarity_calculations": len(qa_result["all_similarity_details"])
        }
        all_similarity_analysis.append(chunk_similarity_data)
        stats = qa_result["statistics"]
        overall_stats["processed_chunks"] += 1
        overall_stats["total_candidates"] += stats["total_candidates_generated"]
        overall_stats["successful_candidates"] += stats["successful_candidates"]
        overall_stats["failed_candidates"] += stats["failed_candidates"]
        overall_stats["total_qa_pairs_evaluated"] += stats["total_qa_pairs_evaluated"]
        overall_stats["total_accepted_pairs"] += stats["accepted_qa_pairs"]
        overall_stats["total_similarity_calculations"] += len(qa_result["all_similarity_details"])
        print(f"[INFO] chunk stats: candidates={stats['total_candidates_generated']}, success={stats['successful_candidates']}, fail={stats['failed_candidates']}, success_rate={stats['candidate_success_rate_percent']}%, avg_attempts={stats['average_attempts_per_candidate']}, sims={len(qa_result['all_similarity_details'])}")
    if overall_stats["total_candidates"] > 0:
        overall_stats["overall_candidate_success_rate"] = round((overall_stats["successful_candidates"] / overall_stats["total_candidates"]) * 100, 2)
    if overall_stats["total_qa_pairs_evaluated"] > 0:
        overall_stats["overall_qa_success_rate"] = round((overall_stats["total_accepted_pairs"] / overall_stats["total_qa_pairs_evaluated"]) * 100, 2)
    all_attempts = []
    for r in results:
        for c in r["candidate_results"]:
            all_attempts.append(c["attempt_rounds"])
    if all_attempts:
        overall_stats["average_attempts_per_candidate"] = round(sum(all_attempts) / len(all_attempts), 2)
    output_data = {
        "parameters": {
            "chars_per_candidate": chars_per_candidate,
            "max_retry": max_retry,
            "similarity_threshold": similarity_threshold,
            "min_score": min_score,
            "api_delay": api_delay,
            "failmem_sim_th": failmem_sim_th
        },
        "overall_statistics": overall_stats,
        "chunk_results": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    similarity_analysis_data = {
        "analysis_summary": {
            "total_chunks_analyzed": len(chunks),
            "total_similarity_calculations": overall_stats["total_similarity_calculations"],
            "similarity_threshold_used": similarity_threshold,
            "accepted_pairs_count": overall_stats["total_accepted_pairs"]
        },
        "chunk_similarity_analysis": all_similarity_analysis,
        "global_keyword_statistics": generate_keyword_statistics(all_similarity_analysis)
    }
    with open(similarity_analysis_path, "w", encoding="utf-8") as f:
        json.dump(similarity_analysis_data, f, ensure_ascii=False, indent=2)
    print("[INFO] done")
    print("[INFO] total chunks:", overall_stats["total_chunks"])
    print("[INFO] processed chunks:", overall_stats["processed_chunks"])
    print("[INFO] total candidates:", overall_stats["total_candidates"])
    print("[INFO] successful candidates:", overall_stats["successful_candidates"])
    print("[INFO] failed candidates:", overall_stats["failed_candidates"])
    print("[INFO] candidate success rate:", overall_stats["overall_candidate_success_rate"], "%")
    print("[INFO] total qa pairs evaluated:", overall_stats["total_qa_pairs_evaluated"])
    print("[INFO] accepted qa pairs:", overall_stats["total_accepted_pairs"])
    print("[INFO] qa success rate:", overall_stats["overall_qa_success_rate"], "%")
    print("[INFO] avg attempts per candidate:", overall_stats["average_attempts_per_candidate"])
    print("[INFO] total similarity calcs:", overall_stats["total_similarity_calculations"])
    print("[OK] saved results to:", output_path)
    print("[OK] saved similarity analysis to:", similarity_analysis_path)
