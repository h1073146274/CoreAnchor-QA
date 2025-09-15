from typing import List, Tuple
from similarity_utils import compute_similarity

class FailureMemory:
    def __init__(self, sim_threshold: float = 0.90):
        self.sim_threshold = float(sim_threshold)
        self.failed_questions: List[str] = []

    def add(self, question: str):
        q = (question or "").strip()
        if q:
            self.failed_questions.append(q)

    def _max_sim(self, q: str) -> float:
        if not self.failed_questions:
            return 0.0
        sims = [compute_similarity(q, old) for old in self.failed_questions]
        try:
            return float(max(sims))
        except Exception:
            return 0.0

    def filter(self, questions: List[str]) -> Tuple[List[str], List[tuple]]:
        kept, dropped = [], []
        for q in questions or []:
            s = self._max_sim(q)
            if s >= self.sim_threshold:
                dropped.append((q, s))
            else:
                kept.append(q)
        return kept, dropped
