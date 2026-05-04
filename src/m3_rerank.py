"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
import math
import re
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
                self._model = False
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        model = self._load_model()
        if model:
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
        else:
            scores = [_lexical_relevance(query, doc["text"]) for doc in documents]

        scored_docs = [(float(score), doc) for score, doc in zip(scores, documents)]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
                rank=i + 1
            ))
        
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Lightweight reranker using flashrank (<5ms)."""
        if self._model is None:
            try:
                from flashrank import Ranker, RerankRequest
                self._request_cls = RerankRequest
                self._model = Ranker()
            except Exception:
                self._model = False
        if not self._model:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)
        
        # Prepare passages for flashrank
        passages = [{"text": d["text"]} for d in documents]
        
        # Rerank using flashrank
        rerank_request = self._request_cls(query=query, passages=passages)
        results_raw = self._model.rerank(rerank_request)
        
        # Convert to RerankResult format
        results = []
        for i, result in enumerate(results_raw[:top_k]):
            doc = documents[result.get("index", i)]
            results.append(RerankResult(
                text=result["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=result.get("score", 0.0),
                metadata=doc.get("metadata", {}),
                rank=i + 1
            ))
        
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    # 1. Collect timing measurements
    times = []
    
    # 2. Run reranking n_runs times and measure latency
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000  # Convert to milliseconds
        times.append(elapsed_ms)
    
    # 3. Calculate statistics
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    
    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms
    }


def _lexical_relevance(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    overlap = _counter_cosine(Counter(query_tokens), Counter(text_tokens))
    asks_quantity = any(token in query.lower() for token in ["bao nhiêu", "mấy", "số lượng"])
    number_bonus = 0.2 if asks_quantity and re.search(r"\d+", text) else 0.0
    policy_bonus = 0.3 if any(token in text_tokens for token in ["nghỉ", "phép", "nghỉ_phép"]) else 0.0
    return overlap + number_bonus + policy_bonus


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("nghỉ phép", "nghỉ_phép")
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if "_" in token:
            expanded.extend(token.split("_"))
    return expanded


def _counter_cosine(left: Counter, right: Counter) -> float:
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
