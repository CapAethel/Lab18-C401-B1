"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANKER_MODEL


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)

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
