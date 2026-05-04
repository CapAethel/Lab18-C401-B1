"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    Uses underthesea word_tokenize to identify proper Vietnamese word
    boundaries.  E.g. "nghỉ phép" → "nghỉ_phép" (single token), which
    is critical for BM25 accuracy on Vietnamese text.
    """
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Expand underscore-joined Vietnamese compound tokens.

    underthesea may join multi-syllable words with underscores (e.g.
    "nghỉ_phép") in longer text but leave them separate in shorter
    queries.  To ensure BM25 matches regardless, we keep the compound
    AND add its parts: ["nghỉ_phép"] → ["nghỉ_phép", "nghỉ", "phép"].
    """
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if "_" in token:
            expanded.extend(token.split("_"))
    return expanded


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks.

        Steps:
        1. Store raw documents for later retrieval.
        2. Segment each chunk's text into Vietnamese words, then split by
           whitespace to get token lists.
        3. Expand compound tokens so sub-words are also indexed.
        4. Build a BM25Okapi index over the token corpus.
        """
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            _expand_tokens(segment_vietnamese(chunk["text"]).split())
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25.

        Segments the query with Vietnamese tokenizer, scores every
        document via BM25, and returns the top-k results sorted by
        descending score.
        """
        if self.bm25 is None:
            return []

        tokenized_query = _expand_tokens(segment_vietnamese(query).split())
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices sorted by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0  # skip zero-score docs
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant.

        Creates (or recreates) a Qdrant collection, encodes all chunk
        texts with the bge-m3 model, and upserts the resulting vectors
        together with their payloads.
        """
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Recreate collection with correct vector dimensions
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, (v, c) in enumerate(zip(vectors, chunks))
        ]

        # Upsert in batches of 100 to avoid payload size limits
        batch_size = 100
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection,
                points=points[start : start + batch_size],
            )

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors.

        Encodes the query with bge-m3, performs an ANN search in Qdrant,
        and returns the top-k results with cosine similarity scores.
        """
        query_vector = self._get_encoder().encode(query).tolist()
        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
        )

        return [
            SearchResult(
                text=hit.payload["text"],
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                method="dense",
            )
            for hit in hits
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).

    Each document accumulates a score across all input rankings.
    Documents appearing in multiple lists receive a higher fused score.
    All output results are tagged with method="hybrid".
    """
    # text → {"score": float, "result": SearchResult}
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result,
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    # Sort by fused score descending
    sorted_items = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in sorted_items[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
