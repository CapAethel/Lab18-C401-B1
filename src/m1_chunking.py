"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import math
import os, sys, glob, re
import subprocess
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                        SEMANTIC_THRESHOLD)
except ModuleNotFoundError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    HIERARCHICAL_PARENT_SIZE = 2048
    HIERARCHICAL_CHILD_SIZE = 256
    SEMANTIC_THRESHOLD = 0.85


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown, text, and PDF files from data/."""
    docs = []
    patterns = ["*.md", "*.txt", "*.pdf"]
    filepaths = []
    for pattern in patterns:
        filepaths.extend(glob.glob(os.path.join(data_dir, pattern)))

    for fp in sorted(filepaths):
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".pdf":
            text = _read_pdf(fp)
        else:
            with open(fp, encoding="utf-8") as f:
                text = f.read()
        if text.strip():
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    similarities = _sentence_similarities(sentences)
    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([])
        groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "chunk_index": i, "strategy": "semantic"},
        )
        for i, group in enumerate(groups)
        if group and " ".join(group).strip()
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parent_texts = _pack_paragraphs(text, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "chunk_index": parent_index,
                "parent_id": parent_id,
            },
        ))

        for child_index, child_text in enumerate(_split_by_size(parent_text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "chunk_index": child_index,
                    "parent_index": parent_index,
                },
                parent_id=parent_id,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if not matches:
        return [
            Chunk(text=c.text, metadata={**c.metadata, "strategy": "structure", **metadata})
            for c in chunk_basic(text, metadata=metadata)
        ]

    chunks: list[Chunk] = []
    preface = text[:matches[0].start()].strip()
    if preface:
        chunks.append(Chunk(
            text=preface,
            metadata={
                **metadata,
                "section": "preface",
                "section_level": 0,
                "chunk_index": len(chunks),
                "strategy": "structure",
            },
        ))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        chunks.append(Chunk(
            text=section_text,
            metadata={
                **metadata,
                "section": match.group(2).strip(),
                "section_level": len(match.group(1)),
                "chunk_index": len(chunks),
                "strategy": "structure",
            },
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    basic_chunks: list[Chunk] = []
    semantic_chunks: list[Chunk] = []
    parent_chunks: list[Chunk] = []
    child_chunks: list[Chunk] = []
    structure_chunks: list[Chunk] = []

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        basic_chunks.extend(chunk_basic(text, metadata=metadata))
        semantic_chunks.extend(chunk_semantic(text, metadata=metadata))
        parents, children = chunk_hierarchical(text, metadata=metadata)
        parent_chunks.extend(parents)
        child_chunks.extend(children)
        structure_chunks.extend(chunk_structure_aware(text, metadata=metadata))

    results = {
        "basic": _chunk_stats(basic_chunks),
        "semantic": _chunk_stats(semantic_chunks),
        "hierarchical": {
            **_chunk_stats(child_chunks),
            "num_parents": len(parent_chunks),
            "num_children": len(child_chunks),
        },
        "structure": _chunk_stats(structure_chunks),
    }

    print("Strategy      | Chunks | Avg Len | Min | Max")
    print("--------------|--------|---------|-----|----")
    for name in ["basic", "semantic", "hierarchical", "structure"]:
        stats = results[name]
        chunk_label = str(stats["num_chunks"])
        if name == "hierarchical":
            chunk_label = f"{stats['num_parents']}p/{stats['num_children']}c"
        print(
            f"{name:<13} | {chunk_label:>6} | "
            f"{stats['avg_length']:>7.1f} | {stats['min_length']:>3} | {stats['max_length']:>3}"
        )
    return results


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        if sentence.strip()
    ]


def _read_pdf(filepath: str) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", filepath, "-"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except Exception:
        return ""


def _sentence_similarities(sentences: list[str]) -> list[float]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
        embeddings = model.encode(sentences)
        return [
            _cosine_similarity(list(embeddings[i - 1]), list(embeddings[i]))
            for i in range(1, len(sentences))
        ]
    except Exception:
        vectors = [_token_counter(sentence) for sentence in sentences]
        return [
            _counter_cosine(vectors[i - 1], vectors[i])
            for i in range(1, len(vectors))
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _token_counter(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _counter_cosine(left: Counter, right: Counter) -> float:
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _pack_paragraphs(text: str, max_size: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    packed: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                packed.append(current.strip())
                current = ""
            packed.extend(_split_by_size(paragraph, max_size))
            continue

        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(paragraph) > max_size:
            packed.append(current.strip())
            current = paragraph
        else:
            current = f"{current}{separator}{paragraph}"

    if current.strip():
        packed.append(current.strip())
    return packed


def _split_by_size(text: str, size: int) -> list[str]:
    if size <= 0:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def _chunk_stats(chunks: list[Chunk]) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
