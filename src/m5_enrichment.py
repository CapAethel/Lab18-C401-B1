"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, sys, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import OPENAI_API_KEY
except ModuleNotFoundError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    text = text.strip()
    if not text:
        return ""

    llm_summary = _call_openai(
        system="Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        user=text,
        max_tokens=150,
    )
    if llm_summary:
        return llm_summary

    sentences = _split_sentences(text)
    if not sentences:
        return text[:240].strip()
    return " ".join(sentences[:2]).strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    text = text.strip()
    if not text or n_questions <= 0:
        return []

    llm_questions = _call_openai(
        system=(
            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
            "Trả về mỗi câu hỏi trên 1 dòng."
        ),
        user=text,
        max_tokens=200,
    )
    if llm_questions:
        questions = [_clean_question(line) for line in llm_questions.splitlines()]
        questions = [q for q in questions if q]
        if questions:
            return questions[:n_questions]

    topic = _infer_topic(text)
    questions = [
        f"{topic.capitalize()} được quy định như thế nào?",
        f"Thông tin chính về {topic} là gì?",
        f"Đoạn này trả lời câu hỏi nào về {topic}?",
    ]
    if re.search(r"\d+", text):
        questions.insert(
            0, f"{topic.capitalize()} là bao nhiêu hoặc trong thời hạn mấy ngày?"
        )
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    text = text.strip()
    if not text:
        return ""

    title = document_title.strip() or "tài liệu nguồn"
    llm_context = _call_openai(
        system=(
            "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu "
            "và nói về chủ đề gì. Chỉ trả về 1 câu."
        ),
        user=f"Tài liệu: {title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    context = llm_context or f"Trích từ {title}, đoạn này nói về {_infer_topic(text)}."
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    text = text.strip()
    if not text:
        return {}

    llm_metadata = _call_openai(
        system=(
            "Trích xuất metadata từ đoạn văn. Trả về JSON: "
            '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'
        ),
        user=text,
        max_tokens=150,
    )
    if llm_metadata:
        parsed = _parse_json_object(llm_metadata)
        if parsed:
            return parsed

    return {
        "topic": _infer_topic(text),
        "entities": _extract_entities(text),
        "category": _infer_category(text),
        "language": _infer_language(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []
    selected = set(methods)
    use_full = "full" in selected

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {}) or {}
        if not text:
            continue

        summary = summarize_chunk(text) if use_full or "summary" in selected else ""
        questions = (
            generate_hypothesis_questions(text)
            if use_full or "hyqa" in selected
            else []
        )
        enriched_text = (
            contextual_prepend(text, metadata.get("source", ""))
            if use_full or "contextual" in selected
            else text
        )
        auto_meta = extract_metadata(text) if use_full or "metadata" in selected else {}

        if questions:
            enriched_text = f"{enriched_text}\n\nCâu hỏi giả định:\n" + "\n".join(
                f"- {q}" for q in questions
            )
        if summary and (use_full or "summary" in selected):
            enriched_text = f"Tóm tắt: {summary}\n\n{enriched_text}"

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**metadata, **auto_meta},
                method="+".join(methods),
            )
        )

    return enriched


def _call_openai(system: str, user: str, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        if sentence.strip()
    ]


def _clean_question(line: str) -> str:
    question = line.strip().lstrip("-*• ")
    question = re.sub(r"^\d+[\).\-\s]+", "", question).strip()
    return question


def _infer_topic(text: str) -> str:
    lowered = text.lower()
    keyword_topics = {
        "nghỉ phép": ["nghỉ phép", "nghỉ năm", "nghỉ không lương"],
        "bảo mật tài khoản": ["mật khẩu", "vpn", "tài khoản", "đăng nhập"],
        "tài chính": ["hóa đơn", "chi phí", "thanh toán", "báo cáo tài chính"],
        "dữ liệu cá nhân": ["dữ liệu cá nhân", "chủ thể dữ liệu", "xử lý dữ liệu"],
        "nhân sự": ["nhân viên", "giám đốc", "phòng ban", "thâm niên"],
    }
    for topic, keywords in keyword_topics.items():
        if any(keyword in lowered for keyword in keywords):
            return topic

    words = re.findall(r"\w+", lowered, flags=re.UNICODE)
    stopwords = {"và", "của", "là", "có", "the", "a", "an", "to", "of", "in"}
    candidates = [word for word in words if len(word) > 3 and word not in stopwords]
    return " ".join(candidates[:3]) if candidates else "nội dung tài liệu"


def _infer_category(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["mật khẩu", "vpn", "tài khoản", "hệ thống"]):
        return "it"
    if any(
        word in lowered for word in ["nghỉ phép", "nhân viên", "thâm niên", "giám đốc"]
    ):
        return "hr"
    if any(word in lowered for word in ["chi phí", "thanh toán", "hóa đơn", "bctc"]):
        return "finance"
    return "policy"


def _infer_language(text: str) -> str:
    vietnamese_chars = (
        "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    )
    return "vi" if any(char in text.lower() for char in vietnamese_chars) else "en"


def _extract_entities(text: str) -> list[str]:
    entities = re.findall(
        r"\b(?:[A-ZĐ][\wÀ-ỹ]+)(?:\s+[A-ZĐ][\wÀ-ỹ]+){0,4}",
        text,
        flags=re.UNICODE,
    )
    cleaned = []
    for entity in entities:
        entity = entity.strip()
        if entity and entity not in cleaned:
            cleaned.append(entity)
    return cleaned[:8]


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
