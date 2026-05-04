# Group Report — Lab 18: Production RAG

**Nhóm:** C401-B1 
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên               | Module            | Hoàn thành | Tests pass |
| ----------------- | ----------------- | ---------- | ---------- |
| Chu Bá Tuấn Anh   | M1: Chunking      | ☑          | 13/13      |
| Nguyễn Mai Phương | M2: Hybrid Search | ☑          | 5/5        |
| Vũ Minh Khải      | M3: Reranking     | ☑          | 5/5        |
| Lê Hồng Anh       | M4: Evaluation    | ☑          | 4/4        |

## Kết quả RAGAS

| Metric            | Naive  | Production | Δ        |
| ----------------- | ------ | ---------- | -------- |
| Faithfulness      | 0.8507 | 0.4167     | -0.4340  |
| Answer Relevancy  | 0.5657 | 0.3524     | -0.2133  |
| Context Precision | 0.7067 | 0.5000     | -0.2067  |
| Context Recall    | 0.8400 | 0.8000     | -0.0400  |

## Key Findings

1. **Biggest challenge — LLM generation gây hallucination:** Pipeline production gọi `gpt-5-nano` để generate câu trả lời từ context, trong khi naive baseline trả thẳng raw context làm answer. Raw context luôn faithful với chính nó, nên naive faithfulness = 0.85. Khi LLM generate, 8/25 câu có faithfulness = 0 vì model thêm kiến thức nền không có trong context. Đây là bottleneck lớn nhất.

2. **Context Precision giảm (0.71 → 0.50):** Naive dùng 164 paragraph chunks, production dùng 331 hierarchical child chunks nhỏ hơn. Hierarchical child size 256 tokens cắt bảng tờ khai thuế thành nhiều đoạn rời, khiến nhiều chunk được retrieve không chứa đúng số liệu cần. Reranker (M3) giúp một phần nhưng không đủ vì ground-truth chunk không vào top-3.

3. **Context Recall ổn định (0.84 → 0.80):** Hybrid search (M2) + hierarchical chunking (M1) giữ được recall gần baseline. Điều này cho thấy M1+M2 hoạt động đúng — phần lớn thông tin cần thiết được tìm thấy, vấn đề là ở precision và generation.

## Presentation Notes (5 phút)

1. **RAGAS scores:** Naive → Production: Faithfulness 0.85 → 0.42, Answer Relevancy 0.57 → 0.35, Context Precision 0.71 → 0.50, Context Recall 0.84 → 0.80.

2. **Biggest finding — Production thấp hơn Naive, và lý do:** Naive pipeline trả raw context làm answer → trivially faithful. Production dùng LLM generation → model hallucinate trên câu hỏi số liệu tài chính cụ thể (tờ khai thuế) và câu hỏi liệt kê pháp lý (Nghị định). Trade-off rõ ràng: answer naturalness vs faithfulness.

3. **Case study failure — Error Tree:** Câu "DHA Surfaces có phát sinh hàng nhập khẩu không?" → faithfulness=0. Error tree: (1) Output sai? Có. (2) Context đúng? Một phần — BM25+dense match tờ khai nhưng chunk bị cắt không chứa chỉ tiêu [30][31]. (3) LLM hallucinate? Có — model thêm nội dung không có trong context. Root cause: hierarchical chunking cắt ngang bảng tờ khai + LLM không có constraint "chỉ dùng context".

4. **Next optimization nếu có thêm 1 giờ:** (1) Thêm hard constraint vào system prompt: "Nếu không có trong context, trả lời 'Không tìm thấy'." (2) Dùng `chunk_structure_aware()` cho tờ khai/form — giữ nguyên section thay vì fixed 256 tokens. (3) Tăng `RERANK_TOP_K` lên 5-7 cho câu hỏi liệt kê. (4) Bật M5 enrichment (HyQA) để tăng context recall thêm.
