# Failure Analysis — Lab 18: Production RAG

**Nhóm:** C401-B1  
**Thành viên:** Chu Bá Tuấn Anh → M1 · Nguyễn Mai Phương → M2 · Vũ Minh Khải → M3 · Lê Hồng Anh → M4

---

## RAGAS Scores

| Metric            | Naive Baseline | Production | Δ   |
| ----------------- | -------------- | ---------- | --- |
| Faithfulness      | 0.8500         | 0.8500     | +0.0000 |
| Answer Relevancy  | 0.1111         | 0.1111     | +0.0000 |
| Context Precision | 0.1111         | 0.1111     | +0.0000 |
| Context Recall    | 1.0000         | 1.0000     | +0.0000 |

## Bottom-5 Failures

### #1
- **Question:** hihi cả nhà tự tạo testset bằng cơm nhé
- **Expected:** Không kịp tạo luôn
- **Got:** Trích từ `test_set_fallback`, đoạn này nói về không luôn. Không kịp tạo luôn. Câu hỏi giả định: Không luôn được quy định như thế nào?...
- **Worst metric:** answer_relevancy = 0.1111
- **Error Tree:** Output sai một phần → Context đúng? Có, context chứa ground truth → Query OK? Không, query là placeholder không thuộc domain RAG →
- **Root cause:** Test set chưa được tạo đúng domain; câu hỏi placeholder không khớp với nội dung pháp lý/tài chính, nên Answer Relevancy thấp dù Context Recall cao.
- **Suggested fix:** Tạo 20 câu hỏi thật từ tài liệu, dùng LLM generation trả lời ngắn dựa trên context, và bỏ fallback test-set context khi có corpus PDF/OCR tốt.

### #2
Không có thêm failure vì `test_set.json` hiện chỉ có 1 câu hỏi.

### #3
Không có thêm failure vì `test_set.json` hiện chỉ có 1 câu hỏi.

### #4
Không có thêm failure vì `test_set.json` hiện chỉ có 1 câu hỏi.

### #5
Không có thêm failure vì `test_set.json` hiện chỉ có 1 câu hỏi.

## Case Study (cho presentation)

**Question chọn phân tích:** hihi cả nhà tự tạo testset bằng cơm nhé

**Error Tree walkthrough:**
1. Output đúng? → Một phần: có chứa expected answer nhưng kèm nhiều context/enrichment text.
2. Context đúng? → Có: context lấy từ fallback ground truth “Không kịp tạo luôn”.
3. Query rewrite OK? → Không áp dụng; query là placeholder, không phải câu hỏi thông tin thật.
4. Fix ở bước: Fix test set + generation prompt; sau đó mới đánh giá sâu M1/M2/M3.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tạo lại test set 20 câu từ corpus thật.
- M1: thêm overlap cho child chunks, cache text extracted từ PDF, và ưu tiên structure-aware chunking khi tài liệu có heading rõ.
- Pipeline: thêm LLM generation để answer không chỉ là raw context.

## M1 Notes — Chunking Diagnostics

- **Khi worst metric là Context Recall:** kiểm tra liệu relevant context có bị mất ngay từ bước M1 không. Nguyên nhân thường gặp là PDF extract ra text rỗng/quá ngắn, chunk quá lớn làm retrieval kém chính xác, hoặc child chunk bị cắt ngay trước/sau số liệu quan trọng.
- **Khi worst metric là Context Precision:** kiểm tra child chunks có quá rộng hoặc thiếu section metadata không. Suggested fix: dùng hierarchical child size nhỏ hơn, thêm overlap có kiểm soát, hoặc dùng structure-aware chunking theo header.
- **Đóng góp M1 hiện tại:** `chunk_semantic()` nhóm câu theo similarity, `chunk_hierarchical()` tạo parent/child với `parent_id`, `chunk_structure_aware()` giữ section headers, và `compare_strategies()` xuất thống kê A/B với baseline.
- **Giới hạn cần nêu khi present:** M1 phụ thuộc chất lượng text extraction từ PDF; nếu tài liệu scan hoặc extract không đầy đủ, cần OCR/MarkItDown/pdfminer tốt hơn trước khi đánh giá RAGAS.
