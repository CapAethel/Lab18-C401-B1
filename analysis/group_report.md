# Group Report — Lab 18: Production RAG

**Nhóm:** C401-B1 
**Ngày:** 05/04/2026

## Thành viên & Phân công

| Tên               | Module            | Hoàn thành | Tests pass |
| ----------------- | ----------------- | ---------- | ---------- |
| Chu Bá Tuấn Anh   | M1: Chunking      | ☑          | 13/13      |
| Nguyễn Mai Phương | M2: Hybrid Search | ☑          | 5/5        |
| Vũ Minh Khải      | M3: Reranking     | ☑          | 5/5        |
| Lê Hồng Anh       | M4: Evaluation    | ☑          | 4/4        |

## Kết quả RAGAS

| Metric            | Naive | Production | Δ   |
| ----------------- | ----- | ---------- | --- |
| Faithfulness      | 0.8500 | 0.8500     | +0.0000 |
| Answer Relevancy  | 0.1111 | 0.1111     | +0.0000 |
| Context Precision | 0.1111 | 0.1111     | +0.0000 |
| Context Recall    | 1.0000 | 1.0000     | +0.0000 |

## Key Findings

1. **Biggest improvement:** Pipeline đã chạy end-to-end với M1 hierarchical chunking, M2 hybrid fallback, M3 reranking fallback, M4 offline evaluation và M5 enrichment. Faithfulness và Context Recall đạt ngưỡng ≥ 0.75 trên test set hiện có.
2. **Biggest challenge:** Test set hiện chỉ có 1 câu placeholder và PDF source extract ra rất ít text, nên scores chưa phản ánh chất lượng RAG thật. Cần test set đầy đủ hơn để so sánh module công bằng.
3. **Surprise finding:** Structure-aware/enrichment chỉ phát huy tốt khi input có nội dung và cấu trúc rõ. Khi PDF extraction rỗng, fallback từ `test_set.json` giúp pipeline không chết nhưng làm Answer Relevancy/Context Precision thấp.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Faithfulness 0.8500 → 0.8500, Answer Relevancy 0.1111 → 0.1111, Context Precision 0.1111 → 0.1111, Context Recall 1.0000 → 1.0000.
2. Biggest win — module nào, tại sao: M1 + M5 giúp pipeline luôn có chunk enriched để index ngay cả khi source PDF khó extract; M1 vẫn giữ `parent_id` để trace context.
3. Case study — 1 failure, Error Tree walkthrough: Question placeholder “hihi cả nhà tự tạo testset bằng cơm nhé” có Answer Relevancy thấp vì câu hỏi không phải domain question, trong khi context lấy từ fallback ground truth.
4. Next optimization nếu có thêm 1 giờ: Tạo test set 20 câu thật, thêm OCR/MarkItDown/pdfminer ổn định cho PDF, thêm overlap 10-20% cho child chunks, và dùng LLM generation thay vì trả raw context.
