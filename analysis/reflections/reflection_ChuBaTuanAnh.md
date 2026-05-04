# Individual Reflection — Lab 18

**Tên:** Chu Bá Tuấn Anh  
**Module phụ trách:** M1: Chunking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: Advanced Chunking Strategies.
- Các hàm/class chính đã viết:
  - `load_documents()` — đọc dữ liệu `.md`, `.txt`, `.pdf` để chuẩn bị corpus cho pipeline.
  - `chunk_semantic()` — split câu và nhóm theo similarity, có fallback lexical cosine khi thiếu embedding model.
  - `chunk_hierarchical()` — tạo parent chunks và child chunks; mỗi child giữ `parent_id` hợp lệ để retrieve child nhưng trả parent context.
  - `chunk_structure_aware()` — parse markdown headers, giữ section logic và metadata `section`.
  - `compare_strategies()` — chạy A/B comparison giữa basic, semantic, hierarchical và structure-aware.
- Số tests pass: 13/13 (`.venv/bin/python -m pytest tests/test_m1.py -q`).

## 2. Kiến thức học được

- Khái niệm mới nhất: hierarchical chunking giúp index child nhỏ để tăng precision, nhưng vẫn giữ parent context cho LLM.
- Điều bất ngờ nhất: chunking không chỉ là chia đều theo độ dài; chất lượng text extraction từ PDF, section/header và ranh giới ý nghĩa đều ảnh hưởng trực tiếp đến context recall.
- Kết nối với bài giảng: Production RAG cần tối ưu từng bước trước retrieval, đặc biệt là chunk granularity, metadata và khả năng trace từ retrieved chunk về source context.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: semantic chunking phụ thuộc embedding model, còn dữ liệu PDF có thể extract ra text rất ít nếu file scan hoặc thiếu parser phù hợp.
- Cách giải quyết: dùng `sentence_transformers` khi có sẵn, thêm fallback lexical cosine để module vẫn chạy offline, và giữ fallback PDF extraction để pipeline không crash.
- Thời gian debug: khoảng 30 phút.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: thêm overlap có kiểm soát cho child chunks, cache kết quả PDF extraction, và thử OCR/MarkItDown/pdfminer để cải thiện dữ liệu đầu vào.
- Module nào muốn thử tiếp: M2 Hybrid Search để đo trực tiếp chunking ảnh hưởng thế nào đến retrieval.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
