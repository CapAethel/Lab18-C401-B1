# Individual Reflection — Lab 18

**Tên:** Chu Bá Tuấn Anh  
**Module phụ trách:** M1: Chunking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: Advanced Chunking Strategies.
- Các hàm/class chính đã viết: `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`.
- Số tests pass: 13/13.

## 2. Kiến thức học được

- Khái niệm mới nhất: hierarchical chunking giúp index child nhỏ để tăng precision, nhưng vẫn giữ parent context cho LLM.
- Điều bất ngờ nhất: chunking không chỉ là chia đều theo độ dài; giữ section/header và ranh giới ý nghĩa ảnh hưởng trực tiếp đến context recall.
- Kết nối với bài giảng: Production RAG cần tối ưu từng bước trước retrieval, đặc biệt là chunk granularity và metadata.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: semantic chunking phụ thuộc embedding model, có thể không chạy được nếu môi trường thiếu dependency hoặc model chưa tải.
- Cách giải quyết: dùng `sentence_transformers` khi có sẵn, và thêm fallback lexical cosine để module vẫn chạy offline.
- Thời gian debug: khoảng 20 phút.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: thêm overlap có kiểm soát cho child chunks để giảm rủi ro mất thông tin ở ranh giới chunk.
- Module nào muốn thử tiếp: M2 Hybrid Search để đo trực tiếp chunking ảnh hưởng thế nào đến retrieval.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
