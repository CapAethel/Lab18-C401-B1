# Individual Reflection — Lab 18

**Tên:** Vũ Minh Khải
**Module phụ trách:** M3 - Reranking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M3 - Reranking
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker._load_model()`: Load model bge-reranker-v2-m3 sử dụng CrossEncoder
  - `CrossEncoderReranker.rerank()`: Rerank documents từ top-20 xuống top-k sử dụng cross-encoder scores
  - `FlashrankReranker.rerank()`: Lightweight reranker alternative sử dụng flashrank
  - `benchmark_reranker()`: Đo latency (avg/min/max) qua n_runs
- Số tests pass: 5/5

## 2. Kiến thức học được

- Khái niệm mới nhất: Cross-encoder reranking và sự khác biệt với bi-encoder, cách sử dụng bge-reranker-v2-m3 để cải thiện độ chính xác của search results
- Điều bất ngờ nhất: Reranking có thể cải thiện đáng kể thứ tự kết quả chỉ với top-k nhỏ, và latency của cross-encoder cao hơn nhiều so với bi-encoder
- Kết nối với bài giảng (slide nào): Slide về Reranking trong RAG pipeline

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Load model bge-reranker-v2-m3 và xử lý format input/output của CrossEncoder
- Cách giải quyết: Đọc documentation của sentence-transformers và FlagEmbedding, test với sample data nhỏ trước
- Thời gian debug: Khoảng 30-45 phút để hiểu cách model.predict() hoạt động và format pairs đúng

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thử implement thêm caching để giảm latency cho queries lặp lại, hoặc batch processing cho nhiều queries
- Module nào muốn thử tiếp: M2 (Hybrid Search) để hiểu rõ hơn về BM25 và dense retrieval, hoặc M4 (RAGAS Evaluation) để đo lường impact của reranking

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5|
| Code quality | 5|
| Teamwork | 5|
| Problem solving |5 |
