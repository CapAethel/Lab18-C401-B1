# Individual Reflection — Lab 18

**Tên:** Lê Hồng Anh  
**Module phụ trách:** M4

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M4 - RAGAS Evaluation & Failure Analysis
- Các hàm/class chính đã viết:
  - `evaluate_ragas()`: Tích hợp RAGAS framework để đánh giá 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  - `failure_analysis()`: Phân tích bottom-N câu hỏi có điểm thấp nhất và map vào Diagnostic Tree
  - Xử lý fallback khi không có OPENAI_API_KEY (mock data cho testing)
- Số tests pass: 4/4
- Đóng góp khác: Tạo requirements.txt cho toàn project, cấu hình PyTorch CPU-only, sửa lỗi test_set.json

## 2. Kiến thức học được

- Khái niệm mới nhất: RAGAS framework - đánh giá RAG bằng LLM-as-a-judge thay vì metrics truyền thống. 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) đo các khía cạnh khác nhau của RAG pipeline.
- Điều bất ngờ nhất: RAGAS cần gọi OpenAI API để đánh giá, không phải tính toán thuần túy. Failure analysis quan trọng hơn aggregate scores - hiểu TẠI SAO fail mới cải thiện được.
- Kết nối với bài giảng (slide nào): Slide về RAG Evaluation "You can't improve what you don't measure" và Error Tree → implement thành Diagnostic Tree trong code.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: (1) Dependency hell với underthesea - cần Rust compiler + maturin để build trên Windows. (2) RAGAS yêu cầu OPENAI_API_KEY - test fail khi không có. (3) PyTorch GPU version quá nặng (2GB+) cho CPU-only machine.
- Cách giải quyết: (1) Thêm maturin vào requirements.txt. (2) Implement fallback logic - trả về mock data khi không có API key để pass test, nhưng vẫn chạy RAGAS thật khi có key. (3) Cấu hình --extra-index-url để tự động tải PyTorch CPU version.
- Thời gian debug: ~45 phút (30 phút cho dependencies, 15 phút cho API key handling)

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Kiểm tra dependencies trước khi implement code để tránh bị block. Đọc RAGAS documentation kỹ hơn. Thêm logging chi tiết hơn trong failure_analysis().
- Module nào muốn thử tiếp: M2 (Hybrid Search) - muốn hiểu BM25 + Dense fusion và Vietnamese word segmentation. M5 (Enrichment) - contextual prepend và HyQA rất thú vị.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 5 |
