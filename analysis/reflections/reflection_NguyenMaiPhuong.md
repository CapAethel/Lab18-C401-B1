# Individual Reflection — Lab 18

**Tên:** Nguyễn Mai Phương  
**Module phụ trách:** M2: Hybrid Search

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M2 — Hybrid Search (BM25 Vietnamese + Dense Vector + RRF Fusion)
- Các hàm/class chính đã viết:
  - `segment_vietnamese()` — Vietnamese word segmentation bằng `underthesea.word_tokenize`
  - `_expand_tokens()` — Xử lý token compound (nghỉ_phép → nghỉ_phép + nghỉ + phép) để fix lỗi inconsistent segmentation
  - `BM25Search.index()` — Segment + tokenize corpus, build BM25Okapi index
  - `BM25Search.search()` — Segment query → BM25 scoring → top-k results
  - `DenseSearch.index()` — Encode chunks bằng bge-m3, upsert vào Qdrant collection
  - `DenseSearch.search()` — Encode query → ANN search Qdrant → cosine similarity results
  - `reciprocal_rank_fusion()` — Merge BM25 + Dense rankings: score(d) = Σ 1/(k + rank + 1)
- Số tests pass: 5/5

## 2. Kiến thức học được

- Khái niệm mới nhất: Reciprocal Rank Fusion (RRF) — cách merge nhiều ranked lists mà không cần normalize scores. Công thức đơn giản `1/(k + rank)` nhưng hiệu quả cao vì chỉ dựa vào thứ hạng, không phụ thuộc scale của score từ mỗi retriever.
- Điều bất ngờ nhất: `underthesea` segment khác nhau tùy theo độ dài text — "nghỉ phép năm" → "nghỉ_phép năm" (đúng), nhưng "nghỉ phép" → "nghỉ phép" (không join). Điều này gây mismatch giữa query tokens và document tokens trong BM25. Phải viết `_expand_tokens()` để xử lý edge case này.
- Kết nối với bài giảng: Slide về Hybrid Search — kết hợp lexical (BM25) + semantic (dense) search để tận dụng ưu điểm cả hai: BM25 tốt cho exact match / keyword, dense tốt cho semantic similarity / paraphrase.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Vietnamese tokenization inconsistency — underthesea xử lý compound words khác nhau giữa short query và long document. BM25 search trả về empty results cho query "nghỉ phép" dù document chứa "nghỉ phép năm 12 ngày" vì tokens không match (nghỉ + phép vs nghỉ_phép).
- Cách giải quyết: Viết helper `_expand_tokens()` — mỗi compound token như "nghỉ_phép" được expand thành cả compound lẫn individual parts ["nghỉ_phép", "nghỉ", "phép"]. Apply cho cả indexing và search để đảm bảo match bất kể underthesea segment thế nào.
- Thời gian debug: ~15 phút — chạy test thấy fail → in segmentation output để so sánh query vs document tokens → phát hiện underscore inconsistency → implement fix → 5/5 pass.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thêm caching cho `segment_vietnamese()` vì underthesea load model mỗi lần gọi khá chậm. Có thể dùng `functools.lru_cache` hoặc pre-segment toàn bộ corpus 1 lần. Ngoài ra sẽ thêm lowercase normalization cho BM25 tokens để tăng recall.
- Module nào muốn thử tiếp: M3 (Reranking) — vì cross-encoder reranker là bước tiếp theo tự nhiên sau hybrid search. Muốn hiểu cách cross-encoder (bge-reranker-v2-m3) so sánh query-document pairs và benchmark latency giữa cross-encoder vs flashrank.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
