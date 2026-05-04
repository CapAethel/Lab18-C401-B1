# Failure Analysis — Lab 18: Production RAG

**Nhóm:** C401-B1  
**Thành viên:** Chu Bá Tuấn Anh → M1 · Nguyễn Mai Phương → M2 · Vũ Minh Khải → M3 · Lê Hồng Anh → M4

---

## RAGAS Scores

| Metric            | Naive Baseline | Production | Δ        |
| ----------------- | -------------- | ---------- | -------- |
| Faithfulness      | 0.8507         | 0.4167     | -0.4340  |
| Answer Relevancy  | 0.5657         | 0.3524     | -0.2133  |
| Context Precision | 0.7067         | 0.5000     | -0.2067  |
| Context Recall    | 0.8400         | 0.8000     | -0.0400  |

## Bottom-5 Failures

### #1
- **Question:** DHA Surfaces có phát sinh hàng hóa, dịch vụ nhập khẩu trong kỳ không?
- **Expected:** Thông tin về hàng nhập khẩu trong kỳ tính thuế GTGT Quý 4/2024
- **Got:** Câu trả lời từ gpt-5-nano với faithfulness = 0.00 (hoàn toàn hallucinated)
- **Worst metric:** faithfulness = 0.00 · avg_score = 0.0087
- **Error Tree:** Output sai → Context có thông tin? Không chắc → Query được search đúng không? Có, hybrid search lấy chunk từ tờ khai → LLM có trả lời đúng? Không, model thêm thông tin không có trong context
- **Root cause:** `gpt-5-nano` với `reasoning_effort=minimal` tự suy diễn câu trả lời thay vì chỉ trích xuất từ context. Câu hỏi Yes/No về nhập khẩu cần thông tin cụ thể từ chỉ tiêu [30]/[31] trong tờ khai, model không tìm thấy nên tự sinh.
- **Suggested fix:** Thêm instruction rõ "Nếu context không có thông tin, trả lời 'Không tìm thấy'". Dùng system prompt constraint mạnh hơn.

### #2
- **Question:** Theo Nghị định 13/2023, dữ liệu cá nhân nhạy cảm gồm những loại nào?
- **Expected:** Danh sách đầy đủ các loại dữ liệu nhạy cảm theo điều khoản của Nghị định
- **Got:** Câu trả lời từ gpt-5-nano với faithfulness = 0.00
- **Worst metric:** faithfulness = 0.00 · avg_score = 0.0629
- **Error Tree:** Output sai → Context có chunk Nghị định không? Có, M2 tìm được chunk Nghị định → LLM extract đúng không? Không, model liệt kê thiếu/sai các loại dữ liệu nhạy cảm
- **Root cause:** Dữ liệu nhạy cảm được định nghĩa trong một đoạn dài với danh sách con. Hierarchical chunking cắt danh sách sang nhiều child chunks. Reranker chỉ lấy top-3, bỏ sót một số mục. LLM hallucinate phần còn lại.
- **Suggested fix:** Tăng `RERANK_TOP_K` từ 3 lên 5 cho câu hỏi dạng liệt kê. Thêm parent chunk retrieval khi child chunk chứa danh sách.

### #3
- **Question:** Tổng doanh thu bán hàng trong kỳ của DHA Surfaces là bao nhiêu?
- **Expected:** 3.703.688.610 đồng (chỉ tiêu [34])
- **Got:** Context không chứa chỉ tiêu [34] trực tiếp — context_precision = 0.00
- **Worst metric:** context_precision = 0.00 · avg_score = 0.0781
- **Error Tree:** Output sai → Context đúng không? Không, context trả về các chunk liên quan nhưng không chứa số liệu cụ thể [34] → M2 search sai không? BM25 match từ "doanh thu" nhưng child chunk chứa context xung quanh, không phải dòng số liệu → Chunking sai không? Có, hierarchical child 256 tokens cắt bảng số liệu tờ khai thành nhiều đoạn không đầy đủ.
- **Root cause:** Tờ khai thuế GTGT có cấu trúc bảng với nhiều chỉ tiêu số. Child chunk 256 tokens cắt ngang bảng. BM25 match "doanh thu" nhưng chunk không chứa đúng dòng [34]. Reranker không giúp được vì ground truth chunk không có trong top-20.
- **Suggested fix:** Dùng structure-aware chunking (giữ nguyên bảng/section) thay vì fixed-size cho tài liệu dạng form/tờ khai. Tăng child chunk size lên 512 token.

### #4
- **Question:** Thuế suất GTGT áp dụng cho hàng hóa bán ra của DHA Surfaces là bao nhiêu?
- **Expected:** 10% (thuế suất GTGT tiêu chuẩn)
- **Got:** Câu trả lời từ gpt-5-nano với faithfulness = 0.00
- **Worst metric:** faithfulness = 0.00 · avg_score = 0.2704
- **Error Tree:** Output sai → Context có số liệu thuế suất không? Có (10%) → LLM trả lời đúng không? Không, model thêm giải thích và chú thích không có trong context
- **Root cause:** Context chứa con số 10% nhưng không có giải thích "thuế suất tiêu chuẩn". `gpt-5-nano` thêm kiến thức nền ("theo luật thuế GTGT hiện hành...") không được xác nhận bởi context → faithfulness = 0.
- **Suggested fix:** System prompt constraint: "Trả lời CHỈ bằng thông tin trong context, không thêm kiến thức nền."

### #5
- **Question:** Theo Nghị định 13/2023, Bên Kiểm soát dữ liệu phải thực hiện yêu cầu xóa dữ liệu trong bao lâu?
- **Expected:** Thời hạn cụ thể theo điều khoản của Nghị định
- **Got:** Câu trả lời từ gpt-5-nano với faithfulness = 0.00
- **Worst metric:** faithfulness = 0.00 · avg_score = 0.2803
- **Error Tree:** Output sai → Context có điều khoản về xóa dữ liệu không? Một phần → Reranker trả về đúng chunk không? Chunk về quyền xóa dữ liệu được trả về nhưng không có thời hạn cụ thể → LLM hallucinate thời hạn?
- **Root cause:** Thông tin về thời hạn xóa dữ liệu nằm trong điều khoản chi tiết có thể bị tách sang chunk khác. LLM tự điền thời hạn không có trong context.
- **Suggested fix:** Retrieve parent chunk khi child chunk về cùng điều khoản được match. Tăng context window bằng cách thêm adjacent chunks.

## Case Study (cho presentation)

**Question chọn phân tích:** DHA Surfaces có phát sinh hàng hóa, dịch vụ nhập khẩu trong kỳ không?

**Error Tree walkthrough:**
1. Output đúng? → Không: faithfulness = 0.00, LLM thêm thông tin không có trong context.
2. Context đúng? → Một phần: hybrid search tìm được chunk từ tờ khai GTGT, nhưng chunk không chứa đúng chỉ tiêu nhập khẩu [30]/[31].
3. Query search OK? → Có: BM25 + dense đều match "nhập khẩu" trong tờ khai.
4. Vấn đề ở đâu? → Hai điểm: (a) chunking cắt bảng tờ khai không giữ nguyên dòng chỉ tiêu, (b) LLM generation không tuân thủ "chỉ dùng context".
5. Fix ở bước: Fix generation prompt (constraint không hallucinate) + dùng structure-aware chunking cho form/tờ khai.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Prompt**: Thêm few-shot example về câu trả lời "Không tìm thấy" khi context thiếu thông tin.
- **Chunking (M1)**: Dùng `chunk_structure_aware()` cho tài liệu dạng bảng/form (tờ khai thuế), giữ nguyên section/bảng thay vì fixed-size.
- **Context window**: Tăng `RERANK_TOP_K` từ 3 lên 5-7, đặc biệt cho câu hỏi liệt kê và câu hỏi về bảng số liệu.
- **Evaluation**: Bật M5 enrichment để so sánh ảnh hưởng của HyQA và contextual prepend lên context_recall.

## Tổng kết

**Nguyên nhân chính production thấp hơn baseline:**  
Naive baseline trả raw context làm answer → faithfulness luôn ≈ 0.85 (raw chunk chứa cả câu không liên quan). Production gọi `gpt-5-nano` generate answer → model hallucinate trên 8/25 câu hỏi. Đây là trade-off: answer naturalness vs faithfulness. Với domain tài liệu pháp lý/tài chính, cần tăng constraint cho LLM generation.
