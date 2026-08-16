# Lộ trình AI Retrieval tối thiểu cho người học phần cứng

Mục tiêu: đủ khả năng đọc, chạy, kiểm tra và sửa pipeline AIC2026 bằng AI hỗ trợ. Không nhằm biến người học thành ML researcher.

Thời lượng gợi ý: 10 buổi, mỗi buổi 60–90 phút. Mỗi buổi phải làm bài tập trước khi chuyển phần.

## 0. Bản đồ hệ thống cần nhớ

```text
Video -> keyframe -> embedding/index
                         |
Query -> chuẩn hóa/mở rộng -> CLIP + SigLIP2 + OCR/object
                         -> FAISS lấy candidate
                         -> fusion/rerank/diversify
                         -> KIS: trả video + frame
                         -> QA: Qwen nhìn candidate và trả answer
                         -> TRAKE: căn chỉnh chuỗi + Qwen xác minh
                         -> benchmark + human review + CSV/ZIP
```

Nguyên tắc: retrieval tìm nhanh và tăng recall; reasoning kiểm tra ít candidate nhưng chính xác hơn.

---

## 1. Python căn bản để đọc và sửa project

### Phải biết

- Kiểu dữ liệu: `str`, `int`, `float`, `bool`, `list`, `tuple`, `dict`, `set`.
- `if`, `for`, list comprehension, hàm, class và import.
- `Path`, đọc JSON/CSV, context manager `with`.
- `argparse`: tham số CLI và giá trị mặc định.
- Exception và traceback: đọc từ dòng cuối lên trên, tìm frame đầu tiên thuộc code project.
- Type hint như `list[dict[str, object]]` chỉ mô tả cấu trúc dữ liệu.

### Quy trình debug traceback

1. Đọc loại lỗi và thông báo ở dòng cuối.
2. Đi ngược lên, tìm file thuộc project thay vì thư viện.
3. Ghi lại input, shape, dtype và giá trị tham số tại dòng đó.
4. Tạo case nhỏ tái hiện lỗi.
5. Sửa nguyên nhân rồi chạy test; không chỉ bọc `try/except` để giấu lỗi.

### Bài tập

1. Mở `trake_search.py`, tìm giá trị mặc định của `candidate_k` và `per_video`.
2. Giải thích kiểu của `event_rows: list[list[dict[str, object]]]`.
3. Chạy `python trake_search.py --help`; xác định tham số bắt buộc.
4. Tạo lỗi có chủ ý bằng một đường dẫn metadata sai, rồi chỉ đúng dòng code project gây lỗi.

### Tự kiểm tra

Hoàn thành khi có thể giải thích một traceback mà không chỉ gửi ảnh lỗi cho AI.

Tài liệu: https://docs.python.org/3/tutorial/ và https://docs.python.org/3/library/traceback.html

---

## 2. Vector embedding

Embedding là hàm biến dữ liệu thành vector số:

```text
text_encoder("một con mèo đen") -> [0.12, -0.31, ...]
image_encoder(frame)             -> [0.10, -0.28, ...]
```

CLIP/SigLIP được huấn luyện để ảnh và câu có ý nghĩa tương ứng nằm gần nhau trong cùng không gian vector. Embedding không phải mô tả hoàn chỉnh và không đảm bảo hiểu đúng hành động, quan hệ hay vật thể rất nhỏ.

### Phải biết

- Dimension: số phần tử của vector.
- Normalize L2: chia vector cho độ dài của nó.
- Dense vector khác keyword/OCR sparse signal.
- Global embedding có thể bỏ qua chi tiết nhỏ; multi-crop bổ sung góc nhìn cục bộ.

### Bài tập

1. Dùng một ảnh xe máy và ba query: `xe máy`, `người phụ nữ`, `người phụ nữ chạy xe máy`. So sánh thứ hạng.
2. Crop riêng người và xe; xem global/crop nào trả query tốt hơn.
3. Tìm một frame đúng nhưng CLIP xếp thấp, ghi giả thuyết: resize, dịch query, hành động hay vật thể nhỏ.

Tài liệu: https://arxiv.org/abs/2103.00020 và https://www.sbert.net/

---

## 3. Cosine similarity và Top-k

Cosine similarity:

```text
cos(a,b) = (a dot b) / (||a|| * ||b||)
```

Nếu hai vector đã normalize L2 thì cosine bằng dot product. Top-k là lấy `k` vector có similarity cao nhất.

### Bài tập tính tay

Cho query `q=(1,0)` và ba frame đã normalize:

- A `(1,0)`
- B `(0.8,0.6)`
- C `(0,1)`

Tính dot product và xếp Top-3.

Đáp án: A = 1.0, B = 0.8, C = 0.0; thứ tự A, B, C.

### Bài tập code

```python
import numpy as np

q = np.array([1.0, 0.0])
x = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
scores = x @ q
print(np.argsort(-scores), scores)
```

Sửa `q`, dự đoán kết quả trước khi chạy.

---

## 4. Retrieval khác reasoning

### Retrieval

- CLIP/SigLIP2 encode query và frame.
- FAISS tìm vector gần nhất.
- Mục tiêu chính: không bỏ sót frame đúng trong Top-100.
- Nhanh, chạy trên toàn bộ kho dữ liệu, nhưng yếu ở đếm, hành động và quan hệ phức tạp.

### Reasoning

- Qwen-VL nhìn trực tiếp một số ảnh đã chọn.
- Trả lời QA hoặc xác minh chuỗi TRAKE.
- Chậm và tốn VRAM, nên không quét toàn bộ 177 nghìn frame.

### Bài tập

Phân loại các việc sau là retrieval hay reasoning:

1. Tìm 100 frame gần query nhất.
2. Đếm số người trong frame.
3. Kiểm tra người đàn ông có thật sự ngồi xuống không.
4. Tìm vector gần nhất bằng FAISS.

Đáp án: 1 retrieval; 2 reasoning; 3 reasoning; 4 retrieval.

Tài liệu: https://github.com/facebookresearch/faiss/wiki và https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

---

## 5. Precision, Recall và Recall@k

- `TP`: kết quả đúng được trả về.
- `FP`: kết quả sai nhưng được trả về.
- `FN`: kết quả đúng bị bỏ sót.
- `Precision = TP / (TP + FP)`.
- `Recall = TP / (TP + FN)`.
- `Recall@k`: trong Top-k có tìm thấy ground truth hay không; với một GT thường là 0 hoặc 1.

Ví dụ frame đúng đứng hạng 7:

- Recall@1 = 0
- Recall@5 = 0
- Recall@20 = 1
- Recall@50 = 1
- Recall@100 = 1
- Mean Top-k theo năm mốc = `3/5 = 0.6`

### Bài tập

Tính điểm nếu frame đúng nằm ở hạng 1, 4, 18, 70 và không có trong Top-100.

Đáp án Mean Top-k:

- Hạng 1: 1.0
- Hạng 4: 0.8
- Hạng 18: 0.6
- Hạng 70: 0.2
- Không có: 0.0

Kết luận: phải tối ưu Top-1 và Top-5, không chỉ nhét đáp án vào Top-100.

---

## 6. Batch, dtype, RAM/VRAM, CUDA và OOM

### Phải biết

- Batch size lớn thường nhanh hơn nhưng tốn bộ nhớ hơn.
- FP32: 4 byte/giá trị; FP16/BF16: 2 byte; INT8: 1 byte; INT4: 0.5 byte.
- Kích thước weight thô không phải tổng VRAM: còn activation, KV cache, CUDA context và workspace.
- CUDA dùng GPU; FAISS CPU và model CPU vẫn dùng RAM.
- `MemoryError`, `std::bad_alloc`, `CUDA out of memory` đều là thiếu bộ nhớ ở các tầng khác nhau.

### Quy trình xử lý OOM

1. Xác định thiếu RAM hay VRAM.
2. Giảm batch/candidate/image resolution.
3. Dùng FP16/INT4 nếu model hỗ trợ.
4. Không chạy đồng thời web, multi-crop, OCR và Qwen nếu máy không đủ RAM.
5. Dùng checkpoint/resume; không xóa tiến độ.

### Bài tập

1. Ước lượng weight 3 tỉ tham số: FP32 ≈ 12 GB; FP16 ≈ 6 GB; INT4 ≈ 1.5 GB thô.
2. Giải thích tại sao model INT4 1.5 GB vẫn có thể dùng hơn 2–3 GB VRAM.
3. Chạy `nvidia-smi` và xác định total/used/free VRAM.
4. Thử encode batch 1, 2, 4; ghi tốc độ và peak VRAM.

Tài liệu: https://pytorch.org/docs/stable/notes/cuda.html và https://huggingface.co/docs/transformers/quantization/bitsandbytes

---

## 7. KIS, QA và TRAKE

### KIS

Input: mô tả bằng văn bản. Output: `video_id, frame_idx`.

Mục tiêu: tìm đúng khoảnh khắc. Dùng CLIP/SigLIP2, object, OCR, metadata, fusion và rerank.

### QA

Input: câu hỏi hình ảnh. Output: `video_id, frame_idx, answer`.

Hai lỗi độc lập: tìm sai frame hoặc Qwen trả lời sai. Vì vậy phải benchmark candidate recall riêng với answer accuracy.

### TRAKE

Input: chuỗi N sự kiện. Output: `video_id, frame_1, ..., frame_N`.

Cần cùng video, đúng thứ tự, đúng từng hành động, khoảng thời gian hợp lý và thường cần cùng chủ thể/cảnh. Dùng dynamic programming/beam search rồi Qwen xác minh.

### Bài tập

1. Viết hai query KIS: một query vật thể và một query có quan hệ người–vật.
2. Viết QA hỏi số lượng, màu sắc và hành động.
3. Viết TRAKE ba event; tự chọn GT và chứng minh frame tăng dần chưa đủ để đúng.
4. Với mỗi task, viết đúng một dòng submission mẫu.

---

## 8. Benchmark bằng ground truth

Ground truth là đáp án được kiểm tra bằng mắt, không phải output model tự gán.

### Quy trình

1. Tạo tập query đại diện: dễ, vừa, khó; người, vật, hành động, chữ, màu, số lượng.
2. Gắn GT video và khoảng frame; QA thêm answer; TRAKE thêm khoảng cho từng event.
3. Chia development và hold-out. Không chỉnh weight trực tiếp trên hold-out.
4. Chạy baseline và lưu rank của GT.
5. Thay đúng một thành phần mỗi lần: OCR, object, SigLIP2, multi-crop, Qwen.
6. So sánh Recall@1/5/20/50/100, latency và peak VRAM.
7. Xem thủ công các query tụt hạng.

### Bài tập capstone benchmark

- Tạo ít nhất 5 KIS, 5 QA và 5 TRAKE có GT thật.
- Lưu kết quả baseline.
- Chạy ablation: CLIP-only, SigLIP2-only, OCR/object-only và fusion.
- Với mỗi lỗi, phân loại: query, embedding, candidate recall, fusion, temporal alignment hay Qwen.
- Chỉ tuyên bố hệ thống tốt hơn nếu metric hold-out tăng.

Trong project, luôn bắt đầu bằng:

```powershell
python evaluate_suite.py --help
python benchmark_candidates.py --help
python benchmark_live.py --help
```

Đọc `BENCHMARK.md` trước khi chạy bộ đánh giá lớn.

---

## 9. Git cơ bản để không phá code

### Các lệnh an toàn cần thuộc

```powershell
git status
git diff
git switch -c ten-branch
git add -- duong-dan-file
git commit -m "mo ta thay doi"
git log --oneline -10
git push -u origin ten-branch
```

### Quy tắc

- Luôn xem `git status` và `git diff` trước commit.
- `git add` đúng file, không mù quáng `git add .`.
- Mỗi commit chỉ giải quyết một vấn đề.
- Không commit index, model, dataset, cache hoặc secret.
- Không dùng `git reset --hard`, `git clean -fd` hay force push nếu chưa hiểu hậu quả.
- Code chạy được chưa đủ: phải có test hoặc bằng chứng benchmark.

### Bài tập

1. Tạo branch `practice/readme`.
2. Thêm một dòng vào file thử nghiệm.
3. Xem diff, commit đúng một file và xem log.
4. Tạo một thay đổi chưa commit rồi dùng `git diff` để giải thích chính xác nó làm gì.

Tài liệu: https://git-scm.com/book/en/v2

---

## 10. Bài tốt nghiệp nhỏ

Chọn một query hệ thống đang trả sai và viết báo cáo một trang:

1. Query và GT.
2. Rank GT ở CLIP-only.
3. Rank GT ở SigLIP2-only.
4. Rank GT ở OCR/object-only.
5. Rank sau fusion.
6. Qwen xác minh đúng hay sai.
7. Latency và peak VRAM.
8. Một thay đổi duy nhất được đề xuất.
9. Metric trước/sau trên toàn bộ benchmark, không chỉ query vừa sửa.

Hoàn thành bài này nghĩa là đã đủ kiến thức để làm kiến trúc sư/tester của project và dùng AI làm lập trình viên hỗ trợ.

## Checklist tối thiểu

- [ ] Đọc được traceback.
- [ ] Giải thích embedding và cosine similarity.
- [ ] Phân biệt retrieval với reasoning.
- [ ] Biết CLIP/SigLIP2, FAISS và Qwen làm gì.
- [ ] Tính được Recall@k và điểm năm mốc.
- [ ] Biết xử lý RAM/VRAM OOM mà không xóa checkpoint.
- [ ] Viết đúng output KIS, QA và TRAKE.
- [ ] Tạo benchmark có ground truth thật.
- [ ] So sánh ablation thay vì đánh giá cảm tính.
- [ ] Dùng Git branch/diff/commit an toàn.
