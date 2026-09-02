# TASK: Phân tích lý thuyết và so sánh mô hình trong project

Hãy **scan toàn bộ project hiện tại** (source code, config, README, training scripts, dataset preparation, experiment logs, model definitions và các file liên quan) để hiểu chính xác project đang sử dụng mô hình, dữ liệu, pipeline và các thí nghiệm nào.

Sau đó tạo một file Markdown hoàn chỉnh:

`THEORY_AND_MODEL_COMPARISON.md`

Mục tiêu: viết phần **lý thuyết ngắn gọn, có tính phân tích**, phục vụ báo cáo/đồ án và có thể **export trực tiếp sang PDF**.

## 1. Nguyên tắc viết

* Không viết textbook dài.
* Ưu tiên **conceptual understanding + comparison + practical relevance**.
* Chỉ giải thích những kiến thức thực sự liên quan đến project.
* Mỗi phần lý thuyết nên trả lời được:

  * What: mô hình/cơ chế là gì?
  * Why: tại sao cần nó?
  * How: hoạt động ở mức khái niệm như thế nào?
  * When: khi nào phù hợp?
* Không sa đà vào chứng minh toán học.
* Công thức chỉ đưa vào nếu giúp giải thích bản chất.
* Không mô tả code từng dòng.
* Không tự bịa thông tin. Nếu một thông tin không thể xác định từ project, ghi rõ là "not specified in the project".
* Các nhận xét về performance phải dựa trên **experiment/log/metric thực tế trong project**, không được suy đoán.

---

# 2. Tổng quan lý thuyết

Viết phần lý thuyết ngắn gọn về các kiến trúc/mô hình thực sự xuất hiện trong project.

Với mỗi mô hình, trình bày theo cấu trúc:

### [Model name]

**Intuition:**
Giải thích trực giác trong 2–4 câu.

**Core mechanism:**
Giải thích cơ chế chính ở mức conceptual.

**Strengths:**

* ...
* ...

**Weaknesses:**

* ...
* ...

**Best suited for:**

* Loại dữ liệu nào?
* Quy mô dữ liệu nào?
* Bài toán nào?

Không cần giải thích những architecture không xuất hiện hoặc không liên quan đến project.

---

# 3. So sánh các mô hình

Đây là phần QUAN TRỌNG NHẤT.

Tạo bảng so sánh các architecture/model có trong project.

Ít nhất xem xét các khía cạnh:

| Aspect                      | CNN | ViT / Transformer | Model đang dùng |
| --------------------------- | --- | ----------------- | --------------- |
| Inductive bias              |     |                   |                 |
| Local vs global information |     |                   |                 |
| Feature extraction          |     |                   |                 |
| Data requirement            |     |                   |                 |
| Computational cost          |     |                   |                 |
| Training difficulty         |     |                   |                 |
| Scalability                 |     |                   |                 |
| Generalization              |     |                   |                 |
| Strength                    |     |                   |                 |
| Weakness                    |     |                   |                 |
| Suitable data               |     |                   |                 |

Nếu project có các CNN cụ thể như ResNet/DenseNet hoặc architecture khác, hãy thêm chúng vào comparison.

Không được viết kiểu:
"Transformer tốt hơn CNN."

Thay vào đó phải giải thích **trong điều kiện nào Transformer/ViT tốt hơn và trong điều kiện nào CNN tốt hơn**.

Ví dụ cách phân tích mong muốn:

* CNN thường có inductive bias mạnh đối với spatial locality và translation equivariance → thường hiệu quả khi dữ liệu ảnh không quá lớn.
* ViT có global self-attention và inductive bias yếu hơn → thường cần nhiều dữ liệu hoặc pretraining tốt để phát huy lợi thế.
* Khi dữ liệu đủ lớn hoặc sử dụng pretrained model mạnh, ViT có thể học global relationships tốt hơn.
* Với dataset nhỏ, CNN hoặc pretrained ViT có thể phù hợp hơn tùy mức độ pretraining và domain shift.

---

# 4. Transformer/ViT vs CNN: Khi nào tốt hơn / xấu hơn?

Tạo một subsection riêng:

## When ViT is better than CNN

Đưa ra các trường hợp cụ thể, ví dụ:

* dataset lớn
* pretrained foundation model
* cần modeling global dependencies
* dữ liệu có quan hệ xa trong không gian
* scaling model/data

Giải thích WHY.

## When CNN is better than ViT

Đưa ra các trường hợp:

* dataset nhỏ
* cần inductive bias spatial mạnh
* computational resources hạn chế
* bài toán chủ yếu phụ thuộc local patterns
* training from scratch

Giải thích WHY.

## Important caveat

Nhấn mạnh:

> Không có kiến trúc nào luôn tốt hơn kiến trúc còn lại.

Performance phụ thuộc vào:

* dataset size
* data quality
* pretraining
* augmentation
* model scale
* compute
* optimization
* domain similarity

Nếu project có experiment trực tiếp CNN vs ViT, phải liên hệ kết quả thực tế vào phần này.

---

# 5. Model phù hợp với loại dữ liệu nào?

Tạo bảng:

| Model       | Data characteristics | Why |
| ----------- | -------------------- | --- |
| CNN         | ...                  | ... |
| ResNet      | ...                  | ... |
| DenseNet    | ...                  | ... |
| ViT         | ...                  | ... |
| Transformer | ...                  | ... |

Chỉ đưa những model liên quan đến project.

Phân tích theo:

* Image
* Text
* Sequential data
* Spatial data
* Temporal data
* Structured/tabular data

Không cần mở rộng sang domain không liên quan nếu project không sử dụng.

---

# 6. Ảnh hưởng của kích thước dataset

Tạo phần:

## Small Dataset

Giải thích:

* model nào thường có lợi thế?
* tại sao?
* pretraining ảnh hưởng thế nào?
* nguy cơ overfitting?
* augmentation có vai trò gì?

## Large Dataset

Giải thích:

* tại sao Transformer/ViT có thể phát huy tốt hơn?
* scaling có ý nghĩa gì?
* data diversity quan trọng thế nào?

## Pretrained vs From Scratch

So sánh riêng:

| Setting                            | Expected behavior |
| ---------------------------------- | ----------------- |
| Small data + training from scratch |                   |
| Small data + pretrained CNN        |                   |
| Small data + pretrained ViT        |                   |
| Large data + training from scratch |                   |
| Large data + pretrained model      |                   |

Không được khẳng định một mô hình luôn thắng. Phải giải thích điều kiện.

---

# 7. Liên hệ trực tiếp với project

Sau phần lý thuyết tổng quát, tạo một section:

# Application to This Project

Scan project và trả lời:

1. Project đang giải quyết bài toán gì?
2. Input data là gì?
3. Dataset size bao nhiêu?
4. Data distribution như thế nào?
5. Model nào đang được sử dụng?
6. Model có pretrained không?
7. Vì sao architecture này phù hợp với project?
8. Những architecture nào có thể là baseline hợp lý?
9. CNN và ViT khác nhau thế nào trong chính bài toán này?
10. Nếu dataset nhỏ hơn thì lựa chọn model có thay đổi không?
11. Nếu dataset lớn hơn thì lựa chọn model có thay đổi không?
12. Bottleneck chính là data, architecture, compute hay optimization?

Nếu project có experiment/benchmark, đưa số liệu thực tế vào đây.

Ví dụ:

> Model A đạt X% accuracy, trong khi Model B đạt Y%. Điều này cho thấy ... trong điều kiện dataset/configuration hiện tại.

Không được suy diễn causal relationship nếu experiment không đủ để chứng minh.

---

# 8. Practical decision guide

Tạo một decision table ngắn:

| Situation                               | Recommended choice | Reason |
| --------------------------------------- | ------------------ | ------ |
| Very small dataset                      |                    |        |
| Small dataset + strong pretrained model |                    |        |
| Medium dataset                          |                    |        |
| Large dataset                           |                    |        |
| Huge dataset + sufficient compute       |                    |        |
| Mostly local patterns                   |                    |        |
| Strong global dependencies              |                    |        |
| Limited GPU                             |                    |        |
| Strong pretrained ViT available         |                    |        |

Mục đích là giúp người đọc **ra quyết định architecture**, không chỉ học thuộc định nghĩa.

---

# 9. Common misconceptions

Thêm phần ngắn:

### Misconception 1

"Transformer/ViT luôn tốt hơn CNN."

→ Giải thích tại sao sai.

### Misconception 2

"Dataset càng nhỏ thì không thể dùng ViT."

→ Giải thích vai trò của pretraining.

### Misconception 3

"Model lớn hơn luôn tốt hơn."

→ Giải thích trade-off giữa capacity, data và compute.

### Misconception 4

"Accuracy cao hơn nghĩa là architecture tốt hơn trong mọi trường hợp."

→ Giải thích dependency vào dataset và experimental setup.

Chỉ thêm misconception thực sự liên quan.

---

# 10. Final takeaway

Kết thúc bằng khoảng **5–8 bullet points**, mỗi bullet chỉ 1–2 câu.

Phải trả lời được:

* CNN mạnh ở đâu?
* ViT/Transformer mạnh ở đâu?
* ViT yếu ở đâu?
* CNN yếu ở đâu?
* Dataset size ảnh hưởng thế nào?
* Pretraining thay đổi điều gì?
* Project này chọn model hiện tại vì lý do gì?
* Khi nào nên đổi sang architecture khác?

---

# 11. Yêu cầu về chất lượng Markdown

File phải có cấu trúc:

# Theory and Model Comparison

## 1. ...

## 2. ...

## 3. ...

Sử dụng:

* Markdown headings
* tables
* bullet points
* inline formulas nếu cần
* Mermaid diagram nếu thực sự giúp hiểu architecture/pipeline

Không chèn HTML phức tạp.

Không sử dụng emoji.

Không tạo nội dung quá dài. Mục tiêu khoảng **5–10 trang PDF**, tùy lượng model thực tế trong project.

Ưu tiên:
**Concept → Comparison → Data requirement → Practical implication → Project-specific conclusion**

Không biến file thành một chương textbook.

---

# 12. Final validation

Trước khi hoàn thành file, kiểm tra:

* [ ] Tất cả model được nhắc đến đều thực sự tồn tại hoặc liên quan đến project.
* [ ] Không hallucinate dataset/model/metric.
* [ ] Có comparison CNN vs ViT/Transformer nếu project liên quan.
* [ ] Có phân tích "khi nào tốt hơn / khi nào xấu hơn".
* [ ] Có phân tích small vs large dataset.
* [ ] Có phân tích pretrained vs from scratch.
* [ ] Có bảng model → suitable data.
* [ ] Có liên hệ với experiment thực tế.
* [ ] Không dùng câu tuyệt đối kiểu "Transformer luôn tốt hơn CNN".
* [ ] Phần lý thuyết ngắn, tập trung vào reasoning.
* [ ] Markdown có thể export trực tiếp sang PDF.

Cuối cùng, chỉ tạo nội dung của:

`THEORY_AND_MODEL_COMPARISON.md`

# ADDITIONAL REQUIREMENT — CHARTS & VISUAL EVIDENCE

## Strict rule for charts

Khi phân tích hoặc minh họa kết quả, **chỉ được sử dụng các chart/figure đã được tạo ra từ các Jupyter Notebook thuộc coursework trong project**.

### Không được:

* Tự tạo chart mới bằng Python/Matplotlib/Seaborn.
* Tạo chart dựa trên các số liệu đã đọc từ log nếu chart đó chưa tồn tại trong coursework notebook.
* Lấy chart từ Internet, paper, documentation hoặc nguồn bên ngoài.
* Tự vẽ lại một chart có sẵn dưới hình thức khác.
* Sử dụng chart giả lập hoặc minh họa bằng số liệu tự tạo.
* Tự suy diễn một biểu đồ từ dataset nếu coursework notebook chưa tạo biểu đồ đó.

### Được phép:

* Scan toàn bộ các `.ipynb` trong project.
* Xác định các chart/figure thực sự được tạo trong coursework notebooks.
* Sử dụng **chính các chart đó** làm evidence cho phần phân tích.
* Tham chiếu đến notebook và section/cell tương ứng nếu xác định được.
* Crop/extract figure từ notebook nếu cần để đưa vào Markdown/PDF, nhưng **không thay đổi nội dung hoặc dữ liệu của figure**.

## Chart selection

Chỉ chọn những chart thực sự hỗ trợ lập luận, ví dụ:

* Dataset distribution
* Class distribution
* Training/validation loss
* Training/validation accuracy
* Confusion matrix
* ROC/PR curve
* Model comparison
* Performance comparison
* Data visualization
* Ablation/experiment results

Không cố đưa tất cả chart vào báo cáo.

## Evidence-based analysis

Mỗi khi sử dụng một chart để đưa ra nhận xét, phải phân biệt rõ:

**Observation:** Điều gì thực sự nhìn thấy từ chart?

**Interpretation:** Điều đó có thể cho thấy điều gì?

**Limitation:** Chart có đủ bằng chứng để kết luận hay không?

Không được biến correlation thành causation.

Ví dụ:

> Chart cho thấy ViT đạt validation accuracy cao hơn CNN trong experiment này.

Không viết:

> Chart chứng minh ViT luôn tốt hơn CNN.

Nếu chart chỉ thể hiện một experiment cụ thể, kết luận phải giới hạn trong **experimental setting của coursework**.

## Coursework priority

Khi có sự khác biệt giữa:

* thông tin từ source code,
* training logs,
* README,
* notebook,
* và chart,

hãy ưu tiên **kết quả thực nghiệm được thể hiện trong coursework notebooks** khi viết phần phân tích trực quan.

Mọi chart được sử dụng trong `THEORY_AND_MODEL_COMPARISON.md` phải có thể truy xuất về một notebook coursework cụ thể.

Nếu không tìm thấy chart phù hợp trong coursework notebooks:

> Không tự tạo chart mới.

Thay vào đó, viết phân tích bằng text/table hoặc ghi rõ:

> "No corresponding coursework chart was found."

## Final chart audit

Trước khi hoàn thành Markdown, kiểm tra:

* [ ] Tất cả chart đều xuất phát từ coursework notebooks.
* [ ] Không có chart tự-generated.
* [ ] Không có chart lấy từ Internet.
* [ ] Không có số liệu giả lập.
* [ ] Không thay đổi dữ liệu của chart gốc.
* [ ] Mỗi chart có context giải thích ngắn gọn.
* [ ] Các conclusion từ chart không vượt quá evidence.
* [ ] Có thể truy xuất mỗi chart về notebook nguồn.
