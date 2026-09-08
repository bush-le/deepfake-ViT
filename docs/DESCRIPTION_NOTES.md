# DESCRIPTION_NOTES.md — Coursework Rubric & Submission Notes

- **Motivation/Background**: Capture specific coursework submission criteria, deliverables checklist, and project description notes.
- **Purpose**: Ensure 100% compliance with course requirements and grading criteria.
- **Overview Pipeline**: Rubric inspection -> deliverable mapping -> verification checklist.
- **Detailed Plan**: §1 Coursework Objective; §2 Key Deliverables Checklist; §3 Verification Requirements; §4 Report Guidelines.
- **References**: `docs/PURPOSE.md`, `docs/OVERVIEW.md`.
- **Created**: 2026-09-06T17:50:10+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

1. Data - Data ban đầu (train:https://huggingface.co/datasets/ManhQuangAI/DF40_train/tree/main - test ban đầu: https://huggingface.co/datasets/ManhQuangAI/df-40-test-full/tree/main, real: FF++: https://drive.google.com/file/d/1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb/view?usp=drive_link và Celeb thì tự tải) - phân tích số lượng chung, các method, nhóm các method, biết phân bố data - dùng cột, hist vân vân hạn chế dùng biểu đồ tròn.

2. Xây dựng bộ test chứa nhiều nhất các method - quan trọng - đây là thước đo tốt nhất: https://huggingface.co/datasets/bushle/deepfake_train_129k_images/tree/main - trình bày cách xây dựng tập data này - tại sao xây dựng nó? cách thức để tránh bị lỗi như leak để có tập data test an toàn dùng mãi về sau. - test cả tập toàn real để xem nó có bị bệnh real thành Fake không? https://huggingface.co/datasets/ManhQuangAI/df40-test-data-v3

2.1 hỏi nó xem - phân hoạch ổn chưa - khá ổn thì được rồi

2.2 trình bày về model

3. Đánh giá pretrained trên các test để biết model yếu ở đâu? cải thiện thế nào? từ đó mới biết chỗ mà finetune
3.1 Chuẩn bị data test - data được tạo ra sao? bổ sung thế nào? - yếu ở đâu bổ sung data ở các nguồn nào. 
4. Finetune (ra sao) và đánh giá mô hình!

-------------------
EDA ví dụ
session 1: Thực hiện thao tác với dữ liệu

1. Thống kê và dữ liệu

- (1) số lượng data gốc DF40, mô tả thành cấu thành DF40 (nghĩa là từ real của FF++ và Celeb v2 - thống kê cả cái này), real fake ra sao.
- (2) Chia method của DF40 thành các nhóm deepFake data dễ so sánh.
- (3) EDA, toàn bộ gạch đầu hàng (1), mô tả size, pixel, biểu đồ hist, cột, grid visual
- Vấn đề của bộ data là gì? - theo tôi là data frames by frames có số lượng lớn nhưng dễ bị trùng identity - trùng identity thì cũng ko phải vấn đề lớn nếu data video nhưng mà ảnh đơn và muốn tổng quát hoá thì nên làm identity độc nhất.
- (4) Cách chia tập train, val, test - làm lại (1) (2) (3) với các tập đã chia (thông tin data lấy ở các file liên quan - tôi đã huấn luyện finetune 2 model mạnh nhất là DINO viT và DINO ConNext)
- Tôi nghĩ là data sẽ leak nhẹ nhưng ổn - quan trọng là train ko leak với test

*Notice*
Cách thức: sinh hình ảnh và viết và .md kèm hình. Viết có trình tự.
