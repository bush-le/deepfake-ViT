# EXP-03 STATUS — Zero-Leakage Dataset Rebuild & Balanced Training

- **Title:** Tái cấu trúc dataset chuẩn Zero-Leakage & Huấn luyện DINOv3 ViT
- **Date created:** 2026-08-22
- **Last updated:** 2026-08-22
- **Description:** Theo dõi tiến độ EXP-03 nhằm giải quyết triệt để rò rỉ dữ liệu (495 ảnh MD5 hash & 697 ID FF++), bổ sung mẫu EFS/MidJourney vào train và tối ưu hóa accuracy.
- **Status:** In Progress (Dataset Rebuilt & Gatekeeper Verified 100% Clean)
- **Experiment doc:** [../experiments/EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md](../experiments/EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md)

## Log

- 2026-08-22: Hoàn thành audit rò rỉ dữ liệu của EXP-02 (phát hiện 495 ảnh Test trùng MD5 với Train, 697 ID FF++ trùng lặp).
- 2026-08-22: Thiết lập kế hoạch EXP-03 với quy tắc 3 Không (Zero Hash, Zero Identity, Zero Video Leakage).
- 2026-08-22: Xây dựng thành công bộ dữ liệu V3 sạch (`train_v3_clean.csv`: 50,000 ảnh 1:1, `val_v3_clean.csv`: 5,000 ảnh 1:1).
- 2026-08-22: **GATEKEEPER VERIFICATION PASSED (100% Clean ✔)**:
  - Exact Path Overlap: **0**
  - MD5 Hash Matches in Test: **0**
  - Balance Ratio: **1.0 : 1.0 (Exact 25K/25K Real/Fake)**
- 2026-08-22: Chạy Smoke Test thành công cho `src/training/train_exp03.py`.

## Current Dataset Baseline (EXP-03 Clean Split)

| Split | Total Samples | Real | Fake | Hash Overlap with Test |
|:---|:---:|:---:|:---:|:---:|
| **Train V3 Clean** | 50,000 | 25,000 | 25,000 | **0 (Zero Leak ✔)** |
| **Val V3 Clean** | 5,000 | 2,500 | 2,500 | **0 (Zero Leak ✔)** |
| **Test Balanced (Frozen)** | 4,134 | 2,067 | 2,067 | Benchmark Chuẩn |

## Next Steps

1. [x] Tạo `src/data/build_dataset_v3_clean.py` & xuất CSV V3.
2. [x] Chạy `src/data/verify_zero_leakage.py` (Passed 100%).
3. [x] Tạo `src/training/train_exp03.py` & Smoke Test Passed.
4. [ ] Khởi chạy full training EXP-03 (10 Epochs với ModelEMA + LLRD + CosineAnnealing).
5. [ ] Đánh giá toàn diện trên test benchmark và trực quan hóa kết quả.
