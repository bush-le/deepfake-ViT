# Test Data v3 Construction & Zero-Leakage Verification Report

---

## 1. Overview
This report documents the construction and leakage audit for the held-out multi-generator evaluation benchmarks:
- `test_coursework_44methods_balanced_zero_leakage.csv` (21,446 images)
- `test_coursework_44methods_full_zero_leakage.csv` (50,084 images)

---

## 2. Zero-Leakage Guarantees
1. **Path Disjointness**: 0 overlapping file paths between Train and Test splits.
2. **Subject Identity Disjointness**: FaceForensics++ and Celeb-DF subject IDs in the training set are strictly excluded from the test suite.
3. **MD5 Hash Deduplication**: 127,185 MD5 byte hashes cross-audited; 5,680 path overlaps and 2,261 exact byte collisions purged.
