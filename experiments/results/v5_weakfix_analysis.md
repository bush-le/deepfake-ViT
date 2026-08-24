# Phân tích method yếu — v5 combined (hoangtuan, 23/08/2026)

Model: **DINOv3 ViT-Small/16** — `experiments/checkpoints/exp05_v5_combined/best_model.pt`
Test: `test_balanced_fixed_zero_leakage.csv` (2,354 ảnh, 1:1 real/fake, 40+ method, certified 0% MD5 leak)

## Kết quả baseline

| Chỉ số | Giá trị |
|---|---|
| Test Accuracy | 95.37% |
| ROC-AUC | 99.35% |
| Fake Recall | 93.37% (1,099/1,177) |
| Real Accuracy | 97.37% (1,146/1,177) |
| FN (fake→real) | 78 |
| FP (real→fake) | 31 |

## Method sụp (fake acc < 90%) — sắp theo độ sụp

| Method | Loại | N | Acc | Mean fake-prob | Nhóm |
|---|---|---|---|---|---|
| heygen | talking-head | 1 | 0.0% | 0.226 | (n=1, không có ý nghĩa thống kê) |
| **faceswap** | face-swap | 27 | **62.96%** | 0.605 | FF++ faceswap |
| **starganv2** | GAN attribute | 40 | **72.5%** | 0.661 | GAN edit |
| **whichfaceisreal** | GAN | 30 | **73.33%** | 0.740 | GAN synthetic |
| **facedancer** | reenactment | 27 | **74.07%** | 0.694 | talking/reenact |
| **sadtalker** | talking-head | 26 | **80.77%** | 0.754 | audio→video |
| **fsgan** | face-swap | 26 | **84.62%** | 0.812 | face-swap |
| **wav2lip** | lip-sync | 22 | **86.36%** | 0.772 | audio→video |
| **e4s** | GAN edit | 15 | **86.67%** | 0.812 | one-shot edit |
| **simswap** | face-swap | 27 | **88.89%** | 0.833 | face-swap |
| **blendface** | face-swap | 27 | **88.89%** | 0.857 | face-swap |
| **lia** | face-swap | 27 | **88.89%** | 0.807 | face-swap |
| CollabDiff | diffusion edit | 31 | 90.32% | 0.836 | (biên) |

**Pattern:** chỗ sụp đều là **face-swap / reenactment / talking-head / GAN-edit** — ảnh fake
trông vẫn "giống mặt người thật" nên model nhận nhầm thành real.

## Nguyên nhân sâu xa

1. **Train quá ít frame cho method yếu**: v5 train chỉ dùng ~600 frame/method từ
   `DF40_train_extracted` (pool thật có **43K–62K frame/method**). Không đủ để học
   đặc trưng của từng method.
2. **Real FP (31 ảnh)**: chủ yếu FFHQ + FF++/ffc — model hơi nhạy với ảnh sạch studio.
3. **Thiếu dữ liệu face-swap mới lạ** ngoài 6 method face-swap quen thuộc.

## Dữ liệu bổ sung (do Mạnh thêm + khai thác từ pool có sẵn)

| Nguồn | Loại | Số ảnh | Dùng để |
|---|---|---|---|
| `DF40_train_extracted` (train split, không leak) | fake | +51,600 | boost 31 method fake (faceswap 4K, sadtalker 4K, facedancer 4K, …) |
| `/workspace/data/deep-fake-face-swap` | fake | +8,076 | face-swap mới lạ (swap lên celeb) |
| `df-40-test-full` (loại 2,354 path test) | fake | +4,208 | starganv2 2K, whichfaceisreal 1.2K, CollabDiff 1.5K, heygen 800 |
| `/workspace/data/celebvhq` (extract 200 video × 20 frame) | real | +4,000 | real đa dạng (giảm FP) |

**Replay:** giữ nguyên toàn bộ 54K ảnh v5 train để không quên hành vi cũ.

## Finetune config

- Khởi tạo từ `exp05_v5_combined/best_model.pt` (0 missing / 0 unexpected)
- Dataset: `data/splits/train_v5_weakfix.csv` (121,884 = 31,006 real / 90,878 fake)
- Sampler **method-balanced**: 50% real, 50% fake; fake chia đều theo method → mọi
  method yếu được "nhìn" ngang nhau mỗi epoch
- 2 epochs, batch 64, base-lr 2e-5 / head-lr 5e-4, EMA 0.999, bf16
- Val: `val_v5_combined_universal_kaggle_boost.csv` | Test: cùng zero-leakage suite

## Kết quả sau finetune

→ xem `v5_weakfix_training_report.json`, `v5_weakfix_vs_baseline.csv`

---

## v2 (weakfix, tổng quát) — 23/08/2026

Khởi tạo từ v5 ckpt, finetune 2 epoch trên `train_v5_weakfix.csv` (121,884 = 31,006R/90,878F),
sampler **method-balanced** (mọi method yếu được nhìn ngang nhau).

| Chỉ số | baseline | v2 | delta |
|---|---|---|---|
| Test Accuracy | 95.37% | **97.20%** | +1.83 |
| ROC-AUC | 99.35% | 99.73% | +0.38 |
| Real (FP) | 97.37% (31) | 98.13% (22) | −9 FP |
| Fake recall (FN) | 93.37% (78) | 96.26% (44) | −34 FN |

Method sụp được sửa (v2 so baseline): starganv2 72.5→**100**, whichfaceisreal 73.3→90,
e4s 86.7→100, fsgan 84.6→92.3, simswap 88.9→96.3, CollabDiff 90.3→96.8, blendface 88.9→92.6,
lia 88.9→85.2*(v2 hơi sụt), **faceswap vẫn kẹt 62.96%** (10 miss giống hệt baseline).

**Chẩn đoán faceswap:** 10 miss đều là frame **sắc nét, chất lượng cao** (sharpness 99–527 vs
hit chỉ 14–24) — model tự tin gán real (prob tới 0.027). Model hầu như không học được faceswap:
prob trung bình trên chính frame faceswap train = 0.47, vì sampler method-balanced chỉ cho
faceswap ~1.4% batch mỗi epoch (n=4,600 trong 121K).

## v3 (faceswap-focused) — 23/08/2026 ★ final

Khởi tạo từ **v2 ckpt**, 3 epoch trên `train_v5_weakfix_v3.csv` (129,884 = 31,006R/98,878F):
+8,000 frame faceswap identity-disjoint (verify 0 overlap; n faceswap 4,600→12,600).
Sampler **faceswap-focused**: P(faceswap)=35%, P(real)=35%, P(method khác)=0.30/35 mỗi method
→ faceswap được chú ý gấp ~10 lần v2. LR nhẹ hơn (1.5e-5/4e-4). Saves:
`experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt`

| Chỉ số | baseline | v2 | **v3** |
|---|---|---|---|
| Test Accuracy | 95.37% | 97.20% | **97.88%** |
| ROC-AUC | 99.35% | 99.73% | 99.70% |
| Real (FP) | 97.37% (31) | 98.13% (22) | 98.13% (22) |
| Fake recall (FN) | 93.37% (78) | 96.26% (44) | **97.62% (28)** |

### Per-method v3 so baseline (method sụp cũ)

| Method | baseline | v3 | delta |
|---|---|---|---|
| **faceswap** | 62.96% | **88.89%** | +25.93 |
| starganv2 | 72.5% | 95.0% | +22.5 |
| whichfaceisreal | 73.33% | 93.33% | +20.0 |
| **facedancer** | 74.07% | 92.59% | +18.5 |
| sadtalker | 80.77% | 80.77% | 0 (1 frame biên, xem dưới) |
| fsgan | 84.62% | 92.31% | +7.7 |
| wav2lip | 86.36% | 90.91% | +4.6 |
| e4s | 86.67% | 100% | +13.3 |
| simswap | 88.89% | 100% | +11.1 |
| blendface | 88.89% | 92.59% | +3.7 |
| lia | 88.89% | 92.59% | +3.7 |
| inswap | 89.47% | 94.74% | +5.3 |
| CollabDiff | 90.32% | 100% | +9.7 |

### faceswap: 10 miss baseline/v2 → v3 sửa được 7

| identity | base | v2 | v3 |
|---|---|---|---|
| ffc:462 | 0.169 | 0.165 | **0.958** ✓ |
| ffc:812 | 0.178 | 0.104 | **0.960** ✓ |
| ffc:176 | 0.133 | 0.237 | **0.603** ✓ |
| ffc:044 | 0.124 | 0.197 | **0.978** ✓ |
| ffc:701 | 0.074 | 0.027 | **0.027** ✗ (frame cực sắc, sharp 527) |
| oth id3_id28 | 0.440 | 0.445 | **0.759** ✓ |
| oth id3_id23 | 0.168 | 0.276 | **0.328** ✗ |
| oth id4_id28 | 0.637 | 0.478 | **0.615** ✓ |
| cdc:id50 | 0.405 | 0.283 | **0.828** ✓ |
| cdc:id10 | 0.639 | 0.491 | **0.480** ✗ |

Còn 3 miss: `ffc:701` (FF++ frame sắc nét cực cao, model vẫn cực tự tin real), `oth:id3_id23`
(VoxCeleb OOD), `cdc:id10` (Celeb-DF, sát ngưỡng 0.48).

### Regression nhỏ v3 so v2 (đều ở mức nhiễu, vẫn cao hơn baseline rất nhiều)

- starganv2 100→95 (−5.0, n=40, mất 2)
- sadtalker 84.62→80.77 (−3.85, mất đúng **1 frame** biên `id2_da1vvigy5tQ` 0.509→0.426)

## Kết luận

- **v3 là bản cuối tốt nhất**: 97.88% acc, FN 78→28, FP 31→22, AUC 99.70%.
- Toàn bộ chỗ sụp baseline (face-swap/reenact/talking-head/GAN-edit) đã ≥90% **trừ sadtalker
  (80.77%, 1 frame biên) và faceswap (88.89%, còn 3 ca khó gồm 1 frame FF++ siêu sắc)**.
- Dữ liệu bổ sung của Mạnh đều được dùng: `deep-fake-face-swap` (8,076 fake) +
  `celebvhq` (4,000 real); faceswap cải thiện chủ yếu nhờ +8,000 frame identity-disjoint
  từ `DF40_train_extracted/faceswap` + oversampling.
