# FINETUNE.md — End-to-End Fine-Tune (DINOv3 / LoRA) on DF40

- **Motivation/Background**: Fine-tuning a real-vs-fake deepfake detector needs
  both real and fake face frames. This runbook captures the corrected,
  reproducible flow after the data/tool mismatch issues surfaced in testing.
- **Purpose**: One document to go from downloaded raw data → split CSVs →
  fine-tune → evaluate, on Windows (PowerShell) with a CUDA GPU.
- **Overview Pipeline**: Real frames (FF++ + Celeb-DF) + fake frames (from
  `DF40_train` zips) → build train/val/test CSVs → LoRA/full fine-tune →
  evaluate.
- **Detailed Plan**: §1 prerequisites; §2 real frames; §3 fake extraction;
  §4 CSVs; §5 test CSV; §6 fine-tune; §7 evaluate; §8 caveats.
- **References**: [src/data/*](../src/data/), [src/training/*](../src/training/),
  [src/eval/*](../src/eval/), [MODELS.md](MODELS.md), [README.md](README.md).

---

## Table of Contents

- [1. Prerequisites & Data Layout](#1-prerequisites--data-layout)
- [2. Real Frames (FF++ + Celeb-DF)](#2-real-frames-ff--celeb-df)
- [3. Extract Fake Frames from DF40_train Zips](#3-extract-fake-frames-from-df40_train-zips)
- [4. Build Train/Val CSVs](#4-build-trainval-csvs)
- [5. Build Test CSV](#5-build-test-csv)
- [6. Run the Fine-Tune](#6-run-the-fine-tune)
- [7. Evaluate the Fine-Tuned Model](#7-evaluate-the-fine-tuned-model)
- [8. Caveats & Leakage](#8-caveats--leakage)

---

## 1. Prerequisites & Data Layout

Assume already present (see [MODELS.md](MODELS.md)):

- `experiments/checkpoints/weights/` — the 3 pretrained backbones
- `data/raw/DF40/` — `DF40_train` zips (one per method; **fake-only**)
- `data/raw/real-root/original_sequences/youtube/c23/frames/` — FF++ real frames
- `test_data_v3/` — test set (real + fake) with `manifest.csv`
- `venv/` — Python env with `torch` (CUDA), `timm`, `PIL`

Check the env:

```powershell
venv\Scripts\python.exe -c "import torch,timm,PIL; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```

> **Windows notes:** run from the repo root; use `--num-workers 0` to avoid
> multiprocessing/spawn issues.

## 2. Real Frames (FF++ + Celeb-DF)

**Key lesson:** training on FF++ real only makes the model learn a
"looks-like-FF++ = real" shortcut → poor accuracy on Celeb-DF real (`cdc`).
Use **both** FF++ and Celeb-DF real.

Create a **combined real root** and link the FF++ identities into it:

```powershell
$comb = "data/raw/real-root/frames/real"
New-Item -ItemType Directory -Force -Path $comb | Out-Null
Get-ChildItem "data/raw/real-root/original_sequences/youtube/c23/frames" -Directory |
  ForEach-Object { New-Item -ItemType Junction -Path (Join-Path $comb $_.Name) -Target $_.FullName -ErrorAction SilentlyContinue | Out-Null }
```

**Add Celeb-DF real** (best: get frames from a teammate who has them; or
download + extract into the same dir):

```powershell
Get-ChildItem <celebdf_real_videos> -Filter *.mp4 |
  ForEach-Object { venv\Scripts\python.exe src/data/extract_real_frames.py --videos $_.FullName --out data/raw/real-root/frames/real --step 6 --size 256 }
```

> `gdown --folder` for the official Celeb-DF Drive link is unreliable (50-item
> limit) — prefer getting the frames directly.

## 3. Extract Fake Frames from DF40_train Zips

```powershell
venv\Scripts\python.exe src/data/extract_df40_local.py --zip-dir data/raw/DF40 --out data_train_local `
  --pairs 30 --seed 42 `
  --methods faceswap facedancer inswap fsgan simswap blendface pixart DiT uniface SiT lia mobileswap MRAA e4s
```

→ `data_train_local/<method>/fake/<pair>/...` + `_extract_manifest.json`
(~13k fake images; ~30–60 min).

## 4. Build Train/Val CSVs

```powershell
venv\Scripts\python.exe src/data/build_data_train_finetune.py `
  --extract data_train_local --out data_train `
  --real data/raw/real-root/frames/real --n-real 13000 --seed 42 `
  --test-manifest test_data_v3/manifest.csv `
  --no-leak-manifest test_data_v3/manifest.csv `
  --ffc-real-manifest test_data_v3/manifest.csv
```

→ `data_train\train.csv` + `val.csv`. The `--no-leak-manifest` /
`--ffc-real-manifest` flags keep eval-test identities out of training.

## 5. Build Test CSV

```powershell
venv\Scripts\python.exe -c "import csv; rows=[(r['path'],r['label']) for r in csv.DictReader(open('test_data_v3/manifest.csv'))]; open('data_train/test.csv','w',newline='').write('path,label\n'+''.join(f'test_data_v3/{p},{l}\n' for p,l in rows))"
```

## 6. Run the Fine-Tune

**LoRA (fastest, train/val only):**
```powershell
venv\Scripts\python.exe src/training/finetune_lora.py `
  --train-csv data_train\train.csv --val-csv data_train\val.csv `
  --lora-rank 16 --lora-alpha 32 --epochs 5 --amp --num-workers 0 --device cuda
```
→ `experiments/results/finetune/vit_lora_finetuned.pt`

**Full fine-tune (backbone + head):**
```powershell
venv\Scripts\python.exe src/training/train.py `
  --train-csv data_train\train.csv --val-csv data_train\val.csv --test-csv data_train\test.csv `
  --epochs 5 --amp --num-workers 0 --device cuda
```
→ `experiments/results/checkpoints/dinov3_finetuned.pt`

**ViT-vs-CNN (matched params):**
```powershell
venv\Scripts\python.exe src/training/finetune_compare.py --model-type vit `
  --train-csv data_train\train.csv --val-csv data_train\val.csv --test-csv data_train\test.csv `
  --epochs 5 --amp --num-workers 0 --device cuda
# repeat with --model-type cnn
```
→ `experiments/results/finetune/vit_finetuned.pt` / `cnn_finetuned.pt`

## 7. Evaluate the Fine-Tuned Model

**Threshold analysis (LoRA):**
```powershell
venv\Scripts\python.exe src/eval/analyze_threshold.py `
  --ckpt experiments/results/finetune/vit_lora_finetuned.pt --root test_data_v3 --device cuda
```

**Fine-tuned eval (full):**
```powershell
venv\Scripts\python.exe src/eval/eval_finetuned.py --ckpt experiments/results/checkpoints/dinov3_finetuned.pt --device cuda
```

> Eval checkpoint defaults now match training save locations
> (`experiments/results/...`). `analyze_threshold.py` supports `--device`.

## 8. Caveats & Leakage

- **Report balanced metrics, not raw accuracy.** The eval-test split is heavily
  fake-heavy (real ~349 / fake ~8,883) — use `bal_acc` (Youden) and per-domain
  breakdowns.
- **`data_train/test.csv` reuses `test_data_v3` identities.** With the
  leak-protection flags above, those identities are excluded from training real
  frames, so the eval is meaningful — but it is the overall-test accuracy, not
  the strict identity-disjoint protocol. For the formal number use
  `eval_identity_disjoint.py` (linear probe) / `eval_finetuned_identity_disjoint.py`.
- **Domain shortcut:** if Celeb-DF real was not in training, `real/cdc` accuracy
  collapses. Always include both FF++ + Celeb-DF real (see §2).
- **Runtime** (RTX 4060): ~2 h for a 5-epoch LoRA run; reduce `--pairs` /
  `--n-real` / `--epochs` for faster iterations.

---

## References

- [MODELS.md](MODELS.md) — downloading backbones + test data
- [README.md](README.md) — environment setup, all run commands
- [src/data/extract_df40_local.py](../src/data/extract_df40_local.py)
- [src/data/build_data_train_finetune.py](../src/data/build_data_train_finetune.py)
- [src/training/finetune_lora.py](../src/training/finetune_lora.py)
- [src/eval/analyze_threshold.py](../src/eval/analyze_threshold.py)
