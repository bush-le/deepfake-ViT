# Running Anti-Deepfake Project on RunPod (Ubuntu + GPU Cloud)

**Objective**: Execute comprehensive GPU evaluation and fine-tuning of Meta DINOv3 on high-performance cloud GPUs.
**Standard Pipeline**: **Local Push to Hugging Face Hub → RunPod Pull → Setup Environment → Execute Benchmarks / Training**.

---

## 1. Local Machine — Push to Hugging Face Hub (One-time Setup)

### 1.1 Install CLI & Authenticate

```bash
.venv/bin/pip install -U "huggingface_hub[cli]"
.venv/bin/hf login        # Enter token (https://huggingface.co/settings/tokens, with WRITE permission)
```

### 1.2 Create Repositories (One-time)

```bash
.venv/bin/hf repo create dinov3-deepfake-detection --type model --private
.venv/bin/hf repo create df40-test-data-v3          --type dataset --private
```

### 1.3 Push Artifacts (Auto-excludes cache and large unneeded files)

```bash
bash src/utils/push_to_hub.sh              # Code + Weights (298MB) -> model repo
bash src/utils/push_dataset_to_hub.sh      # test_data_v3.zip (4.2GB) -> dataset repo
```

---

## 2. Provisioning RunPod Instance

1. **Deploy → GPU Pod → On-Demand / Secure Cloud**.
2. **Template**: Select **RunPod PyTorch** (Ubuntu 22.04 + CUDA + PyTorch).
3. **GPU Recommendation**: RTX 4090 (~$0.79/hr) or A10 (~$0.69/hr) is optimal for fast training and batch inference.
4. **Storage (Optional)**: Attach ~10GB Network Volume if persistent caching is desired.
5. **Start Pod → Connect → Web Terminal** (or SSH).

---

## 3. Inside the Pod — Pull, Configure & Execute

### 3.1 Pull Code & Checkpoints

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_xxx            # or run hf login
git lfs install
git clone https://huggingface.co/ManhQuangAI/dinov3-deepfake-detection
cd dinov3-deepfake-detection
```

### 3.2 Pull Datasets

```bash
hf download ManhQuangAI/df40-test-data-v3 --repo-type dataset --include test_data_v3.zip
unzip test_data_v3.zip -d .
```

### 3.3 Setup Python Environment

```bash
# If using RunPod PyTorch template (PyTorch + CUDA pre-installed):
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If starting from pure Ubuntu template:
bash src/utils/setup_ubuntu.sh
```

Verify GPU availability:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

### 3.4 Execute Evaluation & Benchmarks

```bash
# Evaluate DINOv3 ViT & ConvNeXt on 44-methods zero-leakage benchmark
python scripts/eval_v5_weakfix_v3_report.py
```

### 3.5 Fine-Tuning Models on Cloud GPU

```bash
# DINOv3 ViT-Small/16 Fine-Tuning
python src/training/train.py \
  --train-csv data/splits/train_v5_weakfix_v3.csv \
  --val-csv data/splits/val_v5_combined_universal_kaggle_boost.csv \
  --epochs 5 --batch-size 64 --amp --num-workers 4
```

---

## 4. Retrieving Results Back to Local Machine

```bash
# Export and push results to Hugging Face or download via SCP/rsync
zip -r runpod_results.zip experiments/results/ experiments/plots/
# Download via RunPod File Browser or SCP
```
