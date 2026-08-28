"""
Script to build the scaled-up, zero-leakage test set strictly containing Quang Manh's coursework methods.
No extra OOD methods added (No CelebDF, No DFF, No Kaggle Synth).
Scale sample counts per method from ~28 up to 300 - 1,500 clean images per method.
"""

import os, sys, glob, json, hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

HT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("DF40_ROOT", HT / "data"))

TRAIN_CSV = HT / 'data/splits/train_v5_weakfix_v3.csv'
TRAIN_HASH_FILE = HT / 'data/splits/train_v5_weakfix_v3_hashes.json'
OUT_DIR = DATA / 'zero_leakage_benchmark_fixed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load train hashes & paths
if TRAIN_HASH_FILE.exists():
    with open(TRAIN_HASH_FILE, 'r') as f:
        train_hashes = set(json.load(f))
    print(f'Loaded {len(train_hashes):,} train MD5 hashes.')
else:
    train_hashes = set()

if TRAIN_CSV.exists():
    df_train = pd.read_csv(TRAIN_CSV)
    train_paths = set(df_train['path'])
    print(f'Loaded {len(train_paths):,} train paths.')
else:
    train_paths = set()

# 2. Collect all candidate fake frames strictly from test_data_v3
p_test_v3 = DATA / 'test_data_v3'
candidate_files = []

if p_test_v3.exists():
    for d in sorted(p_test_v3.iterdir()):
        if d.is_dir():
            method_name = d.name
            if method_name != 'real':
                imgs = list(d.glob('**/*.png')) + list(d.glob('**/*.jpg'))
                for p in imgs:
                    candidate_files.append({'path': str(p), 'method': method_name, 'label': 1, 'source': 'test_data_v3'})
            else:
                imgs = list(d.glob('**/*.png')) + list(d.glob('**/*.jpg'))
                for p in imgs:
                    candidate_files.append({'path': str(p), 'method': 'real', 'label': 0, 'source': 'test_data_v3_real'})

# 3. Collect holdout Real candidates from FaceForensics++ and FFHQ
ff_dir = DATA / 'FaceForensics++/original_sequences/youtube/c23/frames'
if ff_dir.exists():
    for p in list(ff_dir.glob('*/*.png')):
        if str(p) not in train_paths:
            candidate_files.append({'path': str(p), 'method': 'real', 'label': 0, 'source': 'ff++_real'})

ffhq_dir = DATA / 'kaggle/philosopher0808/real-vs-ai-generated-faces-dataset/versions/1/data_source/data_source/ffhq'
if ffhq_dir.exists():
    for p in list(ffhq_dir.glob('*.jpg'))[:30000]:
        if str(p) not in train_paths:
            candidate_files.append({'path': str(p), 'method': 'real', 'label': 0, 'source': 'ffhq_real'})

print(f'Total candidates collected: {len(candidate_files):,}')

# 4. MD5 Hashing & 3-Tier Leak Filter
def hash_file(path_str):
    try:
        with open(path_str, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

print('Computing MD5 hashes for all candidates...')
with ThreadPoolExecutor(max_workers=16) as pool:
    paths = [c['path'] for c in candidate_files]
    hashes = list(pool.map(hash_file, paths))

clean_candidates = []
leaked_paths = 0
leaked_md5 = 0
seen_hashes = set()

for c, h in zip(candidate_files, hashes):
    if h is None:
        continue
    if c['path'] in train_paths:
        leaked_paths += 1
        continue
    if h in train_hashes:
        leaked_md5 += 1
        continue
    if h in seen_hashes:
        continue  # Deduplicate exact test frames
    seen_hashes.add(h)
    clean_candidates.append(c)

print(f'Filtering Complete:')
print(f'  - Leaked Paths Filtered : {leaked_paths:,}')
print(f'  - Leaked MD5 Filtered   : {leaked_md5:,}')
print(f'  - Clean Frames Remaining: {len(clean_candidates):,}')

df_clean = pd.DataFrame(clean_candidates)

# 5. Create Two Splits:
df_fake_clean = df_clean[df_clean['label'] == 1].copy()
df_real_clean = df_clean[df_clean['label'] == 0].copy()

print(f'\nTotal Clean Fake Frames (Strictly CourseWork Methods): {len(df_fake_clean):,}')
print(f'Total Clean Real Frames: {len(df_real_clean):,}')

# (A) Balanced 1:1 Suite (up to 300 samples per method)
balanced_fake_rows = []
for m in sorted(df_fake_clean['method'].unique()):
    sub = df_fake_clean[df_fake_clean['method'] == m]
    n_sample = min(len(sub), 300)
    balanced_fake_rows.append(sub.sample(n=n_sample, random_state=42))

df_bal_fake = pd.concat(balanced_fake_rows, ignore_index=True)
total_fake_bal = len(df_bal_fake)

# Sample matching real frames
df_bal_real = df_real_clean.sample(n=total_fake_bal, random_state=42)
df_coursework_balanced = pd.concat([df_bal_real, df_bal_fake], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

# (B) Full suite: all clean fake frames + matching real
df_coursework_full_real = df_real_clean.sample(n=len(df_fake_clean), random_state=42)
df_coursework_full = pd.concat([df_coursework_full_real, df_fake_clean], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save CSVs
bal_csv_path = OUT_DIR / 'test_coursework_40methods_balanced_zero_leakage.csv'
full_csv_path = OUT_DIR / 'test_coursework_40methods_full_zero_leakage.csv'

df_coursework_balanced.to_csv(bal_csv_path, index=False)
df_coursework_full.to_csv(full_csv_path, index=False)

print(f'\n✅ SAVED BALANCED 1:1 TEST SET: {bal_csv_path}')
print(f'   - Total Samples: {len(df_coursework_balanced):,} ({len(df_bal_real):,} Real : {len(df_bal_fake):,} Fake across {df_bal_fake["method"].nunique()} Methods)')
print(f'\n✅ SAVED FULL TEST SET: {full_csv_path}')
print(f'   - Total Samples: {len(df_coursework_full):,} ({len(df_coursework_full_real):,} Real : {len(df_fake_clean):,} Fake across {df_fake_clean["method"].nunique()} Methods)')

# Audit breakdown per method
print('\n=== SAMPLES PER METHOD IN BALANCED TEST SUITE ===')
print(df_coursework_balanced['method'].value_counts().to_string())
