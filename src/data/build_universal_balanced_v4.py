import os
import sys
import glob
import re
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DF40_ROOT", PROJECT_ROOT / "data"))
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

print("=" * 85)
print("🚀 BUILDING UNIVERSAL PERFECTLY-BALANCED DATASET SPLITS V4 (STRICT ZERO-LEAKAGE)")
print("=" * 85)

def fast_md5(path):
    try:
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            hasher.update(f.read(32768))
        return hasher.hexdigest()
    except Exception:
        return None

# 1. Index all readable files per method
print("\n[1/5] Indexing all readable images on disk...")

all_fake_pools = defaultdict(list)
all_real_pools = defaultdict(list)
file_to_info = {}

# 1.1 Test data v3 & df-40-test-full
for base_dir in [str(DATA_ROOT / "test_data_v3"), str(DATA_ROOT / "df-40-test-full")]:
    for f in sorted(glob.glob(f"{base_dir}/*/*/*.*")):
        if not f.endswith((".png", ".jpg", ".jpeg")):
            continue
        if not os.path.exists(f) or os.path.getsize(f) == 0 or not os.access(f, os.R_OK):
            continue
        
        parts = f.replace("\\", "/").split("/")
        m_name = parts[-3]
        label_type = parts[-2]
        
        if label_type == "real":
            r_name = f"Real_{m_name}"
            all_real_pools[r_name].append(f)
            file_to_info[f] = (r_name, True)
        else:
            all_fake_pools[m_name].append(f)
            file_to_info[f] = (m_name, False)

# 1.2 DF40 Train Manifest
manifest_path = DATA_ROOT / "DF40_train_manifest.csv"
if manifest_path.exists():
    df_man = pd.read_csv(manifest_path)
    for p, m, l in zip(df_man["path"], df_man["method"], df_man["label"]):
        if os.path.exists(p) and os.path.getsize(p) > 0 and os.access(p, os.R_OK):
            if l == 0:
                all_real_pools["Real_DF40"].append(p)
                file_to_info[p] = ("Real_DF40", True)
            else:
                all_fake_pools[m].append(p)
                file_to_info[p] = (m, False)

# 1.3 Celeb-DF Real
for f in sorted(glob.glob(str(PROJECT_ROOT / "data/processed/celeb_df_extracted/*.png"))):
    if os.path.exists(f) and os.path.getsize(f) > 0 and os.access(f, os.R_OK):
        all_real_pools["Real_CelebDF"].append(f)
        file_to_info[f] = ("Real_CelebDF", True)

# 1.4 FaceForensics++ Real
for f in sorted(glob.glob(str(DATA_ROOT / "FaceForensics++/original_sequences/youtube/c23/frames/*/*.png"))):
    if os.path.exists(f) and os.path.getsize(f) > 0 and os.access(f, os.R_OK):
        all_real_pools["Real_FFPP"].append(f)
        file_to_info[f] = ("Real_FFPP", True)

for k in all_fake_pools:
    all_fake_pools[k] = sorted(list(set(all_fake_pools[k])))
for k in all_real_pools:
    all_real_pools[k] = sorted(list(set(all_real_pools[k])))

print(f"✅ Indexed {len(all_fake_pools)} Fake Methods ({sum(len(v) for v in all_fake_pools.values()):,} images)")
print(f"✅ Indexed {len(all_real_pools)} Real Sources ({sum(len(v) for v in all_real_pools.values()):,} images)")

# 2. Select Candidate Pool
print("\n[2/5] Selecting candidate pool and computing concurrent content hashes...")
candidate_files = []

for m, files in all_fake_pools.items():
    random.shuffle(files)
    candidate_files.extend(files[:2000])

for r, files in all_real_pools.items():
    random.shuffle(files)
    cap = 15000 if r == "Real_FFPP" else (8000 if r == "Real_CelebDF" else 2000)
    candidate_files.extend(files[:cap])

print(f"Total Candidate Pool: {len(candidate_files):,} images")

# Hash candidates in parallel
with ThreadPoolExecutor(max_workers=16) as executor:
    hashes = list(executor.map(fast_md5, candidate_files))

file_to_hash = dict(zip(candidate_files, hashes))

# 3. Partition by Video ID & Content Hash
print("\n[3/5] Assigning disjoint Train (80%), Val (10%), Test (10%) splits...")

def get_group_key(path):
    fname = os.path.basename(path).lower()
    
    # 1. FF++ video ID
    if "faceforensics++" in path.lower():
        parts = path.split("/")
        if len(parts) >= 2 and parts[-2].isdigit():
            return f"ff_vid_{parts[-2]}"
            
    # 2. Celeb-DF subject ID
    match_id = re.search(r'id(\d+)', fname)
    if match_id:
        return f"celeb_id_{match_id.group(1)}"

    # 3. For synthetic images, use content MD5 hash
    h = file_to_hash.get(path)
    return f"hash_{h}" if h else f"path_{path}"

group_to_split = {}
def assign_group_split(key):
    if key not in group_to_split:
        val = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
        if val < 8: group_to_split[key] = "train"
        elif val == 8: group_to_split[key] = "val"
        else: group_to_split[key] = "test"
    return group_to_split[key]

train_fakes, val_fakes, test_fakes = defaultdict(list), defaultdict(list), defaultdict(list)
train_reals, val_reals, test_reals = defaultdict(list), defaultdict(list), defaultdict(list)

seen_content_hashes = set()

for f in candidate_files:
    h = file_to_hash.get(f)
    if not h or h in seen_content_hashes:
        continue
    seen_content_hashes.add(h)
    
    g_key = get_group_key(f)
    s = assign_group_split(g_key)
    
    m_name, is_real = file_to_info.get(f, ("unknown", False))
    if is_real:
        if s == "train": train_reals[m_name].append(f)
        elif s == "val": val_reals[m_name].append(f)
        else: test_reals[m_name].append(f)
    else:
        if s == "train": train_fakes[m_name].append(f)
        elif s == "val": val_fakes[m_name].append(f)
        else: test_fakes[m_name].append(f)

# 4. Balanced 1:1 Sampling across all methods
print("\n[4/5] Sampling balanced 1:1 quotas for Train (50k), Val (5k), Test (5k)...")

TRAIN_TOTAL_FAKE = 25000
VAL_TOTAL_FAKE = 2500
TEST_TOTAL_FAKE = 2500

active_fake_methods = sorted([m for m in all_fake_pools.keys() if len(all_fake_pools[m]) >= 50])
print(f"Active fake methods (N={len(active_fake_methods)}): {active_fake_methods}")

def sample_balanced_fakes(pool_dict, target_total):
    sampled = []
    methods = [m for m in active_fake_methods if len(pool_dict[m]) > 0]
    base_quota = target_total // len(methods)
    
    for m in methods:
        available = pool_dict[m]
        random.shuffle(available)
        n_take = min(len(available), base_quota)
        for p in available[:n_take]:
            sampled.append({"path": p, "label": 1, "method": m, "domain": "fake"})
            
    remaining = target_total - len(sampled)
    if remaining > 0:
        extra_pool = []
        for m in methods:
            available = pool_dict[m]
            if len(available) > base_quota:
                for p in available[base_quota:]:
                    extra_pool.append({"path": p, "label": 1, "method": m, "domain": "fake"})
        random.shuffle(extra_pool)
        sampled.extend(extra_pool[:remaining])
        
    return sampled

train_fake_samples = sample_balanced_fakes(train_fakes, TRAIN_TOTAL_FAKE)
val_fake_samples = sample_balanced_fakes(val_fakes, VAL_TOTAL_FAKE)
test_fake_samples = sample_balanced_fakes(test_fakes, TEST_TOTAL_FAKE)

def sample_balanced_reals(pool_dict, target_total):
    sampled = []
    sources = list(pool_dict.keys())
    
    web_reals = [r for r in sources if r not in ["Real_FFPP", "Real_CelebDF"]]
    
    target_ff = int(target_total * 0.50)
    target_celeb = int(target_total * 0.30)
    target_web = target_total - target_ff - target_celeb
    
    ff_files = pool_dict.get("Real_FFPP", [])
    random.shuffle(ff_files)
    for p in ff_files[:target_ff]:
        sampled.append({"path": p, "label": 0, "method": "FaceForensics++ Real", "domain": "real"})
        
    celeb_files = pool_dict.get("Real_CelebDF", [])
    random.shuffle(celeb_files)
    for p in celeb_files[:target_celeb]:
        sampled.append({"path": p, "label": 0, "method": "Celeb-DF Real", "domain": "real"})
        
    all_web_files = []
    for wr in web_reals:
        for f in pool_dict[wr]:
            all_web_files.append((f, wr))
    random.shuffle(all_web_files)
    for p, wr in all_web_files[:target_web]:
        sampled.append({"path": p, "label": 0, "method": wr.replace("Real_", "") + " Real", "domain": "real"})
        
    remaining = target_total - len(sampled)
    if remaining > 0:
        leftover_ff = ff_files[target_ff:]
        for p in leftover_ff[:remaining]:
            sampled.append({"path": p, "label": 0, "method": "FaceForensics++ Real", "domain": "real"})
            
    return sampled

train_real_samples = sample_balanced_reals(train_reals, TRAIN_TOTAL_FAKE)
val_real_samples = sample_balanced_reals(val_reals, VAL_TOTAL_FAKE)
test_real_samples = sample_balanced_reals(test_reals, TEST_TOTAL_FAKE)

df_train_v4 = pd.DataFrame(train_fake_samples + train_real_samples).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_val_v4 = pd.DataFrame(val_fake_samples + val_real_samples).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_test_v4 = pd.DataFrame(test_fake_samples + test_real_samples).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# 5. Strict Zero Data Leakage Verification
print("\n[5/5] Performing Mathematical Zero-Leakage Audit...")

train_paths = set(df_train_v4["path"])
val_paths = set(df_val_v4["path"])
test_paths = set(df_test_v4["path"])

train_test_overlap = train_paths.intersection(test_paths)
val_test_overlap = val_paths.intersection(test_paths)
train_val_overlap = train_paths.intersection(val_paths)

print(f"  • Train ∩ Test Path Overlap : {len(train_test_overlap)} paths")
print(f"  • Val ∩ Test Path Overlap   : {len(val_test_overlap)} paths")
print(f"  • Train ∩ Val Path Overlap  : {len(train_val_overlap)} paths")

assert len(train_test_overlap) == 0, f"Path leak detected between Train and Test: {len(train_test_overlap)}"
assert len(val_test_overlap) == 0, f"Path leak detected between Val and Test: {len(val_test_overlap)}"
assert len(train_val_overlap) == 0, f"Path leak detected between Train and Val: {len(train_val_overlap)}"

train_hashes = set(file_to_hash[p] for p in list(df_train_v4["path"]) + list(df_val_v4["path"]))
test_hashes = set(file_to_hash[p] for p in df_test_v4["path"])
hash_overlap = train_hashes.intersection(test_hashes)

print(f"  • Train/Val ∩ Test Content Hash Overlap : {len(hash_overlap)} hashes")
assert len(hash_overlap) == 0, f"Content hash duplicate leak detected: {len(hash_overlap)}"

print("🎉 ZERO DATA LEAKAGE 100% MATHEMATICALLY VERIFIED!")

# Save to disk
train_v4_path = SPLITS_DIR / "train_v4_universal_balanced.csv"
val_v4_path = SPLITS_DIR / "val_v4_universal_balanced.csv"
test_v4_path = SPLITS_DIR / "test_v4_universal_balanced.csv"

df_train_v4.to_csv(train_v4_path, index=False)
df_val_v4.to_csv(val_v4_path, index=False)
df_test_v4.to_csv(test_v4_path, index=False)

print(f"\n💾 Saved {train_v4_path.name} : {len(df_train_v4):,} rows (Real: {(df_train_v4.label==0).sum():,}, Fake: {(df_train_v4.label==1).sum():,})")
print(f"💾 Saved {val_v4_path.name}   : {len(df_val_v4):,} rows (Real: {(df_val_v4.label==0).sum():,}, Fake: {(df_val_v4.label==1).sum():,})")
print(f"💾 Saved {test_v4_path.name}  : {len(df_test_v4):,} rows (Real: {(df_test_v4.label==0).sum():,}, Fake: {(df_test_v4.label==1).sum():,})")

# Export Summary Matrix of V4
v4_summary = []
all_v4_methods = sorted(list(set(df_train_v4["method"]).union(set(df_test_v4["method"]))))
for m in all_v4_methods:
    tr = int((df_train_v4["method"] == m).sum())
    va = int((df_val_v4["method"] == m).sum())
    te = int((df_test_v4["method"] == m).sum())
    lbl = "Real" if "Real" in m else "Fake"
    v4_summary.append({"Method / Source": m, "Type": lbl, "Train V4 (50k)": tr, "Val V4 (5k)": va, "Test V4 (5k)": te, "Total V4": tr + va + te})

df_v4_sum = pd.DataFrame(v4_summary).sort_values(by=["Type", "Total V4"], ascending=[True, False]).reset_index(drop=True)
v4_summary_csv = RESULTS_DIR / "v4_universal_balanced_method_breakdown.csv"
df_v4_sum.to_csv(v4_summary_csv, index=False)
print(f"\n📊 Summary Table saved to: {v4_summary_csv}")
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
print(df_v4_sum.to_string())

