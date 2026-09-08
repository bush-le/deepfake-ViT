"""
Build finetune dataset for v5 combined model (hoangtuan) targeting WEAK methods.

Composition (all fake = label 1, real = label 0):
  1. REPLAY : full v5 train CSV (train_v5_combined_universal_kaggle_boost.csv, 54K)
              -> keeps existing behavior (no forgetting).
  2. WEAK-METHOD BOOST (DF40_train_extracted, train split -> no test overlap):
     tier1 4000/sample: faceswap, sadtalker, facedancer
     tier2 3000/sample: fsgan, simswap, blendface
     tier2 2500/sample: wav2lip, e4s, inswap
     tier2 2000/sample: lia
     tier3 1500/sample: mobileswap
     tier3 1200/sample: one_shot_free, uniface, pirender
     tier3 1000/sample: ddim, SiT, pixart, styleclip, stargan, tpsm, MRAA, mcnet,
                        danet, hyperreenact, fomm, facevid2vid, VQGAN, StyleGAN2,
                        StyleGAN3, StyleGANXL, sd2.1, RDDM
     (dedup vs v5 train paths so a frame is never double-counted in replay)
  3. NEW: deep-fake-face-swap (HuggingFace) 8,076 fake face-swaps
     method label "deepfake_faceswap".
  4. NEW (test-full, excluded test paths): starganv2 2000, whichfaceisreal 1002,
     CollabDiff 1000, heygen_new 800.
  5. NEW real: CelebV-HQ frames (/workspace/data/celebvhq_frames/real) 4000.

Writes:
  - data/splits/train_v5_weakfix.csv  (path,label,method,domain)
  - experiments/results/v5_weakfix_dataset_summary.json
"""
import os, sys, json, random, glob, re
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.getenv("DF40_ROOT", ROOT / "data"))

V5_TRAIN = ROOT / "data/splits/train_v5_combined_universal_kaggle_boost.csv"
TEST_CSV = DATA / "zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv"
TRAIN_EXTRACTED = DATA / "DF40_train_extracted"
DFS_IMG = DATA / "deep-fake-face-swap/images/train"
TEST_FULL = DATA / "df-40-test-full"
CELEBVHQ_FRAMES = DATA / "celebvhq_frames/real"

IMG_EXT = (".png", ".jpg", ".jpeg")
SEED = 42


def method_test_identity_tokens(test_df, method):
    """FF++ numeric identities of method M in the test set (identity-level disjoint).
    Source: identity col 'ffc:N' + FF++ style basename '<method>__A__B.<ext>'."""
    toks = set()
    sub = test_df[test_df["method"] == method]
    for ident in sub["identity"]:
        toks |= set(re.findall(r"ffc:(\d+)", ident))
    for p in sub["path"]:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.match(rf"^{re.escape(method)}__(\d{{2,4}})__(\d{{2,4}})$", stem)
        if m:
            toks |= {m.group(1), m.group(2)}
    return toks


def frame_identity_tokens(path):
    """Numeric identity tokens of a train frame = folder name (e.g. '786_819')."""
    folder = os.path.basename(os.path.dirname(path))
    return set(re.findall(r"\d+", folder))

TIER1 = {"faceswap": 4000, "sadtalker": 4000, "facedancer": 4000}
TIER2 = {"fsgan": 3000, "simswap": 3000, "blendface": 3000,
         "wav2lip": 2500, "e4s": 2500, "inswap": 2500, "lia": 2000}
TIER3 = {"mobileswap": 1500, "one_shot_free": 1200, "uniface": 1200, "pirender": 1200,
         "ddim": 1000, "SiT": 1000, "pixart": 1000, "styleclip": 1000, "stargan": 1000,
         "tpsm": 1000, "MRAA": 1000, "mcnet": 1000, "danet": 1000, "hyperreenact": 1000,
         "fomm": 1000, "facevid2vid": 1000, "VQGAN": 1000, "StyleGAN2": 1000,
         "StyleGAN3": 1000, "StyleGANXL": 1000, "sd2.1": 1000, "RDDM": 1000}

TEST_FULL_FAKE = {"starganv2": 2000, "whichfaceisreal": 1002, "CollabDiff": 1000, "heygen_new": 800}
CELEBVHQ_REAL_TARGET = 4000


def main():
    rng = random.Random(SEED)

    # 1. Replay v5 train
    v5 = pd.read_csv(V5_TRAIN)
    v5 = v5[["path", "label", "method", "domain"]]
    v5_paths = set(v5["path"])
    print(f"[replay] v5 train: {len(v5):,} ({int((v5.label==0).sum()):,} real / {int((v5.label==1).sum()):,} fake)")

    # test paths to exclude from test-full additions
    test_paths = set(pd.read_csv(TEST_CSV, usecols=["path"])["path"])
    print(f"[test] {len(test_paths):,} test paths excluded from df-40-test-full additions")

    rows = []

    # 2. weak-method boost from DF40_train_extracted (dedup vs v5)
    boost = {**TIER1, **TIER2, **TIER3}
    test_tokens = {m: method_test_identity_tokens(pd.read_csv(TEST_CSV, usecols=["method", "identity", "path"]), m)
                   for m in list(boost) + list(TEST_FULL_FAKE)}
    leaked_dropped = {}
    added = {}
    for m, n in boost.items():
        raw_pool = [p for p in glob.glob(str(TRAIN_EXTRACTED / m / "**" / "*.*"), recursive=True)
                    if p.lower().endswith(IMG_EXT) and p not in v5_paths]
        pool = [p for p in raw_pool if not (frame_identity_tokens(p) & test_tokens[m])]
        leaked_dropped[f"train_extracted/{m}"] = len(raw_pool) - len(pool)
        rng.shuffle(pool)
        sel = pool[:n]
        for p in sel:
            rows.append((p, 1, m, "fake"))
        added[f"train_extracted/{m}"] = len(sel)
    print(f"[weak-method boost] added {sum(added.values()):,} fake frames from DF40_train_extracted (identity-disjoint)")

    # 3. deep-fake-face-swap
    dfs = [p for p in sorted(glob.glob(str(DFS_IMG / "*.*"))) if p.lower().endswith(IMG_EXT)]
    for p in dfs:
        rows.append((p, 1, "deepfake_faceswap", "fake"))
    print(f"[deep-fake-face-swap] added {len(dfs):,} fake face-swaps")

    # 4. df-40-test-full weak methods (exclude test paths + identity overlap)
    for m, n in TEST_FULL_FAKE.items():
        raw_pool = [p for p in glob.glob(str(TEST_FULL / m / "fake" / "**" / "*.*"), recursive=True)
                    if p.lower().endswith(IMG_EXT) and p not in test_paths and p not in v5_paths]
        pool = [p for p in raw_pool if not (frame_identity_tokens(p) & test_tokens[m])]
        leaked_dropped[f"test_full/{m}"] = len(raw_pool) - len(pool)
        rng.shuffle(pool)
        sel = pool[:n]
        for p in sel:
            rows.append((p, 1, m, "fake"))
        added[f"test_full/{m}"] = len(sel)
    print(f"[df-40-test-full] added {sum(len([r for r in rows if r[3]=='fake' and r[2] in TEST_FULL_FAKE]) for _ in [0]):,} fake frames (see summary)")

    # 5. CelebV-HQ real frames
    cb = [p for p in sorted(glob.glob(str(CELEBVHQ_FRAMES / "*.*"))) if p.lower().endswith(IMG_EXT)]
    rng.shuffle(cb)
    cb_sel = cb[:CELEBVHQ_REAL_TARGET]
    for p in cb_sel:
        rows.append((p, 0, "celebvhq_real", "real"))
    print(f"[celebvhq] added {len(cb_sel):,} real frames")

    df = pd.DataFrame(rows, columns=["path", "label", "method", "domain"])
    df = pd.concat([v5, df], ignore_index=True)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    out_csv = ROOT / "data/splits/train_v5_weakfix.csv"
    df.to_csv(out_csv, index=False)

    # summary
    real_n = int((df.label == 0).sum())
    fake_n = int((df.label == 1).sum())
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(df), "real": real_n, "fake": fake_n,
        "added_by_source": added,
        "identity_dropped": leaked_dropped,
        "test_identity_tokens": {m: sorted(v) for m, v in test_tokens.items() if v},
        "replay_v5": len(v5),
        "method_counts": df.groupby("method")["label"].agg(["count", "sum"]).to_dict("index"),
        "out_csv": str(out_csv),
    }
    sum_path = ROOT / "experiments/results/v5_weakfix_dataset_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[dataset] total {len(df):,} = {real_n:,} real / {fake_n:,} fake")
    print(f"[dataset] wrote {out_csv}")
    print(f"[dataset] summary -> {sum_path}")


if __name__ == "__main__":
    main()
