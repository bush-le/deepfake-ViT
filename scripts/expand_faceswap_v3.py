"""
Expand faceswap data for the v3 faceswap-focused finetune.

Takes the v2 weakfix train CSV and appends MORE identity-disjoint faceswap frames
from DF40_train_extracted/faceswap (which has 22K+ frames; v2 used only 4K).

Identity-disjoint guarantee is STRENGTHENED vs the original build: for faceswap we
collect EVERY numeric identity token from EVERY test faceswap row regardless of
format (ffc:N, oth:...:idA_idB, cdc:idN), then drop any train frame whose folder
digits intersect those tokens.

Writes data/splits/train_v5_weakfix_v3.csv + a small verification summary.
"""
import os, sys, glob, re, random
import pandas as pd
from pathlib import Path

ROOT = Path("/workspace/hoangtuan/deepfake-ViT")
DATA = Path("/workspace/data")
V2_CSV = ROOT / "data/splits/train_v5_weakfix.csv"
V5_TRAIN = ROOT / "data/splits/train_v5_combined_universal_kaggle_boost.csv"
TEST_CSV = DATA / "zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv"
DFS_IMG = DATA / "deep-fake-face-swap/images/train"
CELEBVHQ_FRAMES = DATA / "celebvhq_frames/real"
IMG_EXT = (".png", ".jpg", ".jpeg")
SEED = 42
FS_ADD_TARGET = 8000  # additional faceswap frames beyond the 4,585 already present


def all_test_identity_tokens(test_df, method):
    """ALL numeric identity tokens of method M in the test set, any format.
    ffc:N, oth:*:idA_idB*, cdc:idN and the FF++ basename <method>__A__B pattern."""
    toks = set()
    sub = test_df[test_df["method"] == method]
    for ident in sub["identity"]:
        for tok in re.findall(r"\d+", ident):
            toks.add(tok)
    for p in sub["path"]:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.match(rf"^{re.escape(method)}__(\d{{2,4}})__(\d{{2,4}})$", stem)
        if m:
            toks |= {m.group(1), m.group(2)}
    return toks


def frame_identity_tokens(path):
    """Numeric identity tokens of a train frame = folder name digits (e.g. '786_819')."""
    folder = os.path.basename(os.path.dirname(path))
    return set(re.findall(r"\d+", folder))


def main():
    rng = random.Random(SEED)
    test_df = pd.read_csv(TEST_CSV, usecols=["method", "identity", "path"])
    test_toks = all_test_identity_tokens(test_df, "faceswap")
    print(f"faceswap test identity tokens ({len(test_toks)}): {sorted(test_toks)[:40]}")

    v2 = pd.read_csv(V2_CSV)
    used = set(v2["path"])
    v5 = set(pd.read_csv(V5_TRAIN, usecols=["path"])["path"])
    print(f"v2 csv: {len(v2):,} rows | faceswap rows: {(v2['method']=='faceswap').sum()}")

    raw_pool = [p for p in glob.glob(str(DATA / "DF40_train_extracted/faceswap/**" / "*.*"), recursive=True)
                if p.lower().endswith(IMG_EXT) and p not in used and p not in v5]
    pool = [p for p in raw_pool if not (frame_identity_tokens(p) & test_toks)]
    dropped = len(raw_pool) - len(pool)
    print(f"candidate faceswap pool: {len(raw_pool)} -> identity-disjoint {len(pool)} (dropped {dropped})")

    rng.shuffle(pool)
    sel = pool[:FS_ADD_TARGET]
    add_rows = pd.DataFrame({"path": sel, "label": 1, "method": "faceswap", "domain": "fake"})
    df = pd.concat([v2, add_rows], ignore_index=True)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    out = ROOT / "data/splits/train_v5_weakfix_v3.csv"
    df.to_csv(out, index=False)

    # verification: identity overlap of newly added frames = must be 0
    overlap = sum(1 for p in sel if frame_identity_tokens(p) & test_toks)
    print(f"\nsaved {out} | total {len(df):,} = {(df.label==0).sum():,}R/{(df.label==1).sum():,}F")
    print(f"faceswap rows now: {(df['method']=='faceswap').sum():,} (added {len(sel):,})")
    print(f"identity overlap in added frames: {overlap} (must be 0)")

    sum_path = ROOT / "experiments/results/v5_weakfix_v3_dataset_summary.json"
    import json
    from datetime import datetime
    (ROOT / "experiments/results").mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "v2_rows": len(v2), "total_rows": len(df),
        "faceswap_before": int((v2["method"] == "faceswap").sum()),
        "faceswap_added": len(sel),
        "faceswap_after": int((df["method"] == "faceswap").sum()),
        "identity_dropped_from_pool": dropped,
        "identity_overlap_added": overlap,
        "test_identity_tokens": sorted(test_toks),
    }, indent=2))
    print(f"summary -> {sum_path}")


if __name__ == "__main__":
    main()
