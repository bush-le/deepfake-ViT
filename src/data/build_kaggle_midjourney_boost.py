"""
================================================================================
Build Script: Kaggle MidJourney Boost Dataset (12,062 Samples)
Author: Hoang Tuan (Workspace: /workspace/hoangtuan/deepfake-ViT)
Description:
  Extracts and builds the standardized 12k training split from the Kaggle 
  real-vs-ai-generated-faces-dataset (philosopher0808) to boost generalization 
  on MidJourney v5/v6, DiT, SiT, and Latent Diffusion models with 1:1 balance.
================================================================================
"""

import os
import glob
import argparse
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DF40_ROOT", PROJECT_ROOT / "data"))


def build_kaggle_midjourney_boost(
    kaggle_source: str = None,
    output_csv: str = None,
    seed: int = 42,
):
    if kaggle_source is None:
        kaggle_source = str(DATA_ROOT / "kaggle/philosopher0808/real-vs-ai-generated-faces-dataset/versions/1/data_source/data_source")
    if output_csv is None:
        output_csv = str(DATA_ROOT / "zero_leakage_benchmark_fixed/train_kaggle_midjourney_boost_12k.csv")

    kaggle_base = Path(kaggle_source)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📁 Scanning Kaggle data source at: {kaggle_base}")
    if not kaggle_base.exists():
        print(f"⚠️ Warning: Kaggle data source directory not found at {kaggle_base}")
        return pd.DataFrame()

    # 1. Extract 1,031 Stable Diffusion Fake Images (Latent Diffusion)
    sd_paths = sorted(glob.glob(str(kaggle_base / "fake/stable_diffusion/*.jpg")))
    print(f"   • Stable Diffusion Images Found : {len(sd_paths):,}")
    df_sd = pd.DataFrame({
        "path": sd_paths,
        "label": 1,
        "method": "stable_diffusion",
        "domain": "kaggle_diffusion",
    })

    # 2. Extract 5,000 SFHQ Part 1 Fake Images (Studio Synthetic Portraits)
    sfhq_paths = sorted(glob.glob(str(kaggle_base / "fake/sfhq/pt1/*.jpg")))[:5000]
    print(f"   • SFHQ pt1 Images Sampled       : {len(sfhq_paths):,}")
    df_sfhq = pd.DataFrame({
        "path": sfhq_paths,
        "label": 1,
        "method": "sfhq_studio",
        "domain": "kaggle_synthetic",
    })

    # 3. Extract 6,031 FFHQ Real Images (512x512 Pristine Studio Real Faces)
    ffhq_paths = sorted(glob.glob(str(kaggle_base / "ffhq/*.jpg")))[:6031]
    print(f"   • FFHQ Real Images Sampled      : {len(ffhq_paths):,}")
    df_ffhq = pd.DataFrame({
        "path": ffhq_paths,
        "label": 0,
        "method": "ffhq_real",
        "domain": "kaggle_real",
    })

    # 4. Combine and Shuffle
    df_kaggle_12k = pd.concat([df_sd, df_sfhq, df_ffhq], ignore_index=True)
    df_kaggle_12k = df_kaggle_12k.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 5. Save Output CSV
    df_kaggle_12k.to_csv(out_path, index=False)
    print(f"\n✅ Successfully generated Kaggle MidJourney Boost dataset:")
    print(f"   • Total Samples : {len(df_kaggle_12k):,}")
    print(f"   • Real Samples  : {(df_kaggle_12k['label'] == 0).sum():,} (FFHQ)")
    print(f"   • Fake Samples  : {(df_kaggle_12k['label'] == 1).sum():,} (SFHQ + Stable Diffusion)")
    print(f"   • Saved to      : {out_path}")
    return df_kaggle_12k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Kaggle MidJourney Boost Dataset (12k)")
    parser.add_argument(
        "--kaggle-source",
        type=str,
        default=str(DATA_ROOT / "kaggle/philosopher0808/real-vs-ai-generated-faces-dataset/versions/1/data_source/data_source"),
        help="Path to Kaggle data_source folder",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DATA_ROOT / "zero_leakage_benchmark_fixed/train_kaggle_midjourney_boost_12k.csv"),
        help="Output CSV filepath",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    build_kaggle_midjourney_boost(args.kaggle_source, args.output_csv, args.seed)
