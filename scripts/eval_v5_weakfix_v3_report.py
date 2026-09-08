"""
Regenerate v3 (faceswap-focused) report files from the saved checkpoint without retraining.
Loads experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt, evaluates the zero-leakage
test set, writes report JSON + per-method CSV + vs-baseline CSV + vs-v2 CSV.
"""
import os, sys, json
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.getenv("DF40_ROOT", ROOT / "data"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_v5_weakfix import (DeepfakeDataset, get_eval_transforms, evaluate)
from torch.utils.data import DataLoader
from src.models.dinov3_vit import build_dinov3_classifier

CKPT = ROOT / "experiments/checkpoints/best_model_v3.pt" if (ROOT / "experiments/checkpoints/best_model_v3.pt").exists() else ROOT / "experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt"
TEST_CSV = os.getenv("TEST_CSV", str(DATA / "zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv"))
BASELINE_CSV = ROOT / "experiments/results/v5_combined_per_method_accuracy.csv"
V2_CSV = ROOT / "experiments/results/v5_weakfix_per_method_accuracy.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_dinov3_classifier(weights_path=str(ROOT / "models/dinov3_small/model.safetensors"),
                                    img_size=256, device=str(device))
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    print(f"loaded {CKPT}: missing={len(missing)} unexpected={len(unexpected)}")

    df_test = pd.read_csv(TEST_CSV)
    test_ds = DeepfakeDataset(df_test, get_eval_transforms(256))
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    res = evaluate(model, test_loader, device)
    print(f"TEST: acc={res['acc']*100:.2f}% auc={res['auc']*100:.2f}% "
          f"prec={res['prec']*100:.2f}% rec={res['rec']*100:.2f}% cm={res['cm']}")

    df_test["pred"] = res["preds"]
    df_test["prob"] = res["probs"]
    method_perf = []
    for m in df_test["method"].unique():
        sub = df_test[df_test["method"] == m]
        method_perf.append({"Method": m,
                            "Label": "REAL" if sub["label"].iloc[0] == 0 else "FAKE",
                            "Samples": len(sub),
                            "Accuracy": round(float((sub["pred"] == sub["label"]).mean()) * 100, 2),
                            "Mean Fake Probability": round(float(sub["prob"].mean()), 4)})
    df_method = pd.DataFrame(method_perf).sort_values(["Label", "Accuracy"], ascending=[True, False])

    baseline = pd.read_csv(BASELINE_CSV)
    merged = df_method.merge(baseline[["Method", "Accuracy"]], on="Method",
                             suffixes=("_weakfix_v3", "_baseline"))
    merged["delta"] = merged["Accuracy_weakfix_v3"] - merged["Accuracy_baseline"]
    merged = merged.sort_values("Accuracy_baseline")

    v2 = pd.read_csv(V2_CSV)[["Method", "Accuracy"]]
    merged_v2 = df_method.merge(v2, on="Method", suffixes=("_v3", "_v2"))
    merged_v2["delta_v3_vs_v2"] = merged_v2["Accuracy_v3"] - merged_v2["Accuracy_v2"]
    merged_v2 = merged_v2.sort_values("Accuracy_v3")

    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(CKPT),
        "test_metrics": {"acc": res["acc"], "auc": res["auc"], "precision": res["prec"],
                         "recall": res["rec"], "f1": res["f1"], "cm": res["cm"]},
        "method_breakdown": df_method.to_dict("records"),
    }
    (ROOT / "experiments/results/v5_weakfix_v3_training_report.json").write_text(
        json.dumps(report, indent=2))
    df_method.to_csv(ROOT / "experiments/results/v5_weakfix_v3_per_method_accuracy.csv", index=False)
    merged.to_csv(ROOT / "experiments/results/v5_weakfix_v3_vs_baseline.csv", index=False)
    merged_v2.to_csv(ROOT / "experiments/results/v5_weakfix_v3_vs_v2.csv", index=False)
    print("Saved: v5_weakfix_v3_training_report.json / per_method_accuracy.csv / vs_baseline.csv / vs_v2.csv")

    print("\n=== v3 vs v2 (regression check, sorted by v3 acc worst-first) ===")
    print(merged_v2[["Method", "Label_v3", "Samples", "Accuracy_v3", "Accuracy_v2",
                     "delta_v3_vs_v2", "Mean Fake Probability_v3"]].to_string(index=False))
    reg = merged_v2[merged_v2["delta_v3_vs_v2"] < -3]
    print("\nregressions >3pts:", "NONE" if len(reg) == 0 else reg[["Method", "Accuracy_v3", "Accuracy_v2",
                                                                   "delta_v3_vs_v2", "Samples"]].to_string(index=False))


if __name__ == "__main__":
    main()
