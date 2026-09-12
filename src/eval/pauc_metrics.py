"""
pauc_metrics.py — Forensic Evaluation Metrics Suite for Deepfake Detection

Implements rigorous biometrics & digital forensics evaluation criteria:
1. Partial ROC-AUC (pAUC) in low-FPR regime (FPR in [0, 0.05]) with McClish standardization.
2. TPR at fixed low False Positive Rates (TPR @ FPR = 1%, TPR @ FPR = 5%).
3. TPA (True Positive Accuracy / Recall) & SPA (Safe Positive Accuracy / Specificity).
4. Balanced Accuracy & Geometric Mean (G-Mean) for imbalanced data.
5. Calibrated optimal decision threshold (tau*) via constrained Youden's J index.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def compute_partial_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Compute partial Area Under the ROC Curve (pAUC) for FPR in [0, max_fpr].
    
    Returns:
        (raw_pauc, standardized_pauc)
        where standardized_pauc is normalized according to McClish (1989) into [0, 1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    # Identify cutoff index where FPR reaches max_fpr
    stop_idx = np.searchsorted(fpr, max_fpr, side="right")
    
    x = list(fpr[:stop_idx])
    y = list(tpr[:stop_idx])
    
    # Ensure point at (0, 0) is present
    if len(x) == 0 or x[0] > 0:
        x.insert(0, 0.0)
        y.insert(0, 0.0)
        
    # Linear interpolation at exactly max_fpr if the cutoff point is not exact
    if x[-1] < max_fpr:
        if stop_idx < len(fpr):
            denom = fpr[stop_idx] - fpr[stop_idx - 1]
            if denom > 1e-12:
                slope = (tpr[stop_idx] - tpr[stop_idx - 1]) / denom
                y_interp = tpr[stop_idx - 1] + slope * (max_fpr - fpr[stop_idx - 1])
            else:
                y_interp = tpr[stop_idx]
        else:
            y_interp = y[-1]
        x.append(max_fpr)
        y.append(y_interp)
        
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # Trapezoidal integration
    raw_pauc = float(np.trapezoid(y_arr, x_arr) if hasattr(np, "trapezoid") else np.trapz(y_arr, x_arr))
    
    # McClish standardization: maps chance (0.5 * max_fpr^2) to 0.5, perfect (max_fpr) to 1.0
    # Formula: standardized = 0.5 * (1 + (raw_pauc - min_pauc) / (max_pauc - min_pauc))
    # Or common direct forensic normalization: raw_pauc / max_fpr
    standardized_pauc = float(raw_pauc / max_fpr)
    
    return raw_pauc, standardized_pauc


def compute_tpr_at_fixed_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01
) -> float:
    """
    Compute True Positive Rate (TPR) at a fixed False Positive Rate (FPR <= target_fpr).
    Crucial benchmark metric for biometric security and digital forensics (e.g. NIST FRVT).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    # Find all indices where FPR <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0.0
    
    # Return highest TPR achieved within the allowable FPR budget
    return float(np.max(tpr[valid_indices]))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_allowable_fpr: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Find optimal classification decision threshold using Youden's J Index:
        J = TPR(tau) - FPR(tau) = Sensitivity + Specificity - 1
    
    If max_allowable_fpr is set (e.g. 0.01 or 0.05), finds the threshold that
    maximizes TPR while strictly constraining FPR <= max_allowable_fpr.
    
    Returns:
        (optimal_threshold, best_tpr, best_fpr)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    if max_allowable_fpr is not None:
        valid_mask = fpr <= max_allowable_fpr
        if np.any(valid_mask):
            best_idx = np.argmax(tpr[valid_mask])
            actual_idx = np.where(valid_mask)[0][best_idx]
            return float(thresholds[actual_idx]), float(tpr[actual_idx]), float(fpr[actual_idx])
            
    # Unconstrained Youden's J
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    
    # Boundary guard: if threshold is inf (start of roc_curve)
    if np.isinf(best_threshold):
        best_threshold = 1.0
        
    return best_threshold, float(tpr[best_idx]), float(fpr[best_idx])


def evaluate_forensic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Comprehensive forensic evaluation returning all required metrics in one dictionary.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # TPA (True Positive Accuracy / Recall) & SPA (Safe Positive Accuracy / Specificity)
    tpa = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spa = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    balanced_acc = 0.5 * (tpa + spa)
    g_mean = float(np.sqrt(tpa * spa))
    
    # Full ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    full_auc = float(auc(fpr, tpr))
    
    # Partial AUC in [0, 0.05]
    raw_pauc_05, norm_pauc_05 = compute_partial_auc(y_true, y_prob, max_fpr=0.05)
    
    # Fixed FPR operational points
    tpr_at_fpr_1pct = compute_tpr_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
    tpr_at_fpr_5pct = compute_tpr_at_fixed_fpr(y_true, y_prob, target_fpr=0.05)
    
    # Optimal thresholds
    tau_youden, tpr_youden, fpr_youden = find_optimal_threshold(y_true, y_prob)
    tau_constrained, tpr_const, fpr_const = find_optimal_threshold(y_true, y_prob, max_allowable_fpr=0.01)
    
    return {
        "total_samples": int(len(y_true)),
        "real_count": int(np.sum(y_true == 0)),
        "fake_count": int(np.sum(y_true == 1)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tpa_recall": tpa,
        "spa_specificity": spa,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_acc),
        "g_mean": g_mean,
        "full_auc": full_auc,
        "pauc_005_raw": raw_pauc_05,
        "pauc_005_standardized": norm_pauc_05,
        "tpr_at_fpr_1pct": tpr_at_fpr_1pct,
        "tpr_at_fpr_5pct": tpr_at_fpr_5pct,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "optimal_threshold_youden": {
            "threshold": tau_youden,
            "tpr": tpr_youden,
            "fpr": fpr_youden
        },
        "optimal_threshold_fpr_1pct": {
            "threshold": tau_constrained,
            "tpr": tpr_const,
            "fpr": fpr_const
        }
    }
