import pytest
import numpy as np
from src.eval.pauc_metrics import (
    compute_partial_auc,
    compute_tpr_at_fixed_fpr,
    find_optimal_threshold,
    evaluate_forensic_metrics
)

def test_perfect_classifier():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.8, 0.85, 0.9, 0.95, 0.99])
    
    raw_pauc, norm_pauc = compute_partial_auc(y_true, y_prob, max_fpr=0.05)
    assert 0.0 <= raw_pauc <= 0.05
    assert norm_pauc == pytest.approx(1.0, rel=1e-3)
    
    tpr_1pct = compute_tpr_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)
    assert tpr_1pct == pytest.approx(1.0, rel=1e-3)

def test_random_classifier():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=1000)
    y_prob = np.random.uniform(0, 1, size=1000)
    
    raw_pauc, norm_pauc = compute_partial_auc(y_true, y_prob, max_fpr=0.05)
    # For random classifier, expected standardized pAUC is roughly around 0.5 (or max_fpr/2 normalized: 0.025)
    assert 0.0 <= raw_pauc <= 0.05
    assert 0.0 <= norm_pauc <= 1.0

def test_evaluate_forensic_metrics():
    y_true = np.array([0] * 50 + [1] * 50)
    # Good predictions with a couple errors
    y_prob = np.concatenate([np.random.uniform(0.01, 0.3, 48), [0.8, 0.7],  # 2 FP
                             np.random.uniform(0.7, 0.99, 47), [0.2, 0.15, 0.3]]) # 3 FN
    
    m = evaluate_forensic_metrics(y_true, y_prob, threshold=0.5)
    assert "tpa_recall" in m
    assert "spa_specificity" in m
    assert "pauc_005_standardized" in m
    assert "tpr_at_fpr_1pct" in m
    assert "g_mean" in m
    assert 0.0 <= m["tpa_recall"] <= 1.0
    assert 0.0 <= m["spa_specificity"] <= 1.0
    assert 0.0 <= m["pauc_005_standardized"] <= 1.0
