"""Tests unitaires pour le module evaluation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.evaluation import (
    recall_at_fpr, precision_at_recall, evaluate_model,
    find_optimal_threshold, compare_models
)


@pytest.fixture
def synthetic_predictions():
    """Génère des prédictions synthétiques pour les tests."""
    np.random.seed(42)
    n = 10000
    y_true = np.random.choice([0, 1], n, p=[0.99, 0.01])
    # Score corrélé avec y_true
    y_score = np.where(y_true == 1,
                        np.random.beta(8, 2, n),  # fraudes : scores élevés
                        np.random.beta(2, 8, n))  # légitimes : scores bas
    return y_true, y_score


def test_recall_at_fpr(synthetic_predictions):
    """recall_at_fpr doit retourner une valeur entre 0 et 1."""
    y_true, y_score = synthetic_predictions
    recall = recall_at_fpr(y_true, y_score, target_fpr=0.05)
    assert 0 <= recall <= 1
    # Sur des données synthétiques bien séparées, recall doit être > 0.5
    assert recall > 0.5


def test_precision_at_recall(synthetic_predictions):
    """precision_at_recall doit retourner une valeur entre 0 et 1."""
    y_true, y_score = synthetic_predictions
    precision = precision_at_recall(y_true, y_score, target_recall=0.75)
    assert 0 <= precision <= 1


def test_evaluate_model_keys(synthetic_predictions):
    """evaluate_model retourne toutes les métriques attendues."""
    y_true, y_score = synthetic_predictions
    metrics = evaluate_model(y_true, y_score, model_name='Test', verbose=False)

    expected_keys = [
        'model', 'pr_auc', 'roc_auc', 'recall_at_5pct_fpr',
        'recall_at_10pct_fpr', 'f1_fraud', 'precision_fraud',
        'recall_fraud', 'threshold',
        'true_negative', 'false_positive', 'false_negative', 'true_positive',
    ]
    for key in expected_keys:
        assert key in metrics, f"Clé manquante : {key}"


def test_evaluate_model_values_range(synthetic_predictions):
    """Toutes les métriques doivent être dans [0, 1]."""
    y_true, y_score = synthetic_predictions
    metrics = evaluate_model(y_true, y_score, model_name='Test', verbose=False)
    for key in ['pr_auc', 'roc_auc', 'recall_at_5pct_fpr',
                'f1_fraud', 'precision_fraud', 'recall_fraud']:
        assert 0 <= metrics[key] <= 1, f"{key} hors [0,1] : {metrics[key]}"


def test_find_optimal_threshold(synthetic_predictions):
    """find_optimal_threshold retourne un seuil dans [0, 1]."""
    y_true, y_score = synthetic_predictions

    thresh_f1 = find_optimal_threshold(y_true, y_score, criterion='f1')
    assert 0 <= thresh_f1 <= 1

    thresh_fpr = find_optimal_threshold(y_true, y_score, criterion='fpr_constraint',
                                          target_fpr=0.05)
    assert 0 <= thresh_fpr <= 1


def test_compare_models(synthetic_predictions):
    """compare_models retourne un DataFrame trié par PR-AUC."""
    y_true, y_score = synthetic_predictions
    m1 = evaluate_model(y_true, y_score, model_name='M1', verbose=False)
    m2 = evaluate_model(y_true, np.random.rand(len(y_true)), model_name='M2', verbose=False)

    df = compare_models([m1, m2])

    assert len(df) == 2
    assert 'pr_auc' in df.columns
    # Vérifier le tri décroissant
    assert df['pr_auc'].iloc[0] >= df['pr_auc'].iloc[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
