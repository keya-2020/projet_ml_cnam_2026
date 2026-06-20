"""
Métriques d'évaluation adaptées à la détection de fraude.

Sur ce dataset, l'accuracy est INUTILISABLE : un modèle prédisant toujours
"légitime" atteint 99% d'accuracy sans détecter aucune fraude.

Métriques pertinentes :
- PR-AUC (Precision-Recall AUC) — métrique de référence
- Recall @ X% FPR — métrique opérationnelle bancaire
- AUC-ROC — comparaison entre modèles
- F1-score sur la classe positive uniquement
"""

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05
) -> float:
    """
    Calcule le recall (taux de détection de fraudes) à un FPR (taux de fausses
    alertes) donné.

    En contexte bancaire opérationnel, le département anti-fraude ne peut traiter
    qu'un nombre limité d'alertes par jour. On fixe donc un budget de fausses
    alertes (ex. 5%) et on mesure combien de vraies fraudes le modèle capture
    dans ce budget.

    Parameters
    ----------
    y_true : array-like
        Vraies étiquettes (0/1).
    y_score : array-like
        Probabilités prédites pour la classe positive.
    target_fpr : float, default=0.05
        FPR maximum acceptable (5% par défaut).

    Returns
    -------
    float
        Recall correspondant au seuil qui produit ce FPR.

    Examples
    --------
    >>> recall_at_fpr(y_true, y_proba, target_fpr=0.05)
    0.634  # Le modèle détecte 63.4% des fraudes en générant 5% de fausses alertes
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Trouver le plus grand TPR (= recall) avec FPR <= target_fpr
    valid_idx = fpr <= target_fpr
    if not valid_idx.any():
        return 0.0
    return float(tpr[valid_idx].max())


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.75
) -> float:
    """
    Précision atteignable pour un recall cible donné.

    Symétrique de recall_at_fpr : si le département anti-fraude exige
    de détecter au moins 75% des fraudes, quel est le ratio de bonnes
    alertes parmi toutes les alertes ?
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid_idx = recall >= target_recall
    if not valid_idx.any():
        return 0.0
    return float(precision[valid_idx].max())


def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray = None,
    threshold: float = 0.5,
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Évaluation complète d'un modèle sur les métriques pertinentes pour la fraude.

    Parameters
    ----------
    y_true : array
        Vraies étiquettes.
    y_score : array
        Probabilités prédites (continues entre 0 et 1).
    y_pred : array, optional
        Prédictions binaires. Si None, calculé via `y_score > threshold`.
    threshold : float, default=0.5
        Seuil de décision pour binariser y_score.
    model_name : str
        Nom du modèle pour l'affichage.
    verbose : bool
        Affiche les résultats.

    Returns
    -------
    dict
        Toutes les métriques en un dictionnaire.
    """
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, y_score, target_fpr=0.05),
        "recall_at_10pct_fpr": recall_at_fpr(y_true, y_score, target_fpr=0.10),
        "precision_at_75pct_recall": precision_at_recall(y_true, y_score, target_recall=0.75),
        "f1_fraud": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_fraud": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fraud": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "threshold": threshold,
    }

    # Matrice de confusion détaillée
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    })

    if verbose:
        print(f"\n{'='*60}")
        print(f"Évaluation : {model_name}")
        print(f"{'='*60}")
        print(f"PR-AUC                       : {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC                      : {metrics['roc_auc']:.4f}")
        print(f"Recall @ 5% FPR              : {metrics['recall_at_5pct_fpr']:.4f}")
        print(f"Recall @ 10% FPR             : {metrics['recall_at_10pct_fpr']:.4f}")
        print(f"Precision @ 75% Recall       : {metrics['precision_at_75pct_recall']:.4f}")
        print(f"F1-Score (fraude)            : {metrics['f1_fraud']:.4f}")
        print(f"Confusion Matrix             :")
        print(f"  TN={tn:>7,}  FP={fp:>6,}")
        print(f"  FN={fn:>7,}  TP={tp:>6,}")
        print(f"{'='*60}\n")

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: str = "f1",
    target_fpr: float = 0.05,
) -> float:
    """
    Trouve le seuil de décision optimal selon un critère donné.

    Parameters
    ----------
    y_true : array
    y_score : array
    criterion : {"f1", "fpr_constraint"}
        - "f1" : seuil qui maximise le F1-score
        - "fpr_constraint" : plus petit seuil respectant FPR <= target_fpr
    target_fpr : float
        Utilisé seulement si criterion="fpr_constraint".

    Returns
    -------
    float
        Seuil optimal.
    """
    if criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = f1_scores.argmax()
        # thresholds a une longueur de N-1 par rapport à precision/recall
        if best_idx >= len(thresholds):
            return float(thresholds[-1])
        return float(thresholds[best_idx])

    elif criterion == "fpr_constraint":
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        valid_idx = fpr <= target_fpr
        if not valid_idx.any():
            return 0.5
        # Parmi les seuils valides, prendre celui qui maximise le TPR
        best_idx = tpr[valid_idx].argmax()
        valid_thresholds = thresholds[valid_idx]
        return float(valid_thresholds[best_idx])

    else:
        raise ValueError(f"Critère inconnu : {criterion}")


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Construit un tableau comparatif synthétique de plusieurs modèles.

    Parameters
    ----------
    metrics_list : list of dict
        Liste de dictionnaires retournés par evaluate_model().

    Returns
    -------
    pd.DataFrame
        Tableau trié par PR-AUC décroissant.
    """
    cols_to_keep = [
        "model", "pr_auc", "roc_auc", "recall_at_5pct_fpr",
        "f1_fraud", "precision_fraud", "recall_fraud",
    ]
    df = pd.DataFrame(metrics_list)[cols_to_keep]
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
