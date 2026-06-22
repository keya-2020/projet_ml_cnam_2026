"""
Visualisations standardisées du projet.

Toutes les fonctions de plotting réutilisables, avec un style cohérent
(palette coolwarm, taille A4-friendly, fond clair).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# Configuration globale
sns.set_style("whitegrid")
sns.set_palette("coolwarm")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


def plot_target_distribution(df: pd.DataFrame, target_col: str = "fraud_bool", save_path: str = None):
    """Distribution de la cible avec annotation explicite du déséquilibre."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100

    bars = ax.bar(["Légitime (0)", "Fraude (1)"], counts.values,
                   color=["#2E75B6", "#C00000"])
    for bar, count, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{count:,}\n({p:.2f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Distribution de {target_col} — Déséquilibre extrême ({pct[1]:.2f}% de fraudes)",
                 fontweight="bold")
    ax.set_ylabel("Nombre d'observations")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_temporal_drift(df: pd.DataFrame, target_col: str = "fraud_bool",
                        month_col: str = "month", save_path: str = None):
    """Évolution du taux de fraude et du volume par mois."""
    monthly = df.groupby(month_col).agg(
        fraud_rate=(target_col, "mean"),
        n_obs=(target_col, "size"),
        n_fraud=(target_col, "sum"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Taux de fraude
    axes[0].plot(monthly[month_col], monthly["fraud_rate"] * 100, marker="o",
                 color="#C00000", linewidth=2.5, markersize=8)
    axes[0].set_xlabel("Mois")
    axes[0].set_ylabel("Taux de fraude (%)")
    axes[0].set_title("Dérive temporelle du taux de fraude")
    axes[0].grid(True, alpha=0.3)

    # Volume
    axes[1].bar(monthly[month_col], monthly["n_obs"], color="#2E75B6", alpha=0.7,
                label="Volume total")
    ax2 = axes[1].twinx()
    ax2.plot(monthly[month_col], monthly["n_fraud"], marker="s",
             color="#C00000", linewidth=2, label="Nb absolu de fraudes")
    axes[1].set_xlabel("Mois")
    axes[1].set_ylabel("Volume de demandes", color="#2E75B6")
    ax2.set_ylabel("Nb absolu de fraudes", color="#C00000")
    axes[1].set_title("Volume mensuel et nombre absolu de fraudes")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_by_class(df: pd.DataFrame, feature: str, target_col: str = "fraud_bool",
                          kind: str = "boxenplot", log_scale: bool = False,
                          save_path: str = None):
    """Distribution d'une feature par classe."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if kind == "boxenplot":
        sns.boxenplot(data=df, x=target_col, y=feature, hue=target_col, ax=ax,
                      palette=["#2E75B6", "#C00000"], legend=False)
    elif kind == "violin":
        sns.violinplot(data=df, x=target_col, y=feature, hue=target_col, ax=ax,
                       palette=["#2E75B6", "#C00000"], legend=False)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Statut (0=Légitime, 1=Fraude)")
    ax.set_title(f"Distribution de '{feature}' par classe", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_with_target(df: pd.DataFrame, target_col: str = "fraud_bool",
                                  top_n: int = 15, save_path: str = None):
    """Top N features par corrélation absolue avec la cible."""
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    corr_abs = corr.abs().sort_values(ascending=False).head(top_n)
    corr_signed = corr[corr_abs.index]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = ["#C00000" if v > 0 else "#2E75B6" for v in corr_signed.values]
    ax.barh(corr_signed.index[::-1], corr_signed.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Coefficient de corrélation")
    ax.set_title(f"Top {top_n} features corrélées avec '{target_col}'", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curves(models_scores: Dict[str, tuple], y_true: np.ndarray,
                                  save_path: str = None):
    """
    Trace les courbes Precision-Recall pour plusieurs modèles.

    Parameters
    ----------
    models_scores : dict
        {model_name: y_score_array}
    y_true : array
        Vraies étiquettes.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, y_score in models_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline aléatoire ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall — Comparaison des modèles", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(models_scores: Dict[str, tuple], y_true: np.ndarray,
                    save_path: str = None):
    """Trace les courbes ROC pour plusieurs modèles."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, y_score in models_scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
            label="Classifieur aléatoire")
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR / Recall)")
    ax.set_title("Courbes ROC — Comparaison des modèles", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_fairness_metrics(fairness_df: pd.DataFrame, attr_name: str,
                          metric: str = "fpr", save_path: str = None):
    """Trace les métriques d'équité par groupe."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(fairness_df[attr_name].astype(str), fairness_df[metric],
                   color="#2E75B6", alpha=0.8)
    for bar, val in zip(bars, fairness_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(attr_name)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} par {attr_name} — Analyse d'équité",
                 fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()