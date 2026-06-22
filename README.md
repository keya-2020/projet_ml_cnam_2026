
## Installation

### Prérequis
- Python 3.10 ou supérieur
- 8 Go de RAM minimum (16 Go recommandés pour l'entraînement complet)
- Compte Kaggle pour télécharger le dataset

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/keya-2020/projet_ml_cnam_2026.git
cd fraud-detection-baf

# Créer un environnement virtuel
python -m venv venv
Sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset (nécessite l'API Kaggle configurée)
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/ --unzip
---

## Utilisation

### Reproduction complète des résultats

Exécuter les notebooks dans l'ordre :

```bash
jupyter lab notebooks/
```

1. **01_exploratory_data_analysis.ipynb** — Compréhension du dataset, valeurs manquantes, déséquilibre, dérive temporelle
2. **02_preprocessing_feature_engineering.ipynb** — Split temporel, encodage, gestion du déséquilibre, features dérivées
3. **03_baseline_models.ipynb** — Dummy classifier et régression logistique (références)
4. **04_unsupervised_models.ipynb** — Isolation Forest (détection sans labels)
5. **05_supervised_models.ipynb** — Random Forest, XGBoost, LightGBM avec interprétabilité SHAP
6. **06_deep_learning.ipynb** — Autoencoder et MLP avec TensorFlow/Keras
7. **07_stacking_ensemble.ipynb** — Méthode d'ensemble combinant les meilleurs modèles
8. **08_fairness_analysis.ipynb** — Analyse FPR par groupe d'âge et de revenu

## Méthodologie

### 1. Analyse exploratoire (EDA)

L'EDA répond à 4 questions fondamentales avant toute modélisation :

- **Quel est le degré de déséquilibre des classes ?** → ~1 % de fraudes → métriques PR-AUC et Recall@FPR
- **Y a-t-il une dérive temporelle ?** → Oui, taux de fraude croissant sur 8 mois → split temporel obligatoire
- **Quels biais structurels existent ?** → Taux de fraude croissant avec l'âge → analyse fairness nécessaire

### 2. Préparation des données

- **Valeurs manquantes** : 6 variables codent les NaN par `-1`, conversion explicite obligatoire
- **Split temporel** : mois 0-5 pour l'entraînement, mois 6-7 pour le test (évite le data leakage)
- **Encodage** : LabelEncoder pour les modèles à base d'arbres, One-Hot pour les modèles linéaires
- **Gestion du déséquilibre** : trois stratégies comparées (SMOTE, scale_pos_weight, threshold tuning)

### 3. Modélisation en entonnoir

Progression méthodique du simple au complexe :

```
Dummy → Logistic Regression → Isolation Forest → Random Forest → 
XGBoost/LightGBM → Autoencoder/MLP → Stacking
```

Chaque modèle doit justifier sa complexité supplémentaire par une amélioration mesurée de PR-AUC.

### 4. Interprétabilité

Les modèles boîte noire (XGBoost, MLP) sont accompagnés de **SHAP values** pour :
- Comprendre les features qui déclenchent une alerte (exigence RGPD)
- Identifier les biais éventuels (variables sensibles utilisées implicitement)
- Calibrer le seuil de décision selon le contexte opérationnel

### 5. Évaluation de l'équité

L'analyse fairness compare le **taux de faux positifs (FPR)** entre groupes :
- Tranche d'âge : 20-30, 30-40, 40-50, 50-60, 60+
- Niveau de revenu : déciles
- Statut professionnel : 7 catégories

Un modèle équitable produit un FPR similaire entre groupes (critère d'Equal Opportunity).

---

### Métriques principales

| Métrique | Définition | Pourquoi |
|---|---|---|
| **PR-AUC** | Aire sous la courbe Précision-Rappel | Insensible au déséquilibre — métrique de référence |
| **Recall @ 5% FPR** | Taux de détection à 5 % de fausses alertes | Métrique opérationnelle bancaire |
| **AUC-ROC** | Aire sous la courbe ROC | Comparaison entre modèles |
| **F1-Score (classe 1)** | Moyenne harmonique précision/rappel sur les fraudes | Compromis global |

### Pourquoi pas l'accuracy ?

Un modèle prédisant toujours "légitime" atteint 99 % d'accuracy sur ce dataset. Cette métrique est donc trompeuse et inutilisable en contexte déséquilibré.

---

## Stack technique

| Catégorie | Technologies |
|---|---|
| Manipulation des données | pandas, numpy |
| Machine Learning classique | scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow, Keras |
| Gestion du déséquilibre | imbalanced-learn (SMOTE) |
| Interprétabilité | SHAP |
| Visualisation | matplotlib, seaborn, plotly |
| Environnement | Jupyter Lab |

---

## Limites du projet

| Limite | Mitigation |
|---|---|
| Données synthétiques (CTGAN) | Conclusions à valider sur données réelles avant production |
| Fraude à l'ouverture uniquement | Périmètre explicite — pas la fraude transactionnelle |
| Déséquilibre persistant | Métriques adaptées (PR-AUC) + tests fairness |
| Boîte noire Deep Learning | SHAP sur XGBoost comme couche explicable |
| Dérive non anticipée | Monitoring continu requis en production |

---

## Pour aller plus loin

- Implémenter un **Variational Autoencoder (VAE)** pour une détection plus robuste
- Tester un **TabTransformer** (état de l'art 2024 sur données tabulaires)
- Ajouter une **détection de dérive concept** en temps réel (ADWIN, KSWIN)
- Intégrer le **monitoring de production** avec MLflow et Evidently

---

## Sources et références

- **Paper original BAF** : Jesus et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. NeurIPS 2022. [arXiv:2211.13358](https://arxiv.org/abs/2211.13358)
- **Datasheet officiel** : [github.com/feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud)
- **Dataset Kaggle** : [kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- **SHAP** : Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NIPS 2017.
- **XGBoost** : Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.

## Auteurs

**Keyagnion Micaël SEA** — Master Finance de Marché CNAM/ESSEC
**Dax AU** — Master Finance de Marché CNAM/ESSEC

Projet de validation du cours GFN260 — Machine Learning — Année 2025–2026
