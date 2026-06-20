# Data

Les fichiers de données sont **exclus du dépôt Git** (volume > 100 Mo).

## Téléchargement du dataset BAF

### Option 1 — Via l'API Kaggle (recommandée)

```bash
# Depuis la racine du projet
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/ --unzip
```

### Option 2 — Téléchargement manuel

1. Aller sur [kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
2. Télécharger les 6 fichiers CSV
3. Les placer dans ce dossier `data/`

## Fichiers attendus

Après téléchargement, ce dossier doit contenir :

```
data/
├── Base.csv             # 1 000 000 lignes — dataset principal
├── Variant I.csv        # Variante I — déséquilibre de groupes
├── Variant II.csv       # Variante II — disparité de prévalence
├── Variant III.csv      # Variante III — biais de séparabilité
├── Variant IV.csv       # Variante IV — dérive temporelle prévalence
└── Variant V.csv        # Variante V — dérive temporelle séparabilité
```

Taille totale : ~1 Go

## Description des données

Voir le notebook `notebooks/01_exploratory_data_analysis.ipynb` pour le dictionnaire complet des 32 variables.

## Source

Jesus, S., Pombal, J., Alves, D., Cruz, A. F., Saleiro, P., Ribeiro, R. P., Gama, J., & Bizarro, P. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*. Advances in Neural Information Processing Systems 35 (NeurIPS 2022).
